import os
import sys
from abc import ABC, abstractmethod

from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Add the 'scripts' directory to Python Path (project root access for imports)
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

# Project-specific helper functions
from gnn.help_functions import validate_model_during_training, LinearWarmupCosineDecayScheduler


class BaseGNN(nn.Module, ABC):
    # (nn comes from torch.nn, see import statement above.  nn.Module = "base class for all neural network modules".)
    # (ABC is the python way to program abstract classes; in some sense we obtain from ABC the missing functionality.)

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.3,
                 use_dropout: bool = False,
                 predict_mode_stats: bool = False,
                 dtype: torch.dtype = torch.float32,
                 log_to_wandb: bool = False):
        """
        Base class for all GNN implementations.
        Defines shared hyperparameters and interface.
        """
        super().__init__()

        # Model dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Regularization
        self.dropout = dropout
        self.use_dropout = use_dropout

        # Multi-task / auxiliary prediction flag
        self.predict_mode_stats = predict_mode_stats

        # Numerical precision
        self.dtype = dtype

        # Logging configuration
        self.log_to_wandb = log_to_wandb

    @abstractmethod
    def define_layers(self):
        """
        Must be implemented in subclass:
        defines neural network architecture.
        """
        pass

    @abstractmethod
    def forward(self, data):
        """
        Must be implemented in subclass:
        forward pass of the model.
        """
        pass

    def initialize_weights(self):
        """
        Default weight initialization for Linear layers.
        Can be overridden in subclasses.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization (good for ReLU-like activations)
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def train_model(self,
                    config: object = None,
                    loss_fct: nn.Module = None,
                    optimizer: optim.Optimizer = None,
                    trainingData_dl: DataLoader = None,
                    validationData_dl: DataLoader = None,
                    device: torch.device = None,
                    early_stopping: object = None,
                    model_save_path: str = None,
                    scalers_train: dict = None,
                    scalers_validation: dict = None) -> tuple:
        """
        Full training loop for GNN models.
        Includes training, validation, logging, checkpointing, and early stopping.
        """

        # Safety check: config is mandatory
        if config is None:
            raise ValueError("Config cannot be None")

        # Mixed precision gradient scaler (stabilizes FP16 training)
        scaler = GradScaler()

        # Total number of optimization steps (for LR scheduler)
        total_steps = config.num_epochs * len(trainingData_dl)

        # Learning rate scheduler (warmup + cosine decay)
        scheduler = LinearWarmupCosineDecayScheduler(
            initial_lr=config.lr,
            total_steps=total_steps
        )

        # Track best validation loss
        best_val_loss = float('inf')

        # Directory for checkpoints
        checkpoint_dir = os.path.join(os.path.dirname(model_save_path), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Auxiliary loss for multi-task setup
        mode_stats_loss = nn.MSELoss().to(dtype=torch.float32).to(device)

        # Setup logging metrics for wandb
        from training.help_functions import setup_wandb_metrics
        setup_wandb_metrics(predict_mode_stats=config.predict_mode_stats)

        # -----------------------
        # Resume training logic
        # -----------------------
        if config.continue_training:

            checkpoint = torch.load(config.base_checkpoint_path)

            # Restore model + optimizer + scaler state
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            best_val_loss = checkpoint['best_val_loss']
            start_epoch = checkpoint['epoch'] + 1

            print(f"Resuming training from epoch {start_epoch} with best validation loss: {best_val_loss}")
        else:
            start_epoch = 0

        # =======================
        # Training loop (epochs)
        # =======================
        for epoch in range(start_epoch, config.num_epochs):

            self.train()  # set dropout/batchnorm to training mode

            optimizer.zero_grad()

            # Accumulators for epoch-level statistics
            epoch_train_loss = 0
            epoch_train_loss_node_predictions = 0
            epoch_train_loss_mode_stats = 0

            # =======================
            # Training loop (batches)
            # =======================
            for idx, data in tqdm(enumerate(trainingData_dl), total=len(trainingData_dl),
                                  desc=f"Epoch {epoch+1}/{config.num_epochs}"):

                # global training step index
                step = epoch * len(trainingData_dl) + idx

                # update learning rate per step
                lr = scheduler.get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # move batch to GPU/CPU device
                data = data.to(device)

                # target for node-level prediction task
                targets_node_predictions = data.y

                # inverse transform features for custom loss computations
                x_unscaled = scalers_train["x_scaler"].inverse_transform(
                    data.x.detach().clone().cpu().numpy()
                )

                # optional auxiliary target
                if config.predict_mode_stats:
                    targets_mode_stats = data.mode_stats

                # mixed precision forward pass
                with autocast():
                    # (autocast is a cuda function)

                    if config.predict_mode_stats:
                        # multi-output model
                        predicted, mode_stats_pred = self(data)

                        # main loss
                        train_loss_node_predictions = loss_fct(
                            predicted,
                            targets_node_predictions,
                            x_unscaled
                        )

                        # auxiliary loss
                        train_loss_mode_stats = mode_stats_loss(
                            mode_stats_pred,
                            targets_mode_stats
                        )

                        train_loss = train_loss_node_predictions + train_loss_mode_stats

                    else:
                        predicted = self(data)
                        train_loss = loss_fct(predicted, targets_node_predictions, x_unscaled)

                # accumulate loss stats
                epoch_train_loss += train_loss.item()

                if config.predict_mode_stats:
                    epoch_train_loss_node_predictions += train_loss_node_predictions.item()
                    epoch_train_loss_mode_stats += train_loss_mode_stats.item()

                # backward pass with gradient scaling (FP16 stability)
                scaler.scale(train_loss).backward()

                # optional gradient clipping
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                # gradient accumulation step
                if (idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # batch-level logging to wandb
                if config.predict_mode_stats:
                    wandb.log({
                        "batch_train_loss": train_loss.item(),
                        "batch_train_loss-node_predictions": train_loss_node_predictions.item(),
                        "batch_train_loss-mode_stats": train_loss_mode_stats.item(),
                        "batch_step": step
                    })
                else:
                    wandb.log({
                        "batch_train_loss": train_loss.item(),
                        "batch_step": step
                    })

            # flush remaining gradients if accumulation not aligned
            if len(trainingData_dl) % config.gradient_accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # -----------------------
            # Validation phase
            # -----------------------
            if config.predict_mode_stats:

                val_loss, r_squared, spearman_corr, pearson_corr, \
                    val_loss_node_predictions, val_loss_mode_stats = validate_model_during_training(
                    config=config,
                    model=self,
                    dataset=validationData_dl,
                    loss_func=loss_fct,
                    device=device,
                    scalers_validation=scalers_validation
                )

            else:

                val_loss, r_squared, spearman_corr, pearson_corr = validate_model_during_training(
                    config=config,
                    model=self,
                    dataset=validationData_dl,
                    loss_func=loss_fct,
                    device=device,
                    scalers_validation=scalers_validation
                )

            # epoch-level logging
            wandb.log({
                "val_loss": val_loss,
                "train_loss": epoch_train_loss / len(trainingData_dl),
                "lr": lr,
                "r^2": r_squared,
                "spearman": spearman_corr,
                "pearson": pearson_corr,
                "epoch": epoch
            })

            print(f"epoch: {epoch}, validation loss: {val_loss}, lr: {lr}, r^2: {r_squared}")

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if model_save_path:
                    torch.save(self.state_dict(), model_save_path)
                    print(f"Best model saved to {model_save_path}")

            # periodic checkpointing
            if epoch % 20 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)

                print(f"Checkpoint saved: {checkpoint_path}")

            # early stopping check
            early_stopping(val_loss)

            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # final summary
        print("Best validation loss: ", best_val_loss)
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.finish()

        return val_loss, epoch