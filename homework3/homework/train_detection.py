import argparse
from datetime import datetime
from pathlib import Path
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
import numpy as np
from .models import load_model, save_model
from homework.datasets.road_dataset import load_data
from torch.nn import functional as F


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load model
  
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # Load data
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # Create loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        correct_train = 0
        total_train = 0
        total_train_loss = 0

        # Training step
        for data in train_data:
            images, depth, labels = data["image"].to(device), data["depth"].to(device), data["label"].to(device)

            optimizer.zero_grad()  # Zero gradients from previous step

            # Forward pass
            logits, raw_depth = model(images)

            # Calculate loss (you may use a combination of classification loss and depth regression loss)
            classification_loss = F.cross_entropy(logits, labels)
            depth_loss = F.mse_loss(raw_depth, depth)
            loss = classification_loss + depth_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track accuracy
            _, predicted = torch.max(logits, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            total_train_loss += loss.item()

            global_step += 1

        train_acc = correct_train / total_train
        train_loss = total_train_loss / global_step
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)

        # Disable gradient computation for evaluation
        with torch.no_grad():
            model.eval()

            correct_val = 0
            total_val = 0
            total_val_loss = 0

            # Validation step
            for data in val_data:
                images, depth, labels = data["image"].to(device), data["depth"].to(device), data["label"].to(device)

                logits, raw_depth = model(images)

                # Calculate loss
                classification_loss = F.cross_entropy(logits, labels)
                depth_loss = F.mse_loss(raw_depth, depth)
                loss = classification_loss + depth_loss

                # Track accuracy
                _, predicted = torch.max(logits, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                total_val_loss += loss.item()

            val_acc = correct_val / total_val
            val_loss = total_val_loss / len(val_data)
            metrics["val_loss"].append(val_loss)
            metrics["val_acc"].append(val_acc)

        # Log average train and val accuracy and loss to tensorboard
        epoch_train_acc = np.mean(metrics["train_acc"])
        epoch_val_acc = np.mean(metrics["val_acc"])
        epoch_train_loss = np.mean(metrics["train_loss"])
        epoch_val_loss = np.mean(metrics["val_loss"])

        logger.add_scalar("train/accuracy", epoch_train_acc, epoch)
        logger.add_scalar("val/accuracy", epoch_val_acc, epoch)
        logger.add_scalar("train/loss", epoch_train_loss, epoch)
        logger.add_scalar("val/loss", epoch_val_loss, epoch)

        # Print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_loss={epoch_train_loss:.4f} "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_loss={epoch_val_loss:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # Save model
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
