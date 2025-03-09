import argparse
from datetime import datetime
from pathlib import Path
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
import numpy as np

from homework.models import ClassificationLoss, load_model, save_model
from homework.utils import load_data


def train(exp_dir="logs", model_name="classifier", num_epoch=50, lr=1e-3, batch_size=128, seed=2024, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs).to(device)
    model.train()

    train_data = load_data("classification_data/train", transform_pipeline="aug", batch_size=batch_size, shuffle=True)
    val_data = load_data("classification_data/val", transform_pipeline="default", batch_size=batch_size, shuffle=False)

    loss_func = ClassificationLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    best_val_acc = 0.0

    for epoch in range(num_epoch):
        model.train()
        correct_train, total_train = 0, 0

        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            logits = model(img)
            loss = loss_func(logits, label)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(logits, 1)
            correct_train += (predicted == label).sum().item()
            total_train += label.size(0)

            global_step += 1

        train_acc = correct_train / total_train

        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                logits = model(img)
                _, predicted = torch.max(logits, 1)
                correct_val += (predicted == label).sum().item()
                total_val += label.size(0)

        val_acc = correct_val / total_val
        logger.add_scalar("train/accuracy", train_acc, epoch)
        logger.add_scalar("val/accuracy", val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model)
            torch.save(model.state_dict(), log_dir / f"{model_name}_best.pth")

        print(f"Epoch {epoch + 1}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))
