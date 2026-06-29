"""
pretrain.py
----------------------------------------------------
Mini-SimCLR Self-Supervised Pretraining

Workflow

CIFAR10
        ↓
Two Augmented Views
        ↓
Shared Encoder
        ↓
Projection Head
        ↓
NT-Xent Loss
        ↓
Save Encoder
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from tqdm import tqdm

from data_utils import (
    get_pretrain_loader,
    set_seed
)

from models import SimCLR
from losses import NTXentLoss


##############################################################
# Train One Epoch
##############################################################

def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device):

    model.train()

    running_loss = 0.0

    pbar = tqdm(loader)

    for (x1, x2), _ in pbar:

        x1 = x1.to(device)

        x2 = x2.to(device)

        _, z1 = model(x1)

        _, z2 = model(x2)

        loss = criterion(z1, z2)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        pbar.set_description(
            "Loss {:.4f}".format(loss.item())
        )

    epoch_loss = running_loss / len(loader)

    return epoch_loss


##############################################################
# Main
##############################################################

def main(args):

    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Mini-SimCLR Pretraining")
    print("=" * 60)

    print("Device :", device)

    ##########################################################

    os.makedirs("logs", exist_ok=True)

    os.makedirs("results", exist_ok=True)

    os.makedirs("report", exist_ok=True)

    ##########################################################

    train_loader, indices = get_pretrain_loader(

        train_size=args.train_size,

        batch_size=args.batch_size

    )

    ##########################################################

    model = SimCLR(
        backbone=args.backbone
    ).to(device)

    ##########################################################

    criterion = NTXentLoss(
        temperature=args.temperature
    )

    ##########################################################

    optimizer = optim.Adam(

        model.parameters(),

        lr=args.lr,

        weight_decay=1e-6

    )

    ##########################################################

    history = []

    best_loss = 999

    start = time.time()

    ##########################################################

    log_path = os.path.join(
        "logs",
        "pretrain_log.txt"
    )

    logfile = open(log_path, "w")

    logfile.write("Mini SimCLR Pretraining\n")

    logfile.write("=" * 50 + "\n")

    ##########################################################

    for epoch in range(args.epochs):

        print()

        print("-" * 50)

        print(
            "Epoch {}/{}".format(
                epoch + 1,
                args.epochs
            )
        )

        print("-" * 50)

        loss = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            device

        )

        history.append(loss)

        print("Average Loss : {:.4f}".format(loss))

        logfile.write(

            "Epoch {:02d} Loss {:.6f}\n".format(

                epoch + 1,

                loss

            )

        )

        ######################################################

        if loss < best_loss:

            best_loss = loss

            checkpoint = {

                "epoch": epoch + 1,

                "encoder_state_dict":
                    model.encoder.state_dict(),

                "projection_state_dict":
                    model.projector.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "loss": loss,

                "args": vars(args)

            }

            torch.save(

                checkpoint,

                "results/pretrain_model.pth"

            )

            print("Best model saved.")

    ##########################################################

    logfile.close()

    ##########################################################

    np.save(

        "results/train_indices.npy",

        indices

    )

    ##########################################################

    end = time.time()

    train_time = end - start

    ##########################################################

    print()

    print("=" * 60)

    print("Training Finished")

    print("=" * 60)

    print("Best Loss : {:.4f}".format(best_loss))

    print("Time : {:.2f} min".format(train_time / 60))

    ##########################################################

    plt.figure(figsize=(7, 5))

    plt.plot(

        range(1, args.epochs + 1),

        history,

        marker="o",

        linewidth=2

    )

    plt.xlabel("Epoch")

    plt.ylabel("NT-Xent Loss")

    plt.title("Mini SimCLR Pretraining")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        "report/loss_curve.png",

        dpi=300

    )

    plt.close()

    ##########################################################

    with open(

            "results/results.txt",

            "w") as f:

        f.write("Mini SimCLR\n")

        f.write("====================\n")

        f.write(

            "Train Images : {}\n".format(

                args.train_size

            )

        )

        f.write(

            "Epochs : {}\n".format(

                args.epochs

            )

        )

        f.write(

            "Batch Size : {}\n".format(

                args.batch_size

            )

        )

        f.write(

            "Temperature : {}\n".format(

                args.temperature

            )

        )

        f.write(

            "Learning Rate : {}\n".format(

                args.lr

            )

        )

        f.write(

            "Best Loss : {:.4f}\n".format(

                best_loss

            )

        )

        f.write(

            "Training Time : {:.2f} min\n".format(

                train_time / 60

            )

        )

    print()

    print("Loss Curve Saved")

    print("Model Saved")

    print("Training Log Saved")


##############################################################
# Entry
##############################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--train_size",

        type=int,

        default=3000

    )

    parser.add_argument(

        "--epochs",

        type=int,

        default=10

    )

    parser.add_argument(

        "--batch_size",

        type=int,

        default=64

    )

    parser.add_argument(

        "--lr",

        type=float,

        default=1e-3

    )

    parser.add_argument(

        "--temperature",

        type=float,

        default=0.5

    )

    parser.add_argument(

        "--backbone",

        type=str,

        default="resnet18",

        choices=[

            "resnet18",

            "smallcnn"

        ]

    )

    args = parser.parse_args()

    main(args)