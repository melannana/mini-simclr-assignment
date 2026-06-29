"""
linear_probe.py
----------------------------------------------------
Mini-SimCLR Linear Probe

Workflow

Pretrained Encoder (Frozen)
            │
            ▼
      Feature Vector
            │
            ▼
     Linear Classifier
            │
            ▼
 CrossEntropy Loss
"""

import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from data_utils import (
    get_linear_loader,
    get_test_loader,
    set_seed
)

from models import Encoder


##############################################################
# Linear Classifier
##############################################################

class LinearClassifier(nn.Module):
    """
    Linear Probe

    Encoder输出：
        512维（ResNet18）
        或128维（SmallCNN）

    输出：
        10类
    """

    def __init__(self, in_dim=512, num_classes=10):

        super().__init__()

        self.fc = nn.Linear(
            in_dim,
            num_classes
        )

    def forward(self, x):

        return self.fc(x)


##############################################################
# Load Frozen Encoder
##############################################################

def load_pretrained_encoder(
        checkpoint_path,
        backbone,
        device):
    """
    加载预训练Encoder，并冻结参数
    """

    encoder = Encoder(backbone=backbone)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    encoder.load_state_dict(
        checkpoint["encoder_state_dict"]
    )

    encoder.to(device)

    encoder.eval()

    # 冻结参数
    for param in encoder.parameters():
        param.requires_grad = False

    print("Loaded pretrained encoder.")

    return encoder


##############################################################
# Train One Epoch
##############################################################

def train_one_epoch(
        encoder,
        classifier,
        loader,
        criterion,
        optimizer,
        device):

    encoder.eval()

    classifier.train()

    running_loss = 0.0

    correct = 0

    total = 0

    pbar = tqdm(loader)

    for images, labels in pbar:

        images = images.to(device)

        labels = labels.to(device)

        # 不计算Encoder梯度
        with torch.no_grad():

            features = encoder(images)

        outputs = classifier(features)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

        acc = 100.0 * correct / total

        pbar.set_description(
            "Loss {:.4f}  Acc {:.2f}%".format(
                loss.item(),
                acc
            )
        )

    epoch_loss = running_loss / len(loader)

    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


##############################################################
# Evaluate
##############################################################

@torch.no_grad()
def evaluate(
        encoder,
        classifier,
        loader,
        criterion,
        device):
    """
    在测试集上评估
    """

    encoder.eval()

    classifier.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in loader:

        images = images.to(device)

        labels = labels.to(device)

        features = encoder(images)

        outputs = classifier(features)

        loss = criterion(outputs, labels)

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    loss = running_loss / len(loader)

    accuracy = 100.0 * correct / total

    return loss, accuracy
##############################################################
# Main
##############################################################

def main(args):

    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Mini-SimCLR Linear Probe")
    print("=" * 60)
    print("Device :", device)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    ##########################################################
    # Load Train Indices
    ##########################################################

    indices = np.load(
        "results/train_indices.npy"
    )

    train_loader = get_linear_loader(
        indices,
        batch_size=args.batch_size
    )

    test_loader = get_test_loader(
        batch_size=args.batch_size
    )

    ##########################################################
    # Load Frozen Encoder
    ##########################################################

    encoder = load_pretrained_encoder(
        args.checkpoint,
        args.backbone,
        device
    )

    ##########################################################
    # Build Linear Classifier
    ##########################################################

    feature_dim = encoder.out_dim

    classifier = LinearClassifier(
        feature_dim,
        10
    ).to(device)

    ##########################################################
    # Loss & Optimizer
    ##########################################################

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        classifier.parameters(),
        lr=args.lr
    )

    ##########################################################

    best_acc = 0.0

    log_file = open(
        "logs/linear_probe_log.txt",
        "w"
    )

    log_file.write("Mini SimCLR Linear Probe\n")
    log_file.write("=" * 50 + "\n")

    ##########################################################

    for epoch in range(args.epochs):

        print()

        print("-" * 60)

        print(
            "Epoch {}/{}".format(
                epoch + 1,
                args.epochs
            )
        )

        print("-" * 60)

        train_loss, train_acc = train_one_epoch(

            encoder,

            classifier,

            train_loader,

            criterion,

            optimizer,

            device

        )

        test_loss, test_acc = evaluate(

            encoder,

            classifier,

            test_loader,

            criterion,

            device

        )

        print(
            "Train Loss : {:.4f}".format(
                train_loss
            )
        )

        print(
            "Train Acc  : {:.2f}%".format(
                train_acc
            )
        )

        print(
            "Test  Loss : {:.4f}".format(
                test_loss
            )
        )

        print(
            "Test  Acc  : {:.2f}%".format(
                test_acc
            )
        )

        ######################################################

        log_file.write(

            "Epoch {:02d} "

            "TrainLoss {:.4f} "

            "TrainAcc {:.2f} "

            "TestLoss {:.4f} "

            "TestAcc {:.2f}\n".format(

                epoch + 1,

                train_loss,

                train_acc,

                test_loss,

                test_acc

            )

        )

        ######################################################

        if test_acc > best_acc:

            best_acc = test_acc

            torch.save(

                {

                    "classifier":

                        classifier.state_dict(),

                    "accuracy":

                        best_acc

                },

                "results/linear_probe.pth"

            )

            print("Best classifier saved.")

    ##########################################################

    log_file.close()

    ##########################################################

    print()

    print("=" * 60)

    print("Linear Probe Finished")

    print("=" * 60)

    print(
        "Best Test Accuracy : {:.2f}%".format(
            best_acc
        )
    )

    ##########################################################
    # Save Results
    ##########################################################

    with open(
            "results/results.txt",
            "a") as f:

        f.write("\n")

        f.write("Linear Probe\n")

        f.write("=========================\n")

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
            "Learning Rate : {}\n".format(
                args.lr
            )
        )

        f.write(
            "Best Accuracy : {:.2f}%\n".format(
                best_acc
            )
        )

    print("Results Saved.")

##############################################################
# Entry
##############################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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
        "--backbone",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "smallcnn"
        ]
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/pretrain_model.pth"
    )

    args = parser.parse_args()

    main(args)