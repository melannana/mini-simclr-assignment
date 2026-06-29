"""
utils.py
-----------------------------------------
Common utilities for Mini-SimCLR
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch


############################################################
# Random Seed
############################################################

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


############################################################
# Make Directory
############################################################

def make_dirs():

    os.makedirs("logs", exist_ok=True)

    os.makedirs("results", exist_ok=True)

    os.makedirs("report", exist_ok=True)

    os.makedirs("checkpoints", exist_ok=True)


############################################################
# Save Checkpoint
############################################################

def save_checkpoint(
        state,
        filename):

    torch.save(state, filename)


############################################################
# Load Checkpoint
############################################################

def load_checkpoint(
        filename,
        device):

    checkpoint = torch.load(
        filename,
        map_location=device
    )

    return checkpoint


############################################################
# Save Loss Curve
############################################################

def plot_loss_curve(
        losses,
        save_path):

    plt.figure(figsize=(7,5))

    plt.plot(
        range(1,len(losses)+1),
        losses,
        marker="o",
        linewidth=2
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Contrastive Loss")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


############################################################
# Save Accuracy Curve
############################################################

def plot_accuracy_curve(
        train_acc,
        test_acc,
        save_path):

    plt.figure(figsize=(7,5))

    plt.plot(
        train_acc,
        label="Train"
    )

    plt.plot(
        test_acc,
        label="Test"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


############################################################
# Count Parameters
############################################################

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


############################################################
# Save Log
############################################################

def write_log(
        file,
        text):

    print(text)

    file.write(text+"\n")

    file.flush()