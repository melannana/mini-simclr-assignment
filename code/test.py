"""
test.py
------------------------------------
Evaluate Mini-SimCLR Linear Probe
"""

import argparse
import numpy as np
import torch
import torch.nn as nn

from data_utils import get_test_loader
from models import Encoder
from linear_probe import LinearClassifier


classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def load_models(args, device):

    encoder = Encoder(args.backbone)

    ckpt = torch.load(
        args.encoder,
        map_location=device
    )

    encoder.load_state_dict(
        ckpt["encoder_state_dict"]
    )

    encoder.to(device)
    encoder.eval()

    classifier = LinearClassifier(
        encoder.out_dim,
        10
    )

    cls = torch.load(
        args.classifier,
        map_location=device
    )

    classifier.load_state_dict(
        cls["classifier"]
    )

    classifier.to(device)
    classifier.eval()

    return encoder, classifier


@torch.no_grad()
def evaluate(
        encoder,
        classifier,
        loader,
        device):

    total = 0
    correct = 0

    class_correct = np.zeros(10)
    class_total = np.zeros(10)

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        features = encoder(images)

        outputs = classifier(features)

        pred = outputs.argmax(1)

        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        for i in range(labels.size(0)):

            label = labels[i].item()

            class_total[label] += 1

            if pred[i] == labels[i]:

                class_correct[label] += 1

    acc = 100 * correct / total

    print("=" * 60)
    print("Overall Accuracy")
    print("=" * 60)
    print("{:.2f}%".format(acc))

    print()

    print("=" * 60)
    print("Per Class Accuracy")
    print("=" * 60)

    for i in range(10):

        print(
            "{:<10s}: {:.2f}%".format(
                classes[i],
                100 * class_correct[i] / class_total[i]
            )
        )


def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    loader = get_test_loader(
        batch_size=args.batch_size
    )

    encoder, classifier = load_models(
        args,
        device
    )

    evaluate(
        encoder,
        classifier,
        loader,
        device
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int
    )

    parser.add_argument(
        "--backbone",
        default="resnet18"
    )

    parser.add_argument(
        "--encoder",
        default="results/pretrain_model.pth"
    )

    parser.add_argument(
        "--classifier",
        default="results/linear_probe.pth"
    )

    args = parser.parse_args()

    main(args)