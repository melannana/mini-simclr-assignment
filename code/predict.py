"""
predict.py
---------------------------------------
Show 5 prediction samples
"""

import random

import matplotlib.pyplot as plt
import torch

from torchvision.datasets import CIFAR10
from torchvision import transforms

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


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

########################################################

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914,0.4822,0.4465),
        (0.2023,0.1994,0.2010)
    )

])

dataset = CIFAR10(

    root="data",

    train=False,

    download=True,

    transform=transform

)

########################################################

encoder = Encoder("resnet18")

ckpt = torch.load(
    "results/pretrain_model.pth",
    map_location=device
)

encoder.load_state_dict(
    ckpt["encoder_state_dict"]
)

encoder.to(device)
encoder.eval()

########################################################

classifier = LinearClassifier(
    encoder.out_dim,
    10
)

cls = torch.load(
    "results/linear_probe.pth",
    map_location=device
)

classifier.load_state_dict(
    cls["classifier"]
)

classifier.to(device)
classifier.eval()

########################################################

mean = torch.tensor(
    [0.4914,0.4822,0.4465]
).view(3,1,1)

std = torch.tensor(
    [0.2023,0.1994,0.2010]
).view(3,1,1)

fig = plt.figure(figsize=(12,3))

indices = random.sample(
    range(len(dataset)),
    5
)

for i,index in enumerate(indices):

    img,label = dataset[index]

    with torch.no_grad():

        feature = encoder(
            img.unsqueeze(0).to(device)
        )

        pred = classifier(feature)

        pred = pred.argmax(1).item()

    image = img.cpu()*std + mean

    image = image.permute(1,2,0)

    ax = plt.subplot(1,5,i+1)

    ax.imshow(image)

    ax.axis("off")

    ax.set_title(

        "T:{}\nP:{}".format(

            classes[label],

            classes[pred]

        ),

        fontsize=9

    )

plt.tight_layout()

plt.savefig(

    "report/prediction_samples.png",

    dpi=300

)

plt.show()

print()

print("Prediction figure saved:")
print("report/prediction_samples.png")