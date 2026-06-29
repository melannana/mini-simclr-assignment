import torch
import torch.nn as nn
import torch.nn.functional as F

############################################################
# BasicBlock
############################################################

class BasicBlock(nn.Module):

    expansion=1

    def __init__(self,in_planes,planes,stride=1):

        super().__init__()

        self.conv1=nn.Conv2d(
            in_planes,
            planes,
            3,
            stride,
            1,
            bias=False
        )

        self.bn1=nn.BatchNorm2d(planes)

        self.conv2=nn.Conv2d(
            planes,
            planes,
            3,
            1,
            1,
            bias=False
        )

        self.bn2=nn.BatchNorm2d(planes)

        self.shortcut=nn.Sequential()

        if stride!=1 or in_planes!=planes:

            self.shortcut=nn.Sequential(

                nn.Conv2d(
                    in_planes,
                    planes,
                    1,
                    stride,
                    bias=False
                ),

                nn.BatchNorm2d(planes)

            )

    def forward(self,x):

        out=F.relu(self.bn1(self.conv1(x)))

        out=self.bn2(self.conv2(out))

        out+=self.shortcut(x)

        out=F.relu(out)

        return out


############################################################
# ResNet18
############################################################

class ResNet18_CIFAR(nn.Module):

    def __init__(self):

        super().__init__()

        self.in_planes=64

        self.conv1=nn.Conv2d(
            3,
            64,
            3,
            1,
            1,
            bias=False
        )

        self.bn1=nn.BatchNorm2d(64)

        self.layer1=self.make_layer(64,2,1)

        self.layer2=self.make_layer(128,2,2)

        self.layer3=self.make_layer(256,2,2)

        self.layer4=self.make_layer(512,2,2)

        self.pool=nn.AdaptiveAvgPool2d((1,1))

    def make_layer(self,planes,num_blocks,stride):

        strides=[stride]+[1]*(num_blocks-1)

        layers=[]

        for s in strides:

            layers.append(
                BasicBlock(
                    self.in_planes,
                    planes,
                    s
                )
            )

            self.in_planes=planes

        return nn.Sequential(*layers)

    def forward(self,x):

        x=F.relu(self.bn1(self.conv1(x)))

        x=self.layer1(x)

        x=self.layer2(x)

        x=self.layer3(x)

        x=self.layer4(x)

        x=self.pool(x)

        x=torch.flatten(x,1)

        return x


############################################################
# Small CNN
############################################################

class SmallCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder=nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))

        )

    def forward(self,x):

        x=self.encoder(x)

        return torch.flatten(x,1)


############################################################
# Encoder
############################################################

class Encoder(nn.Module):

    def __init__(self,backbone="resnet18"):

        super().__init__()

        if backbone=="smallcnn":

            self.encoder=SmallCNN()

            self.out_dim=128

        else:

            self.encoder=ResNet18_CIFAR()

            self.out_dim=512

    def forward(self,x):

        return self.encoder(x)


############################################################
# Projection Head
############################################################

class ProjectionHead(nn.Module):

    def __init__(
            self,
            in_dim=512,
            hidden_dim=512,
            out_dim=128):

        super().__init__()

        self.mlp=nn.Sequential(

            nn.Linear(in_dim,hidden_dim),

            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim,out_dim)

        )

    def forward(self,x):

        z=self.mlp(x)

        z=F.normalize(z,dim=1)

        return z


############################################################
# SimCLR
############################################################

class SimCLR(nn.Module):

    def __init__(self,backbone="resnet18"):

        super().__init__()

        self.encoder=Encoder(backbone)

        self.projector=ProjectionHead(
            self.encoder.out_dim,
            512,
            128
        )

    def forward(self,x):

        h=self.encoder(x)

        z=self.projector(h)

        return h,z