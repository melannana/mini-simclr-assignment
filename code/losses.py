"""
losses.py
---------------------------------------------------------
Mini-SimCLR NT-Xent Loss

Reference:
A Simple Framework for Contrastive Learning
of Visual Representations (SimCLR)

https://arxiv.org/abs/2002.05709
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)

    输入:
        z1 : (N, D)
        z2 : (N, D)

    输出:
        scalar loss
    """

    def __init__(self, temperature=0.5):
        super().__init__()

        self.temperature = temperature

    def forward(self, z1, z2):

        batch_size = z1.size(0)

        device = z1.device

        ####################################################
        # Step 1
        # 拼接两个 View
        ####################################################

        z = torch.cat([z1, z2], dim=0)

        # (2N,D)

        ####################################################
        # Step 2
        # L2 Normalize
        ####################################################

        z = F.normalize(z, dim=1)

        ####################################################
        # Step 3
        # Cosine Similarity Matrix
        ####################################################

        similarity = torch.matmul(z, z.T)

        similarity = similarity / self.temperature

        ####################################################
        # Step 4
        # 去除自己
        ####################################################

        mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=device
        )

        similarity = similarity.masked_fill(
            mask,
            -1e9
        )

        ####################################################
        # Step 5
        # Positive Pair
        #
        # 前 N 个样本对应后 N 个
        # 后 N 个对应前 N 个
        ####################################################

        positive = torch.cat([

            torch.diag(
                similarity,
                batch_size
            ),

            torch.diag(
                similarity,
                -batch_size
            )

        ], dim=0)

        ####################################################
        # Step 6
        # Denominator
        #
        # log(sum(exp(sim)))
        ####################################################

        denominator = torch.logsumexp(

            similarity,

            dim=1

        )

        ####################################################
        # Step 7
        #
        # L = -(positive-logsumexp)
        ####################################################

        loss = -(

            positive

            -

            denominator

        )

        return loss.mean()


##############################################################
# Functional API
##############################################################

def nt_xent_loss(
        z1,
        z2,
        temperature=0.5):

    criterion = NTXentLoss(
        temperature=temperature
    )

    return criterion(
        z1,
        z2
    )


##############################################################
# Unit Test
##############################################################

if __name__ == "__main__":

    torch.manual_seed(42)

    batch_size = 8

    feature_dim = 128

    z1 = torch.randn(
        batch_size,
        feature_dim
    )

    z2 = torch.randn(
        batch_size,
        feature_dim
    )

    criterion = NTXentLoss(
        temperature=0.5
    )

    loss = criterion(
        z1,
        z2
    )

    print("=" * 50)
    print("NT-Xent Loss Test")
    print("=" * 50)
    print("Batch Size :", batch_size)
    print("Feature Dim:", feature_dim)
    print("Temperature:", 0.5)
    print("Loss:", loss.item())