"""
models/resnet18.py — Architecture ProtoNet encoder

ResNet-18 modifié pour le few-shot learning facial :
  - conv1 kernel 3x3 (adapté aux images 112x112)
  - maxpool remplacé par Identity
  - tête de projection : Linear → ReLU → Linear → BatchNorm1d
  - normalisation L2 dans le forward (espace sphérique)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ProtoNetEncoder(nn.Module):
    """
    Encodeur ProtoNet basé sur ResNet-18 pré-entraîné ImageNet.
    Produit des embeddings normalisés L2 pour la distance cosinus.
    """

    def __init__(self, embedding_dim: int = 256, backbone_name: str = "resnet18"):
        super().__init__()

        # Backbone ResNet-18 avec poids ImageNet
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Adapté aux images 112x112 :
        # kernel 3x3 au lieu de 7x7 + suppression du maxpool
        backbone.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        in_features   = backbone.fc.in_features   # 512 pour ResNet-18
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Tête de projection : 512 → embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : tenseur [B, 3, H, W] normalisé ImageNet

        Returns:
            embeddings : tenseur [B, embedding_dim] normalisé L2
                         → prêt pour la distance cosinus
        """
        features   = self.features(x)           # [B, 512, 1, 1]
        features   = features.flatten(1)         # [B, 512]
        embeddings = self.projection(features)   # [B, embedding_dim]
        return F.normalize(embeddings, p=2, dim=1)  # normalisation L2


def cosine_distance(
    query_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """
    Distance cosinus entre query et prototypes.
    Les deux tenseurs doivent être normalisés L2.

    Args:
        query_embeddings : [N_queries, dim]
        prototypes       : [N_classes, dim]

    Returns:
        distances : [N_queries, N_classes]  valeurs dans [0, 2]
                    0 = identique, 2 = opposé
    """
    return 1 - torch.mm(query_embeddings, prototypes.t())


def compute_prototypes(
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    k_way: int,
) -> torch.Tensor:
    """
    Calcule les prototypes (moyenne des embeddings par classe).
    Renormalise L2 après la moyenne.

    Args:
        support_embeddings : [K_WAY * N_SHOT, dim]
        support_labels     : [K_WAY * N_SHOT]
        k_way              : nombre de classes

    Returns:
        prototypes : [K_WAY, dim] normalisés L2
    """
    dim        = support_embeddings.size(-1)
    prototypes = torch.zeros(k_way, dim, device=support_embeddings.device)

    for label in range(k_way):
        mask             = support_labels == label
        prototypes[label] = support_embeddings[mask].mean(dim=0)

    return F.normalize(prototypes, p=2, dim=1)


def prototypical_loss(
    support_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    query_labels: torch.Tensor,
    k_way: int,
    scale: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loss ProtoNet avec distance cosinus et temperature scaling.

    Args:
        support_embeddings : [K_WAY * N_SHOT, dim]
        query_embeddings   : [K_WAY * N_QUERY, dim]
        support_labels     : [K_WAY * N_SHOT]
        query_labels       : [K_WAY * N_QUERY]
        k_way              : nombre de classes
        scale              : temperature scaling (défaut 20.0)

    Returns:
        loss     : scalaire
        accuracy : scalaire
    """
    prototypes   = compute_prototypes(support_embeddings, support_labels, k_way)
    distances    = cosine_distance(query_embeddings, prototypes)
    scaled_logits = -distances * scale
    log_probs    = F.log_softmax(scaled_logits, dim=1)
    loss         = F.nll_loss(log_probs, query_labels)
    predictions  = log_probs.argmax(dim=1)
    accuracy     = (predictions == query_labels).float().mean()

    return loss, accuracy