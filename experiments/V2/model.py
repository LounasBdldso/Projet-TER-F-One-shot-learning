"""
Prototypical Network (ProtoNet) avec backbone configurable.
Inclut la modification conv1/maxpool pour petites images
et le gradient checkpointing pour économiser la mémoire GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import models

import config


class ProtoNetEncoder(nn.Module):
    """
    Encoder ProtoNet avec :
    - Modification conv1 (stride=1) + suppression maxpool
      pour préserver les détails sur des images 112x112
    - Gradient checkpointing optionnel pour réduire la mémoire GPU
    """

    def __init__(self, embedding_dim=128, backbone_name=None, use_checkpointing=True):
        super().__init__()

        backbone_name = backbone_name or config.BACKBONE
        self.use_checkpointing = use_checkpointing

        if backbone_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Backbone inconnu : {backbone_name}")

        # Modification pour petites images (112x112)
        # stride=1 au lieu de 2 → préserve la résolution
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Supprime le maxpool → plus de réduction de résolution précoce
        backbone.maxpool = nn.Identity()

        in_features = backbone.fc.in_features

        # Séparer les blocs pour pouvoir appliquer le checkpointing par bloc
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool  # Identity
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim)
        )

        mode = "avec checkpointing" if use_checkpointing else "sans checkpointing"
        print(f"Backbone : {backbone_name} modifié (features: {in_features}) "
              f"→ embedding: {embedding_dim} ({mode})")

    def _forward_features(self, x):
        """Forward pass à travers le backbone avec checkpointing optionnel."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.use_checkpointing and self.training:
            # Checkpoint chaque layer — recalcule les activations pendant le backward
            # au lieu de les garder en mémoire
            x = checkpoint(self.layer1, x, use_reentrant=False)
            x = checkpoint(self.layer2, x, use_reentrant=False)
            x = checkpoint(self.layer3, x, use_reentrant=False)
            x = checkpoint(self.layer4, x, use_reentrant=False)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        x = self.avgpool(x)
        return x

    def forward(self, x):
        features = self._forward_features(x)
        features = features.flatten(1)
        embeddings = self.projection(features)
        return embeddings


def compute_prototypes(support_embeddings, support_labels, k_way):
    embedding_dim = support_embeddings.size(-1)
    prototypes = torch.zeros(k_way, embedding_dim, device=support_embeddings.device)
    for label in range(k_way):
        mask = support_labels == label
        prototypes[label] = support_embeddings[mask].mean(dim=0)
    return prototypes


def euclidean_distance(query_embeddings, prototypes):
    return torch.cdist(query_embeddings, prototypes, p=2)


def prototypical_loss(support_embeddings, query_embeddings,
                      support_labels, query_labels, k_way):
    prototypes = compute_prototypes(support_embeddings, support_labels, k_way)
    distances = euclidean_distance(query_embeddings, prototypes)
    log_probs = F.log_softmax(-distances, dim=1)
    loss = F.nll_loss(log_probs, query_labels)
    predictions = log_probs.argmax(dim=1)
    accuracy = (predictions == query_labels).float().mean()
    return loss, accuracy