
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config


class ProtoNetEncoder(nn.Module):
    """
    Encoder ProtoNet basé sur ResNet pré-entraîné ImageNet.
    Backbone configurable : resnet18 (512 features) ou resnet50 (2048 features).
    """

    def __init__(self, embedding_dim=128, backbone_name=None):
        super().__init__()

        backbone_name = backbone_name or config.BACKBONE

        if backbone_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif backbone_name == "resnet101":
            backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            raise ValueError(f"Backbone inconnu : {backbone_name}")


        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        in_features = backbone.fc.in_features
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Projection avec BatchNorm1d en plus
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        features = self.features(x)
        features = features.flatten(1)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)


def compute_prototypes(support_embeddings, support_labels, k_way):
    embedding_dim = support_embeddings.size(-1)
    prototypes = torch.zeros(k_way, embedding_dim, device=support_embeddings.device)
    for label in range(k_way):
        mask = support_labels == label
        prototypes[label] = support_embeddings[mask].mean(dim=0)
    return F.normalize(prototypes, p=2, dim=1)


def cosine_distance(query_embeddings, prototypes):
    return 1 - torch.mm(query_embeddings, prototypes.t())


def prototypical_loss(support_embeddings, query_embeddings,
                      support_labels, query_labels, k_way, scale=20.0):
    
    prototypes = compute_prototypes(support_embeddings, support_labels, k_way)
    distances = cosine_distance(query_embeddings, prototypes)
    scaled_logits = -distances * scale
    log_probs = F.log_softmax(scaled_logits, dim=1)
    loss = F.nll_loss(log_probs, query_labels)
    predictions = log_probs.argmax(dim=1)
    accuracy = (predictions == query_labels).float().mean()
    
    return loss, accuracy
