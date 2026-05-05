"""
Prototypical Network (ProtoNet) avec backbone configurable.
"""

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

        # Initialisation et récupération du nombre de canaux
        if backbone_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_channels = 512
        elif backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_channels = 2048
        elif backbone_name == "resnet101":
            backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            in_channels = 2048
        else:
            raise ValueError(f"Backbone inconnu : {backbone_name}")
        
        # 1. Stem façon ArcFace : On garde les détails (112x112 en sortie)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        #  On force la réduction spatiale au début du layer1 
        # Pour que l'image finisse exactement en 7x7 à la fin du réseau (au lieu de 14x14)
        if backbone_name == "resnet18":
            backbone.layer1[0].conv1.stride = (2, 2)
            backbone.layer1[0].downsample = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64)
            )
        else: # Gestion des ResNet50/101 (blocs de type Bottleneck)
            backbone.layer1[0].conv2.stride = (2, 2)
            backbone.layer1[0].downsample = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256)
            )

        # 3. On supprime l'avgpool ET le fc (on s'arrête à -2 au lieu de -1)
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        # Calcul de la taille de la carte aplatie (Canaux * 7 * 7)
        flatten_dim = in_channels * 7 * 7 

        # 4. La projection façon ArcFace (Directement Flatten -> Linear -> BatchNorm)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim) # Crucial pour stabiliser les vecteurs finaux !
        )

        #print(f"Backbone : {backbone_name} (Flattened map: {flatten_dim}) → embedding: {embedding_dim}")

    def forward(self, x):
        features = self.features(x)
        # Plus besoin du .flatten(1) ici, car il est géré par nn.Flatten() dans self.projection
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
