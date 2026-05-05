

import os
import random
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import config


def load_celeba_identities(celeba_dir, min_images):
    """Charge les identités depuis la structure par dossiers."""
    identity_to_images = {}

    for folder_name in sorted(os.listdir(celeba_dir)):
        folder_path = os.path.join(celeba_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        images = glob(os.path.join(folder_path, "*.jpg"))
        images += glob(os.path.join(folder_path, "*.jpeg"))
        images += glob(os.path.join(folder_path, "*.png"))

        if len(images) > 0:
            identity_to_images[folder_name] = sorted(images)

    total = len(identity_to_images)
    total_imgs = sum(len(imgs) for imgs in identity_to_images.values())

    filtered = {
        identity: images
        for identity, images in identity_to_images.items()
        if len(images) >= min_images
    }
    filtered_imgs = sum(len(imgs) for imgs in filtered.values())

    print(f"Identités totales                    : {total}")
    print(f"Images totales                       : {total_imgs}")
    print(f"Identités avec >= {min_images} images : {len(filtered)}")
    print(f"Images retenues                      : {filtered_imgs}")

    if filtered:
        counts = [len(imgs) for imgs in filtered.values()]
        print(f"Images par identité — min: {min(counts)}, max: {max(counts)}, "
              f"moyenne: {sum(counts)/len(counts):.1f}")

    return filtered


def split_identities(identity_to_images, train_ratio, val_ratio, seed=42):
    all_identities = list(identity_to_images.keys())
    random.seed(seed)
    random.shuffle(all_identities)

    n = len(all_identities)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = all_identities[:n_train]
    val_ids = all_identities[n_train:n_train + n_val]
    test_ids = all_identities[n_train + n_val:]

    train_data = {id_: identity_to_images[id_] for id_ in train_ids}
    val_data = {id_: identity_to_images[id_] for id_ in val_ids}
    test_data = {id_: identity_to_images[id_] for id_ in test_ids}

    print(f"\nSplit des identités :")
    print(f"  Train : {len(train_data)}")
    print(f"  Val   : {len(val_data)}")
    print(f"  Test  : {len(test_data)}")

    return train_data, val_data, test_data

    
def get_train_transforms():
    return transforms.Compose([
        # --- 1. Opérations sur l'image (Format PIL) ---
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class EpisodicDataset(Dataset):
    """Dataset générant des épisodes pour ProtoNet."""

    def __init__(self, identity_to_images, n_episodes, k_way, n_shot, n_query, transform=None):
        self.identity_to_images = dict(identity_to_images)
        self.identities = list(self.identity_to_images.keys())
        self.n_episodes = n_episodes
        self.k_way = k_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.transform = transform

        min_needed = n_shot + n_query
        insufficient = [id_ for id_, imgs in self.identity_to_images.items()
                        if len(imgs) < min_needed]

        if insufficient:
            for id_ in insufficient:
                del self.identity_to_images[id_]
            self.identities = list(self.identity_to_images.keys())

        assert len(self.identities) >= k_way, (
            f"Pas assez d'identités ({len(self.identities)}) pour faire du {k_way}-way."
        )

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, index):
        chosen_ids = random.sample(self.identities, self.k_way)

        support_images, query_images = [], []
        support_labels, query_labels = [], []

        for label, identity in enumerate(chosen_ids):
            images = self.identity_to_images[identity]
            chosen = random.sample(images, self.n_shot + self.n_query)
            supp_paths = chosen[:self.n_shot]
            query_paths = chosen[self.n_shot:]

            for path in supp_paths:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                support_images.append(img)
                support_labels.append(label)

            for path in query_paths:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                query_images.append(img)
                query_labels.append(label)

        return (torch.stack(support_images),
                torch.stack(query_images),
                torch.tensor(support_labels),
                torch.tensor(query_labels))


def create_datasets():
    """Retourne les datasets (pas les loaders — ils sont créés dans train.py pour DDP)."""
    identity_to_images = load_celeba_identities(
        config.CELEBA_DIR, config.MIN_IMAGES_PER_IDENTITY
    )

    train_data, val_data, test_data = split_identities(
        identity_to_images, config.TRAIN_RATIO, config.VAL_RATIO
    )

    # En DDP, chaque GPU fait EPISODES_PER_EPOCH/world_size épisodes
    # Le total reste EPISODES_PER_EPOCH
    train_dataset = EpisodicDataset(
        train_data, n_episodes=config.EPISODES_PER_EPOCH,
        k_way=config.K_WAY, n_shot=config.N_SHOT, n_query=config.N_QUERY,
        transform=get_train_transforms()
    )

    val_dataset = EpisodicDataset(
        val_data, n_episodes=config.EPISODES_PER_EPOCH,
        k_way=config.K_WAY, n_shot=config.N_SHOT, n_query=config.N_QUERY,
        transform=get_eval_transforms()
    )

    return train_dataset, val_dataset, test_data