"""
Entraînement ProtoNet avec DDP + Mixed Precision (AMP).

Mixed Precision : les forward/backward se font en FP16 (2x moins de mémoire)
                  seul l'update des poids reste en FP32 (pour la stabilité).

Gradient Checkpointing : activé dans model.py, recalcule les activations
                         au lieu de les stocker → encore moins de mémoire.

LANCEMENT :
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train.py

    Ou avec 4 GPU :
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py
"""

import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import config
from dataset import create_datasets, EpisodicDataset, get_eval_transforms
from model import ProtoNetEncoder, prototypical_loss


# ============================================================
# SETUP DDP
# ============================================================

def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, local_rank, world_size, device


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def print_main(msg):
    if is_main_process():
        print(msg, flush=True)


# ============================================================
# TRAIN / EVAL avec Mixed Precision
# ============================================================

def train_one_epoch(encoder, train_loader, optimizer, scaler, device):
    encoder.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for batch in train_loader:
        support_images, query_images, support_labels, query_labels = batch
        support_images = support_images.squeeze(0).to(device, non_blocking=True)
        query_images = query_images.squeeze(0).to(device, non_blocking=True)
        support_labels = support_labels.squeeze(0).to(device, non_blocking=True)
        query_labels = query_labels.squeeze(0).to(device, non_blocking=True)

        n_support = support_images.size(0)
        all_images = torch.cat([support_images, query_images], dim=0)

        optimizer.zero_grad()

        # Mixed Precision : forward en FP16
        with autocast():
            all_emb = encoder(all_images)
            support_emb = all_emb[:n_support]
            query_emb = all_emb[n_support:]

            loss, acc = prototypical_loss(
                support_emb, query_emb, support_labels, query_labels, config.K_WAY
            )

        # Backward en FP16 avec GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc += acc.item()
        n += 1

    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(encoder, val_loader, device, k_way):
    encoder.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for batch in val_loader:
        support_images, query_images, support_labels, query_labels = batch
        support_images = support_images.squeeze(0).to(device, non_blocking=True)
        query_images = query_images.squeeze(0).to(device, non_blocking=True)
        support_labels = support_labels.squeeze(0).to(device, non_blocking=True)
        query_labels = query_labels.squeeze(0).to(device, non_blocking=True)

        n_support = support_images.size(0)
        all_images = torch.cat([support_images, query_images], dim=0)

        with autocast():
            all_emb = encoder(all_images)
            support_emb = all_emb[:n_support]
            query_emb = all_emb[n_support:]

            loss, acc = prototypical_loss(
                support_emb, query_emb, support_labels, query_labels, k_way
            )

        total_loss += loss.item()
        total_acc += acc.item()
        n += 1

    return total_loss / n, total_acc / n


def reduce_metric(value, world_size):
    tensor = torch.tensor(value).cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / world_size).item()


# ============================================================
# MAIN
# ============================================================

def main():
    rank, local_rank, world_size, device = setup_ddp()

    print_main(f"DDP initialisé — {world_size} GPU(s)")
    print_main(f"Mixed Precision : activé (FP16)")
    print_main(f"Gradient Checkpointing : activé")

    if is_main_process():
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # --- Données ---
    print_main("\n" + "=" * 60)
    print_main("CHARGEMENT DES DONNÉES")
    print_main("=" * 60)

    if is_main_process():
        train_dataset, val_dataset, test_data = create_datasets()
    dist.barrier()
    if not is_main_process():
        train_dataset, val_dataset, test_data = create_datasets()

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset, batch_size=1, sampler=train_sampler,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, sampler=val_sampler,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    # --- Modèle avec checkpointing ---
    print_main("\n" + "=" * 60)
    print_main("INITIALISATION DU MODÈLE")
    print_main("=" * 60)

    encoder = ProtoNetEncoder(
        embedding_dim=config.EMBEDDING_DIM,
        use_checkpointing=True  # Économise ~50% de mémoire GPU
    ).to(device)

    encoder = DDP(encoder, device_ids=[local_rank])

    if is_main_process():
        total_params = sum(p.numel() for p in encoder.parameters())
        print_main(f"Paramètres : {total_params:,}")

        # Afficher la mémoire GPU utilisée
        mem = torch.cuda.memory_allocated(local_rank) / 1e9
        print_main(f"Mémoire GPU après init : {mem:.2f} Go")

    # --- Optimiseur + GradScaler pour mixed precision ---
    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
    )

    # GradScaler gère la mise à l'échelle des gradients en FP16
    scaler = GradScaler()

    # --- Training ---
    print_main("\n" + "=" * 60)
    print_main("ENTRAÎNEMENT")
    print_main(f"  Backbone            : {config.BACKBONE} (modifié conv1/maxpool)")
    print_main(f"  Setup               : {config.N_SHOT}-shot {config.K_WAY}-way")
    print_main(f"  Queries par classe  : {config.N_QUERY}")
    print_main(f"  Épisodes/époque     : {config.EPISODES_PER_EPOCH} "
               f"(~{config.EPISODES_PER_EPOCH // world_size} par GPU)")
    print_main(f"  Époques max         : {config.NUM_EPOCHS}")
    print_main(f"  Optimisations mémoire : FP16 + Gradient Checkpointing")
    print_main("=" * 60)

    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            encoder, train_loader, optimizer, scaler, device
        )
        val_loss, val_acc = evaluate(encoder, val_loader, device, k_way=config.K_WAY)

        train_loss = reduce_metric(train_loss, world_size)
        train_acc = reduce_metric(train_acc, world_size)
        val_loss = reduce_metric(val_loss, world_size)
        val_acc = reduce_metric(val_acc, world_size)

        scheduler.step(val_acc)
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Afficher mémoire GPU de temps en temps
        mem_info = ""
        if epoch % 20 == 1 and is_main_process():
            mem = torch.cuda.max_memory_allocated(local_rank) / 1e9
            mem_info = f" | Mem: {mem:.1f}Go"

        print_main(
            f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s{mem_info}"
        )

        if is_main_process():
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0

                checkpoint = {
                    "epoch": epoch,
                    "encoder_state_dict": encoder.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": {
                        "embedding_dim": config.EMBEDDING_DIM,
                        "k_way": config.K_WAY,
                        "n_shot": config.N_SHOT,
                        "image_size": config.IMAGE_SIZE,
                        "backbone": config.BACKBONE,
                    }
                }
                torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, "best_protonet.pth"))
                print_main(f"  → Sauvegardé ! (Val Acc: {val_acc:.4f})")
            else:
                epochs_without_improvement += 1

        es_tensor = torch.tensor([epochs_without_improvement]).cuda()
        dist.broadcast(es_tensor, src=0)
        epochs_without_improvement = es_tensor.item()

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print_main(f"\nEarly stopping à l'époque {epoch}")
            break

    # --- Évaluation finale ---
    if is_main_process():
        print_main("\n" + "=" * 60)
        print_main("ÉVALUATION FINALE")
        print_main("=" * 60)

        best_ckpt = torch.load(
            os.path.join(config.CHECKPOINT_DIR, "best_protonet.pth"),
            map_location=device
        )
        encoder.module.load_state_dict(best_ckpt["encoder_state_dict"])
        print_main(f"Meilleur modèle : époque {best_ckpt['epoch']}, "
                   f"val_acc: {best_ckpt['val_acc']:.4f}")

        test_dataset = EpisodicDataset(
            test_data, n_episodes=200,
            k_way=config.K_WAY, n_shot=config.N_SHOT, n_query=config.N_QUERY,
            transform=get_eval_transforms()
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        test_loss, test_acc = evaluate(encoder, test_loader, device, k_way=config.K_WAY)
        print_main(f"\nTest {config.N_SHOT}-shot {config.K_WAY}-way :")
        print_main(f"  Accuracy : {test_acc:.4f} ({test_acc * 100:.2f}%)")

    cleanup_ddp()


if __name__ == "__main__":
    main()