"""Training loop for TreeSegPy with per-epoch metrics and checkpoints."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from .config import make_config
from .dataset import PatchDataset, collate_single, make_file_batches
from .model import TreeSegModel


def list_patch_files(patch_dir):
    return sorted(Path(patch_dir).glob("*.npz"))


def plot_name_from_patch(path):
    stem = Path(path).stem
    return stem.split("_patch")[0]


def split_by_plot(files, val_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    plots = sorted(set(plot_name_from_patch(f) for f in files))
    rng.shuffle(plots)
    n_val = max(1, int(len(plots) * val_fraction))
    val_plots = set(plots[:n_val])
    train_files = [f for f in files if plot_name_from_patch(f) not in val_plots]
    val_files = [f for f in files if plot_name_from_patch(f) in val_plots]
    return train_files, val_files, sorted(val_plots)


def make_loader(file_groups, config, shuffle, seed, num_workers):
    ds = PatchDataset(file_groups, config)
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_single,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    ), ds


def evaluate(model, loader, device):
    """Overall loss and accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            labels = batch.labels
            loss = model.loss(outputs, labels)
            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_points += len(labels)
    return total_loss / max(1, total_points), total_correct / max(1, total_points)


def compute_tree_f1(model, loader, device):
    """Tree-class precision, recall, F1."""
    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            preds = outputs.argmax(dim=1)
            labels = batch.labels
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return precision, recall, f1


def train(args):
    config = make_config()
    config.batch_num = args.batch_num

    files = list_patch_files(args.patch_dir)
    print(f"total patches: {len(files)}")
    if len(files) == 0:
        raise RuntimeError(f"No .npz patches found in {args.patch_dir}")

    train_files, val_files, val_plots = split_by_plot(
        files, val_fraction=args.val_fraction, seed=args.seed,
    )
    print(f"train patches: {len(train_files)}")
    print(f"val patches:   {len(val_files)} ({len(val_plots)} plots)")
    print(f"val plots: {val_plots}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TreeSegModel(config).to(device)
    print(f"model on {device}, {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.98, weight_decay=1e-6)
    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-4)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_f1 = 0.0
    # Early stopping after
    patience = 6
    epochs_no_improve = 0

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        t0 = time.time()

        train_groups = make_file_batches(
            train_files, args.batch_num, shuffle=True, seed=args.seed + epoch,
        )
        train_loader, _ = make_loader(
            train_groups, config, shuffle=False,
            seed=args.seed + epoch, num_workers=args.num_workers,
        )

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_points = 0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            labels = batch.labels
            loss = model.loss(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()

            epoch_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_points += len(labels)
            n_batches += 1

        scheduler.step()

        train_loss = epoch_loss / max(1, epoch_points)
        train_acc = epoch_correct / max(1, epoch_points)

        val_groups = make_file_batches(val_files, args.batch_num, shuffle=False)
        val_loader, _ = make_loader(
            val_groups, config, shuffle=False,
            seed=args.seed, num_workers=args.num_workers,
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        val_precision, val_recall, val_f1 = compute_tree_f1(model, val_loader, device)

        elapsed = time.time() - t0
        print(
            f"epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"tree P {val_precision:.4f} R {val_recall:.4f} F1 {val_f1:.4f} | "
            f"{elapsed:.1f}s | lr {scheduler.get_last_lr()[0]:.5f}",
            flush=True,
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "tree_f1": val_f1,
            "tree_precision": val_precision,
            "tree_recall": val_recall,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  -> new best val loss {val_loss:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(ckpt, ckpt_dir / "best_f1.pt")
            print(f"  -> new best tree F1 {val_f1:.4f}", flush=True)
        else:
            epochs_no_improve += 1
            print(f"  (no F1 improvement for {epochs_no_improve}/{patience} epochs)", flush=True)
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best F1 was {best_f1:.4f}", flush=True)
                break

        metrics_file = ckpt_dir / "metrics.csv"
        write_header = not metrics_file.exists()
        with open(metrics_file, "a") as f:
            if write_header:
                f.write("epoch,train_loss,train_acc,val_loss,val_acc,"
                        "tree_precision,tree_recall,tree_f1,lr,elapsed\n")
            f.write(
                f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                f"{val_loss:.6f},{val_acc:.6f},"
                f"{val_precision:.6f},{val_recall:.6f},{val_f1:.6f},"
                f"{scheduler.get_last_lr()[0]:.6f},{elapsed:.1f}\n"
            )

    print("training done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir",
                        default=str(Path.home() / "projects/TreeSegPy/data/patches_geom"))
    parser.add_argument("--checkpoint-dir",
                        default=str(Path.home() / "projects/TreeSegPy/logs"))
    parser.add_argument("--batch-num", type=int, default=2)
    parser.add_argument("--max-epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()