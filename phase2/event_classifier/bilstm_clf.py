"""
CASTER Phase 2 -- bidirectional LSTM event classifier (upgrade path).

Swap this in once the XGBoost MVP (xgboost_clf.py) is working end-to-end
and you want the higher ML-depth option described in the deep-dive
docs: a model that reads the full (30, 12) sequence rather than
flattened summary statistics, so it can pick up genuinely temporal
patterns XGBoost throws away -- e.g. the difference between a ball
that decelerates smoothly to a stop (DOT_BALL) versus one that
decelerates because it was just fielded sharply (SINGLE/DOUBLE), which
can look identical in mean/std/max aggregate stats but very different
frame-by-frame.

Same training data, same event vocabulary, same inference interface
(predict_window) as xgboost_clf.py -- run_phase2.py can use either
classifier interchangeably by swapping one import.

Usage:
    python phase2/event_classifier/bilstm_clf.py --sport football --epochs 30
    python phase2/event_classifier/bilstm_clf.py --sport football --epochs 30
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import EVENTS_BY_SPORT, SAVED_MODELS_DIR, FEATURE_VECTOR_SIZE, SUPPORTED_SPORTS

from phase2.event_classifier.dataset import (
    load_labelled_windows, print_class_distribution, label_id_to_event,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class CASTEREventLSTM(nn.Module):
        """
        Input:  (batch, 30, 12)
        LSTM:   2 layers, 128 hidden units, bidirectional
        Dense:  256 -> 128 -> num_classes
        Output: event logits (apply softmax for probabilities)

        One instance per sport -- the architecture is identical, only
        num_classes differs (currently 11 for football, including the
        NO_EVENT class), kept independent so adding another event to
        one sport later doesn't silently change the other.
        """

        def __init__(self, num_classes: int, input_size: int = FEATURE_VECTOR_SIZE,
                     hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size * 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            last_step = out[:, -1, :]   # final timestep's hidden state (both directions)
            return self.head(last_step)


    class WindowDataset(Dataset):
        def __init__(self, X_windows, y):
            self.X = torch.tensor(X_windows, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


def model_path_for(sport: str) -> Path:
    return SAVED_MODELS_DIR / f"bilstm_{sport}.pt"


def train_and_save(sport: str, epochs: int = 30, batch_size: int = 16,
                    lr: float = 1e-3, verbose: bool = True):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for this script. Install with: pip install torch")

    X_windows, y = load_labelled_windows(sport)
    if verbose:
        print_class_distribution(sport, y)

    num_classes = len(EVENTS_BY_SPORT[sport])
    dataset = WindowDataset(X_windows, y)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = CASTEREventLSTM(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total if total > 0 else 0.0
        best_val_acc = max(best_val_acc, val_acc)

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            avg_loss = total_loss / len(train_ds) if len(train_ds) > 0 else 0.0
            print(f"  epoch {epoch:>3}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.3f}")

    out_path = model_path_for(sport)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "sport": sport,
    }, out_path)
    print(f"Saved trained BiLSTM model (best val_acc={best_val_acc:.3f}) -> {out_path}")
    return model


def load_classifier(sport: str) -> "CASTEREventLSTM":
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    path = model_path_for(sport)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run:\n"
            f"  python phase2/event_classifier/bilstm_clf.py --sport {sport}\n"
            f"first."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = CASTEREventLSTM(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_window(model: "CASTEREventLSTM", sport: str, window_vectors) -> tuple:
    """Same interface as xgboost_clf.predict_window -- (event_name, confidence)."""
    x = torch.tensor([window_vectors], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    best_idx = int(torch.argmax(probs).item())
    confidence = float(probs[best_idx].item())
    event = label_id_to_event(sport, best_idx)
    return event, confidence


def main():
    parser = argparse.ArgumentParser(description="Train CASTER's BiLSTM event classifier")
    parser.add_argument("--sport", required=True, choices=list(SUPPORTED_SPORTS) + ["both"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    sports = list(SUPPORTED_SPORTS) if args.sport == "both" else [args.sport]
    for sport in sports:
        print(f"\n{'=' * 60}\nTraining BiLSTM classifier: {sport}\n{'=' * 60}")
        train_and_save(sport, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
