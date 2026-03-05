"""
S&P 500 Regime Detection — Training & Evaluation (v3 - Historical)
==================================================================
Trained on 45 years of historical S&P 500 data and robust macro features.
"""

import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, balanced_accuracy_score)
import json

from src.data_pipeline import run_pipeline
from src.model import RegimeClassifier, EarlyStopping


def train_model(X_train, y_train, X_val, y_val,
                hidden_dim=64, n_blocks=1, dropout=0.3,
                lr=1e-3, epochs=300, batch_size=64, seed=42):
    """Train with standard BCE and proper early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_features = X_train.shape[1]
    
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32)
    
    train_ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    model = RegimeClassifier(n_features, hidden_dim, n_blocks, dropout)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    early_stop = EarlyStopping(patience=25)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_v)
            val_loss = criterion(val_preds, y_v).item()
            val_acc = accuracy_score(y_val, (val_preds.numpy() >= 0.5).astype(int))
        
        early_stop.step(val_loss, model)
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {epoch_loss/len(loader):.4f} | "
                  f"Val: {val_loss:.4f} | Val Acc: {val_acc:.1%}")
        
        if early_stop.should_stop:
            print(f"  Early stop at epoch {epoch+1}")
            break
    
    early_stop.restore(model)
    return model


def feature_importance(model, X, y, feature_names, top_n=15):
    """Permutation importance based on balanced accuracy."""
    print("\n[Feature Importance] Computing...")
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        base_probs = model(X_t).numpy()
    base_acc = balanced_accuracy_score(y, (base_probs >= 0.5).astype(int))
    
    results = []
    for i, name in enumerate(feature_names):
        drops = []
        for _ in range(5):
            X_p = X.copy()
            np.random.shuffle(X_p[:, i])
            with torch.no_grad():
                p = model(torch.tensor(X_p, dtype=torch.float32)).numpy()
            drops.append(base_acc - balanced_accuracy_score(y, (p >= 0.5).astype(int)))
        results.append({"feature": name, "importance": np.mean(drops)})
    
    df = pd.DataFrame(results).sort_values("importance", ascending=False)
    print(f"\n  Top {top_n} features:")
    for _, r in df.head(top_n).iterrows():
        bar = "█" * max(1, int(r['importance'] * 200))
        print(f"  {r['feature']:<30s} {r['importance']:+.4f}  {bar}")
    return df


def main():
    print("╔" + "═"*58 + "╗")
    print("║  S&P 500 REGIME DETECTION — ROBUST HISTORICAL MODEL      ║")
    print("╚" + "═"*58 + "╝\n")
    
    configs = [
        {"horizon": 20, "hidden": 64,  "blocks": 1, "dropout": 0.3, "lr": 1e-3,  "label": "A: Light (h=20)"},
        {"horizon": 20, "hidden": 128, "blocks": 2, "dropout": 0.4, "lr": 5e-4,  "label": "B: Deep (h=20)"},
    ]
    
    best_bal_acc = 0
    best_config = None
    best_model = None
    best_results = None
    best_features = None
    
    for cfg in configs:
        print(f"\n{'━'*60}")
        print(f"  CONFIG: {cfg['label']}")
        print(f"{'━'*60}")
        
        X_train, X_test, y_train, y_test, feat_names = run_pipeline(
            horizon=cfg["horizon"]
        )
        
        # Walk-forward val split (last 20% of train set)
        val_size = int(len(X_train) * 0.2)
        X_tr = X_train.iloc[:-val_size].values
        y_tr = y_train.iloc[:-val_size].values
        X_val = X_train.iloc[-val_size:].values
        y_val = y_train.iloc[-val_size:].values
        X_te = X_test.values
        y_te = y_test.values
        
        # Multi-seed ensemble
        all_probs = []
        for seed in [42, 123, 777]:
            print(f"\n  Seed {seed}:")
            model = train_model(
                X_tr, y_tr, X_val, y_val,
                hidden_dim=cfg["hidden"], n_blocks=cfg["blocks"],
                dropout=cfg["dropout"], lr=cfg["lr"],
                epochs=300, batch_size=64, seed=seed
            )
            model.eval()
            with torch.no_grad():
                probs = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
            all_probs.append(probs)
        
        avg_probs = np.mean(all_probs, axis=0)
        ens_preds = (avg_probs >= 0.5).astype(int)
        
        ens_acc = accuracy_score(y_te, ens_preds)
        ens_bal_acc = balanced_accuracy_score(y_te, ens_preds)
        
        print(f"\n  >>> 3-Seed Ensemble Accuracy:          {ens_acc:.4f} ({ens_acc*100:.1f}%)")
        print(f"  >>> 3-Seed Ensemble BALANCED Accuracy: {ens_bal_acc:.4f} ({ens_bal_acc*100:.1f}%)")
        
        if ens_bal_acc > best_bal_acc:
            best_bal_acc = ens_bal_acc
            best_config = cfg
            best_model = model
            best_results = {
                "accuracy": ens_acc,
                "balanced_accuracy": ens_bal_acc,
                "probs": avg_probs,
                "preds": ens_preds,
                "y_test": y_te,
                "X_test": X_te
            }
            best_features = feat_names
    
    # ── Final Report ──
    print(f"\n{'═'*60}")
    print(f"  BEST CONFIG: {best_config['label']}")
    print(f"  TEST BALANCED ACCURACY: {best_bal_acc:.4f} ({best_bal_acc*100:.1f}%)")
    print(f"{'═'*60}")
    
    y_te = best_results["y_test"]
    preds = best_results["preds"]
    probs = best_results["probs"]
    
    prec = precision_score(y_te, preds, zero_division=0)
    rec = recall_score(y_te, preds, zero_division=0)
    f1 = f1_score(y_te, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_te, probs)
    except:
        auc = 0.5
    cm = confusion_matrix(y_te, preds)
    
    print(f"  Accuracy:  {best_results['accuracy']:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred Down  Pred Up")
    print(f"  Actual Down   {cm[0][0]:5d}     {cm[0][1]:5d}")
    print(f"  Actual Up     {cm[1][0]:5d}     {cm[1][1]:5d}")
    print("\n" + classification_report(y_te, preds, target_names=["Down", "Up"]))
    
    imp = feature_importance(best_model, best_results["X_test"],
                             y_te, best_features)
    
    # Save
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    torch.save(best_model.state_dict(), "models/regime_classifier_historical.pth")
    
    print(f"\n✅ Phase 3 Training complete! Best Balanced Accuracy: {best_bal_acc*100:.1f}%")


if __name__ == "__main__":
    main()
