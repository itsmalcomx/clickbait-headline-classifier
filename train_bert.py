#this script trains a DistilBERT-based clickbait classifier using the Hugging Face Transformers library. It performs 5-fold cross-validation, computes metrics, and saves results to the 'results/' folder.
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    RocCurveDisplay,
)

# configurations and constants 

DATA_PATH    = "headlines.csv"
RESULTS_DIR  = "results"
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 64       # headlines are short; 64 tokens is plenty
BATCH_SIZE   = 16
EPOCHS       = 3        # 3 epochs is standard for fine-tuning BERT
LR           = 2e-5
N_FOLDS      = 5
RANDOM_SEED  = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  DistilBERT Clickbait Classifier")
print(f"  Device: {device}")
print("=" * 60)

# loading the data

df = pd.read_csv(DATA_PATH).dropna(subset=["headline", "label"])
df["headline"] = df["headline"].astype(str).str.strip()
df = df[df["headline"].str.len() > 5].reset_index(drop=True)

texts  = df["headline"].tolist()
labels = df["label"].astype(int).tolist()

print(f"\n📂 Loaded {len(df)} headlines")
print(f"   Clickbait (1): {sum(labels)}  |  Legit (0): {len(labels)-sum(labels)}")

# dataset class 

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

class HeadlineDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }

# training the helpers

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_b       = batch["labels"].to(device)
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels_b)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()


def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(batch["labels"])
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds  = logits.argmax(dim=1).numpy()
    return labels, preds, probs

# 5-Fold Cross Validation

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
texts_arr  = np.array(texts)
labels_arr = np.array(labels)

fold_metrics = []
all_true, all_pred, all_prob = [], [], []

print(f"\n {N_FOLDS}-Fold Cross-Validation")

for fold, (train_idx, val_idx) in enumerate(cv.split(texts_arr, labels_arr)):
    print(f"\n  Fold {fold+1}/{N_FOLDS} ...", flush=True)

    train_ds = HeadlineDataset(
        texts_arr[train_idx].tolist(),
        labels_arr[train_idx].tolist(),
    )
    val_ds = HeadlineDataset(
        texts_arr[val_idx].tolist(),
        labels_arr[val_idx].tolist(),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # Fresh model each fold
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    for epoch in range(EPOCHS):
        train_epoch(model, train_loader, optimizer, scheduler)
        print(f"    Epoch {epoch+1}/{EPOCHS} done", flush=True)

    y_true, y_pred, y_prob = evaluate(model, val_loader)
    all_true.extend(y_true)
    all_pred.extend(y_pred)
    all_prob.extend(y_prob)

    fold_metrics.append({
        "fold":      fold + 1,
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "auroc":     roc_auc_score(y_true, y_prob),
    })

    m = fold_metrics[-1]
    print(f"    Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
          f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  "
          f"AUROC={m['auroc']:.4f}")

    # Free GPU/CPU memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Aggregating the results and saving to CSV 

metrics_df = pd.DataFrame(fold_metrics)
mean_row = metrics_df.mean(numeric_only=True).to_dict()
mean_row["fold"] = "mean"
std_row  = metrics_df.std(numeric_only=True).to_dict()
std_row["fold"]  = "std"
metrics_df = pd.concat(
    [metrics_df, pd.DataFrame([mean_row, std_row])],
    ignore_index=True,
)
metrics_df.to_csv(f"{RESULTS_DIR}/bert_metrics.csv", index=False)

print("\n DistilBERT Summary ")
m = fold_metrics
print(f"  Accuracy  : {np.mean([x['accuracy']  for x in m]):.4f}  "
      f"(± {np.std([x['accuracy']  for x in m]):.4f})")
print(f"  F1        : {np.mean([x['f1']        for x in m]):.4f}  "
      f"(± {np.std([x['f1']        for x in m]):.4f})")
print(f"  Precision : {np.mean([x['precision'] for x in m]):.4f}  "
      f"(± {np.std([x['precision'] for x in m]):.4f})")
print(f"  Recall    : {np.mean([x['recall']    for x in m]):.4f}  "
      f"(± {np.std([x['recall']    for x in m]):.4f})")
print(f"  AUROC     : {np.mean([x['auroc']     for x in m]):.4f}  "
      f"(± {np.std([x['auroc']     for x in m]):.4f})")
print(f"\n Saved → {RESULTS_DIR}/bert_metrics.csv")

# Confusion Matrix

all_true = np.array(all_true)
all_pred = np.array(all_pred)
all_prob = np.array(all_prob)

cm = confusion_matrix(all_true, all_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax,
            xticklabels=["Legit (0)", "Clickbait (1)"],
            yticklabels=["Legit (0)", "Clickbait (1)"])
ax.set_title(f"DistilBERT Confusion Matrix\nAccuracy: "
             f"{accuracy_score(all_true, all_pred):.4f}", fontsize=13)
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/bert_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f" Saved → {RESULTS_DIR}/bert_confusion_matrix.png")

# ROC Curve 

fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(all_true, all_prob,
                                  name="DistilBERT", ax=ax, color="mediumpurple")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
ax.set_title("ROC Curve — DistilBERT", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/bert_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f" Saved → {RESULTS_DIR}/bert_roc_curve.png")

print("\n" + "=" * 60)
print("  DistilBERT training is complete ")
print("  Compare bert_metrics.csv with metrics_summary.csv")
print("  to see how deep learning stacks up against classical ML.")
print("=" * 60)