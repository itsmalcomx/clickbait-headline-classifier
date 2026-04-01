#this script runs additional experiments on the clickbait dataset, testing class balancing, text augmentation, and dimensionality reduction techniques. 
# It saves results to CSV and generates bar charts for each experiment.
import os
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
random.seed(42)

from sklearn.pipeline            import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model        import LogisticRegression
from sklearn.decomposition       import TruncatedSVD
from sklearn.model_selection     import StratifiedKFold, cross_val_score
from sklearn.metrics             import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling      import SMOTE
from imblearn.pipeline           import Pipeline as ImbPipeline

DATA_PATH   = "headlines.csv"
RESULTS_DIR = "results/experiments"
N_FOLDS     = 5
RANDOM_SEED = 42
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("  Clickbait — Additional Experiments")
print("=" * 60)

df = pd.read_csv(DATA_PATH).dropna(subset=["headline", "label"])
df["headline"] = df["headline"].astype(str).str.strip()
df = df[df["headline"].str.len() > 5].reset_index(drop=True)
X = df["headline"].values
y = df["label"].astype(int).values
print(f"\n📂 {len(df)} headlines  |  Clickbait: {(y==1).sum()}  Legit: {(y==0).sum()}")

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

TFIDF_PARAMS = dict(ngram_range=(1, 2), max_features=10_000,
                    sublinear_tf=True, strip_accents="unicode",
                    min_df=1)

def run_pipeline(pipe, label):
    acc   = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy",  n_jobs=-1).mean()
    f1    = cross_val_score(pipe, X, y, cv=cv, scoring="f1",        n_jobs=-1).mean()
    auroc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc",   n_jobs=-1).mean()
    print(f"    {label:<35} Acc={acc:.4f}  F1={f1:.4f}  AUROC={auroc:.4f}")
    return {"Method": label, "Accuracy": acc, "F1": f1, "AUROC": auroc}

def bar_chart(df_res, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_res)))
    for ax, metric in zip(axes, ["Accuracy", "F1", "AUROC"]):
        bars = ax.bar(range(len(df_res)), df_res[metric], color=colors, width=0.5)
        ax.set_xticks(range(len(df_res)))
        ax.set_xticklabels(df_res["Method"], rotation=20, ha="right", fontsize=8)
        ax.set_title(metric)
        lo = df_res[metric].min()
        ax.set_ylim(max(0, lo - 0.05), min(1.0, df_res[metric].max() + 0.05))
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# EXPERIMENT 1 — CLASS BALANCE

print("\n  Experiment 1: Class Balance ")

rows = []
rows.append(run_pipeline(Pipeline([
    ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
    ("clf",   LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
]), "Baseline (no resampling)"))

rows.append(run_pipeline(Pipeline([
    ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
    ("clf",   LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                  class_weight="balanced")),
]), "Class weighting"))

rows.append(run_pipeline(ImbPipeline([
    ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
    ("smote", SMOTE(random_state=RANDOM_SEED)),
    ("clf",   LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
]), "SMOTE oversampling"))

df1 = pd.DataFrame(rows)
df1.to_csv(f"{RESULTS_DIR}/exp1_smote.csv", index=False)
bar_chart(df1, "Experiment 1: Effect of Class Balancing",
          f"{RESULTS_DIR}/exp1_smote.png")
print(f" Saved exp1_smote.csv + exp1_smote.png")


# EXPERIMENT 2 — TEXT AUGMENTATION

print("\n  Experiment 2: Text Augmentation ")

SYNONYM_MAP = {
    "shocking": ["surprising", "astonishing", "stunning"],
    "amazing":  ["incredible", "remarkable", "extraordinary"],
    "secret":   ["hidden", "unknown", "concealed"],
    "weird":    ["strange", "bizarre", "unusual"],
    "best":     ["top", "greatest", "finest"],
    "worst":    ["terrible", "awful", "dreadful"],
    "world":    ["global", "international", "worldwide"],
    "people":   ["individuals", "persons", "folks"],
    "says":     ["claims", "states", "reports"],
    "new":      ["latest", "recent", "fresh"],
    "big":      ["major", "significant", "large"],
    "found":    ["discovered", "uncovered", "identified"],
    "show":     ["reveal", "demonstrate", "display"],
    "get":      ["obtain", "receive", "acquire"],
}

def augment(text, p=0.25):
    words = text.split()
    out = []
    for w in words:
        key = w.lower().rstrip(".,!?")
        if key in SYNONYM_MAP and random.random() < p:
            out.append(random.choice(SYNONYM_MAP[key]))
        else:
            out.append(w)
    return " ".join(out)

def aug_dataset(X_in, y_in, n=1):
    Xa, ya = list(X_in), list(y_in)
    for text, label in zip(X_in, y_in):
        for _ in range(n):
            Xa.append(augment(text))
            ya.append(label)
    return np.array(Xa), np.array(ya)

rows2 = []
base_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
    ("clf",   LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
])

# No augmentation
scores = {"Accuracy": [], "F1": [], "AUROC": []}
for tr, va in cv.split(X, y):
    base_pipe.fit(X[tr], y[tr])
    yp = base_pipe.predict(X[va])
    ypr = base_pipe.predict_proba(X[va])[:, 1]
    scores["Accuracy"].append(accuracy_score(y[va], yp))
    scores["F1"].append(f1_score(y[va], yp))
    scores["AUROC"].append(roc_auc_score(y[va], ypr))
row = {"Method": "No augmentation"}
row.update({k: np.mean(v) for k, v in scores.items()})
rows2.append(row)
print(f"    {'No augmentation':<35} Acc={row['Accuracy']:.4f}  F1={row['F1']:.4f}  AUROC={row['AUROC']:.4f}")

# With augmentation
for n_aug, lbl in [(1, "Synonym augmentation (1x)"), (2, "Synonym augmentation (2x)")]:
    scores = {"Accuracy": [], "F1": [], "AUROC": []}
    for tr, va in cv.split(X, y):
        Xtr_aug, ytr_aug = aug_dataset(X[tr], y[tr], n=n_aug)
        base_pipe.fit(Xtr_aug, ytr_aug)
        yp = base_pipe.predict(X[va])
        ypr = base_pipe.predict_proba(X[va])[:, 1]
        scores["Accuracy"].append(accuracy_score(y[va], yp))
        scores["F1"].append(f1_score(y[va], yp))
        scores["AUROC"].append(roc_auc_score(y[va], ypr))
    row = {"Method": lbl}
    row.update({k: np.mean(v) for k, v in scores.items()})
    rows2.append(row)
    print(f"    {lbl:<35} Acc={row['Accuracy']:.4f}  F1={row['F1']:.4f}  AUROC={row['AUROC']:.4f}")

df2 = pd.DataFrame(rows2)
df2.to_csv(f"{RESULTS_DIR}/exp2_augmentation.csv", index=False)
bar_chart(df2, "Experiment 2: Effect of Text Augmentation (Synonym Replacement)",
          f"{RESULTS_DIR}/exp2_augmentation.png")
print(f" Saved exp2_augmentation.csv + exp2_augmentation.png")


# EXPERIMENT 3 — DIMENSIONALITY REDUCTION

print("\n── Experiment 3: Dimensionality Reduction (SVD/LSA) ────")

# Find actual feature count first
n_feats = TfidfVectorizer(**TFIDF_PARAMS).fit_transform(X).shape[1]
print(f"  Actual TF-IDF feature count: {n_feats}")

candidates = [50, 100, 200, 500]
valid_components = [n for n in candidates if n < n_feats]
print(f"  Testing SVD components: {valid_components} + full baseline")

rows3 = []
for n_comp in valid_components:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("svd",   TruncatedSVD(n_components=n_comp, random_state=RANDOM_SEED)),
        ("clf",   LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])
    rows3.append(run_pipeline(pipe, f"SVD n_components={n_comp}"))

rows3.append(run_pipeline(Pipeline([
    ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
    ("clf",   LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
]), "No reduction (full TF-IDF)"))

df3 = pd.DataFrame(rows3)
df3.to_csv(f"{RESULTS_DIR}/exp3_pca.csv", index=False)

# Line chart for experiment 3
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Experiment 3: TF-IDF vs TF-IDF + SVD (Dimensionality Reduction)",
             fontsize=13, fontweight="bold")
for ax, metric in zip(axes, ["Accuracy", "F1", "AUROC"]):
    ax.plot(range(len(df3)), df3[metric], "o-", color="steelblue", linewidth=2, markersize=7)
    ax.set_xticks(range(len(df3)))
    ax.set_xticklabels(df3["Method"], rotation=20, ha="right", fontsize=8)
    ax.set_title(metric)
    lo = df3[metric].min()
    ax.set_ylim(max(0, lo - 0.05), min(1.0, df3[metric].max() + 0.05))
    ax.grid(alpha=0.3)
    for i, val in enumerate(df3[metric]):
        ax.annotate(f"{val:.4f}", (i, val), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/exp3_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print(f" Saved exp3_pca.csv + exp3_pca.png")

print("\n" + "=" * 60)
print("  All 3 experiments complete!")
print(f"  Files saved to: {RESULTS_DIR}/")
print("=" * 60)