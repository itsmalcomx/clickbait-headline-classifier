#this script trains and evaluates clickbait classifiers using logistic regression and linear SVM, with TF-IDF features. It performs 5-fold cross-validation, computes metrics, and saves results to the 'results/' folder.
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without a display)
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.pipeline          import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import LinearSVC
from sklearn.calibration       import CalibratedClassifierCV
from sklearn.model_selection   import StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics           import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)

#configurations and constants

DATA_PATH   = "headlines.csv"
RESULTS_DIR = "results"
N_FOLDS     = 5          # cross-validation folds
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

#loading the data 

print("=" * 60)
print("  Clickbait Classifier — Training & Evaluation")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["headline", "label"])
df["headline"] = df["headline"].astype(str).str.strip()
df = df[df["headline"].str.len() > 5]

X = df["headline"].values
y = df["label"].astype(int).values

print(f"\n Loaded {len(df)} headlines")
print(f"   Clickbait (1) : {(y == 1).sum()}  ({(y==1).mean()*100:.1f}%)")
print(f"   Legit     (0) : {(y == 0).sum()}  ({(y==0).mean()*100:.1f}%)")

# display warning if there is an imbalance
ratio = (y == 1).sum() / (y == 0).sum()
if ratio < 0.6 or ratio > 1.67:
    print(f"\n  Class imbalance detected (ratio {ratio:.2f}). "
          f"Consider running experiments with class_weight='balanced'.")

# Feature Engineering
# TF-IDF on unigrams + bigrams, capped at 10k features

TFIDF = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10_000,
    sublinear_tf=True,       # log-scale TF, helps with text
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{2,}", # skip single-char tokens
)

# Defining the models to train and evaluate

models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TFIDF),
        ("clf",   LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TFIDF),
        # CalibratedClassifierCV wraps SVM so we get probability scores for AUROC
        ("clf",   CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=RANDOM_SEED))),
    ]),
}

# Cross Validation & Metrics

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

scoring = {
    "accuracy":  "accuracy",
    "f1":        "f1",
    "precision": "precision",
    "recall":    "recall",
    "roc_auc":   "roc_auc",
}

print(f"\n {N_FOLDS}-Fold Cross-Validation ")

all_metrics = []

for model_name, pipeline in models.items():
    print(f"\n  Training [{model_name}] ...")
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    row = {"Model": model_name}
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        row[metric.capitalize()] = f"{scores.mean():.4f} ± {scores.std():.4f}"
        row[f"_{metric}_mean"] = scores.mean()   # hidden numeric cols for plots

    all_metrics.append(row)

    print(f"    Accuracy  : {cv_results['test_accuracy'].mean():.4f}  "
          f"(± {cv_results['test_accuracy'].std():.4f})")
    print(f"    F1        : {cv_results['test_f1'].mean():.4f}  "
          f"(± {cv_results['test_f1'].std():.4f})")
    print(f"    Precision : {cv_results['test_precision'].mean():.4f}  "
          f"(± {cv_results['test_precision'].std():.4f})")
    print(f"    Recall    : {cv_results['test_recall'].mean():.4f}  "
          f"(± {cv_results['test_recall'].std():.4f})")
    print(f"    AUROC     : {cv_results['test_roc_auc'].mean():.4f}  "
          f"(± {cv_results['test_roc_auc'].std():.4f})")

# Saving the metrics to a CSV file for later analysis and reporting

display_cols = ["Model", "Accuracy", "F1", "Precision", "Recall", "Roc_auc"]
metrics_df = pd.DataFrame(all_metrics)
# rename for cleaner CSV
metrics_df = metrics_df.rename(columns={"Roc_auc": "AUROC"})
save_cols = [c for c in ["Model","Accuracy","F1","Precision","Recall","AUROC"] if c in metrics_df.columns]
metrics_df[save_cols].to_csv(f"{RESULTS_DIR}/metrics_summary.csv", index=False)
print(f"\n Metrics saved → {RESULTS_DIR}/metrics_summary.csv")

# Confusion Matrices 
# Re-fit on full data and predict via cross_val_predict for the confusion matrix

from sklearn.model_selection import cross_val_predict

print("\n Generating Confusion Matrices")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrices (5-Fold CV Predictions)", fontsize=14, fontweight="bold")

for ax, (model_name, pipeline) in zip(axes, models.items()):
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Legit (0)", "Clickbait (1)"],
        yticklabels=["Legit (0)", "Clickbait (1)"],
    )
    acc = accuracy_score(y, y_pred)
    ax.set_title(f"{model_name}\nAccuracy: {acc:.4f}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {RESULTS_DIR}/confusion_matrices.png")

# ROC Curves

print("\n Generating ROC Curves ")

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["steelblue", "tomato"]

for (model_name, pipeline), color in zip(models.items(), colors):
    y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    RocCurveDisplay.from_predictions(y, y_prob, name=model_name, ax=ax, color=color)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
ax.set_title("ROC Curves — Logistic Regression vs. SVM", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {RESULTS_DIR}/roc_curve.png")

# Learning Curve
# Shows how accuracy changes as training data grows (required experiment!)

print("\n Generating Learning Curves")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Learning Curves : Accuracy vs. Training Size", fontsize=14, fontweight="bold")

train_sizes = np.linspace(0.1, 1.0, 10)

for ax, (model_name, pipeline) in zip(axes, models.items()):
    train_sz, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    ax.plot(train_sz, train_mean, "o-", color="steelblue", label="Train accuracy")
    ax.fill_between(train_sz, train_mean - train_std, train_mean + train_std, alpha=0.15, color="steelblue")
    ax.plot(train_sz, val_mean, "o-", color="tomato", label="Validation accuracy")
    ax.fill_between(train_sz, val_mean - val_std, val_mean + val_std, alpha=0.15, color="tomato")

    ax.set_title(model_name)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f" Saved → {RESULTS_DIR}/learning_curve.png")

# Top Predictive Features
# What words most strongly predict clickbait vs. legit?

print("\n Top Predictive Features (Logistic Regression)")

lr_pipeline = models["Logistic Regression"]
lr_pipeline.fit(X, y)

tfidf_fitted = lr_pipeline.named_steps["tfidf"]
lr_fitted    = lr_pipeline.named_steps["clf"]

feature_names = np.array(tfidf_fitted.get_feature_names_out())
coefs         = lr_fitted.coef_[0]

top_n = 15
top_clickbait = feature_names[np.argsort(coefs)[-top_n:][::-1]]
top_legit     = feature_names[np.argsort(coefs)[:top_n]]

print(f"\n  Top {top_n} words → CLICKBAIT:")
print("  " + ", ".join(top_clickbait))
print(f"\n  Top {top_n} words → LEGIT NEWS:")
print("  " + ", ".join(top_legit))

# Save feature importance plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Top Predictive Words (Logistic Regression Coefficients)",
             fontsize=13, fontweight="bold")

top_cb_coefs = coefs[np.argsort(coefs)[-top_n:][::-1]]
top_lg_coefs = np.abs(coefs[np.argsort(coefs)[:top_n]])

axes[0].barh(top_clickbait[::-1], top_cb_coefs[::-1], color="tomato")
axes[0].set_title("Clickbait indicators")
axes[0].set_xlabel("Coefficient")

axes[1].barh(top_legit[::-1], top_lg_coefs[::-1], color="steelblue")
axes[1].set_title("Legit news indicators")
axes[1].set_xlabel("|Coefficient|")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n Saved → {RESULTS_DIR}/feature_importance.png")

# the final summary 

print("\n" + "=" * 60)
print("  All done! Results saved to the 'results/' folder:")
print(f"    confusion_matrices.png")
print(f"    roc_curve.png")
print(f"    learning_curve.png")
print(f"    feature_importance.png")
print(f"    metrics_summary.csv")
print("=" * 60)