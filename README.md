# Clickbait Headline Dataset

Binary classification dataset of 1,896 English news headlines labeled as clickbait (1) or legitimate news (0).

## Files
- headlines.csv — the dataset
- scraper.py — RSS feed data collection script
- train.py — Logistic Regression and SVM training
- train_bert.py — DistilBERT fine-tuning
- experiments.py — SMOTE, augmentation, and PCA experiments

## Dataset
- 1,034 clickbait headlines from BuzzFeed, NY Post, Daily Mail, etc.
- 862 legitimate headlines from Reuters, BBC, NPR, Guardian, etc.
- Collected via automated RSS scraping (feedparser)

