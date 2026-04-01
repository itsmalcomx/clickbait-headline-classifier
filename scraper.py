#this script collects headlines from various RSS feeds, labels them as clickbait (1) or non-clickbait (0), and saves them to a CSV file for later use in training a machine learning model. 
# It uses the feedparser library to parse RSS feeds and pandas to manage the dataset. The script is designed to be run periodically to keep the dataset up-to-date with fresh headlines.
import os
import feedparser
import pandas as pd
import time
from datetime import datetime

# the RSS feed sources to be used 

CLICKBAIT_FEEDS = {
    # BuzzFeed sections
    "buzzfeed_world":       "https://www.buzzfeed.com/world.xml",
    "buzzfeed_tech":        "https://www.buzzfeed.com/tech.xml",
    "buzzfeed_lol":         "https://www.buzzfeed.com/lol.xml",
    "buzzfeed_omg":         "https://www.buzzfeed.com/omg.xml",
    "buzzfeed_celebrity":   "https://www.buzzfeed.com/celebrity.xml",
    "buzzfeed_food":        "https://www.buzzfeed.com/food.xml",
    # Tabloids
    "nypost":               "https://nypost.com/feed/",
    "dailymail":            "https://www.dailymail.co.uk/articles.rss",
    "dailystar":            "https://www.dailystar.co.uk/news/rss.xml",
    "thesun":               "https://www.thesun.co.uk/feed/",
    "mirror":               "https://www.mirror.co.uk/rss.xml",
    # Viral / entertainment
    "ladbible":             "https://www.ladbible.com/rss",
    "distractify":          "https://www.distractify.com/rss",
    "screenrant":           "https://screenrant.com/feed/",
    "looper":               "https://www.looper.com/feed/",
    "thegamer":             "https://www.thegamer.com/feed/",
    "gamerant":             "https://gamerant.com/feed/",
    "thedailybeast":        "https://www.thedailybeast.com/rss",
    "cracked":              "https://www.cracked.com/rss.xml",
    "thoughtcatalog":       "https://thoughtcatalog.com/feed/",
    "elitedaily":           "https://www.elitedaily.com/rss",
    "heavy":                "https://heavy.com/feed/",
    "okmagazine":           "https://okmagazine.com/feed/",
    "intouchweekly":        "https://www.intouchweekly.com/feed/",
    "closerweekly":         "https://www.closerweekly.com/feed/",
    "upworthy":             "https://feeds.feedburner.com/Upworthy",
    "mtonews":              "https://mtonews.com/feed",
}

NON_CLICKBAIT_FEEDS = {
    # Wire services
    "reuters_world":        "https://feeds.reuters.com/reuters/worldNews",
    "reuters_tech":         "https://feeds.reuters.com/reuters/technologyNews",
    "reuters_business":     "https://feeds.reuters.com/reuters/businessNews",
    "reuters_science":      "https://feeds.reuters.com/reuters/scienceNews",
    "reuters_health":       "https://feeds.reuters.com/reuters/healthNews",
    # BBC
    "bbc_world":            "https://feeds.bbci.co.uk/news/world/rss.xml",
    "bbc_tech":             "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "bbc_science":          "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "bbc_business":         "https://feeds.bbci.co.uk/news/business/rss.xml",
    "bbc_health":           "https://feeds.bbci.co.uk/news/health/rss.xml",
    # US public / broadsheet
    "npr":                  "https://feeds.npr.org/1001/rss.xml",
    "npr_world":            "https://feeds.npr.org/1004/rss.xml",
    "npr_science":          "https://feeds.npr.org/1007/rss.xml",
    "apnews":               "https://rsshub.app/apnews/topics/ap-top-news",
    "guardian_world":       "https://www.theguardian.com/world/rss",
    "guardian_tech":        "https://www.theguardian.com/technology/rss",
    "guardian_science":     "https://www.theguardian.com/science/rss",
    "nyt_world":            "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "nyt_tech":             "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "nyt_science":          "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
    "washpost_world":       "https://feeds.washingtonpost.com/rss/world",
    "washpost_tech":        "https://feeds.washingtonpost.com/rss/business/technology",
    # Science / tech broadsheets
    "sciencedaily":         "https://www.sciencedaily.com/rss/all.xml",
    "newscientist":         "https://www.newscientist.com/feed/home/",
    "arstechnica":          "https://feeds.arstechnica.com/arstechnica/index",
    "wired":                "https://www.wired.com/feed/rss",
    "techcrunch":           "https://techcrunch.com/feed/",
    "theverge":             "https://www.theverge.com/rss/index.xml",
    # Financial
    "ft":                   "https://www.ft.com/rss/home",
    "economist":            "https://www.economist.com/feeds/print-edition.rss",
    "bloomberg_tech":       "https://feeds.bloomberg.com/technology/news.rss",
    "wsj_world":            "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    # International english press
    "aljazeera":            "https://www.aljazeera.com/xml/rss/all.xml",
    "dw_world":             "https://rss.dw.com/rss/rss-en-all",
    "france24":             "https://www.france24.com/en/rss",
}

# Scraper

def scrape_feed(name, url, label):
    """Parse one RSS feed and return a list of row dicts."""
    print(f"  Fetching [{name}] ...", end=" ", flush=True)
    rows = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            headline = entry.get("title", "").strip()
            if not headline:
                continue
            # Parse date if available
            published = ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
            rows.append({
                "headline": headline,
                "source":   name,
                "label":    label,
                "date":     published,
            })
        print(f"{len(rows)} headlines")
    except Exception as e:
        print(f"FAILED ({e})")
    time.sleep(0.3)  
    return rows


def collect_all():
    all_rows = []

    print(f"\n🟠 Collecting CLICKBAIT feeds (label=1)  [{len(CLICKBAIT_FEEDS)} sources]:")
    for name, url in CLICKBAIT_FEEDS.items():
        all_rows.extend(scrape_feed(name, url, label=1))

    print(f"\n🔵 Collecting NON-CLICKBAIT feeds (label=0)  [{len(NON_CLICKBAIT_FEEDS)} sources]:")
    for name, url in NON_CLICKBAIT_FEEDS.items():
        all_rows.extend(scrape_feed(name, url, label=0))

    df = pd.DataFrame(all_rows, columns=["headline", "source", "label", "date"])

    # basic cleanup 
    df["headline"] = df["headline"].str.strip()
    df = df[df["headline"].str.len() > 10]   # drop suspiciously short entries
    df = df.drop_duplicates(subset="headline")
    df = df.reset_index(drop=True)

    return df


# Main 

if __name__ == "__main__":
    print("=" * 55)
    print("  Clickbait Headline Scraper")
    print(f"  {len(CLICKBAIT_FEEDS)} clickbait sources | {len(NON_CLICKBAIT_FEEDS)} legit sources")
    print("=" * 55)

    out_path = "headlines.csv"

    # load existing data if file already exists
    if os.path.exists(out_path):
        existing_df = pd.read_csv(out_path)
        existing_count = len(existing_df)
        print(f"\n Found existing dataset with {existing_count} headlines — will append new ones.")
    else:
        existing_df = pd.DataFrame(columns=["headline", "source", "label", "date"])
        existing_count = 0
        print("\n No existing dataset found — starting fresh.")

    #  Scrape fresh headlines 
    new_df = collect_all()

    #  merge with existing data and duplication check 
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset="headline")
    combined_df = combined_df.reset_index(drop=True)

    added = len(combined_df) - existing_count

    # summary 
    print("\n Dataset Summary ")
    print(f"  Total headlines : {len(combined_df)}  (+{added} new)")
    print(f"  Clickbait  (1)  : {(combined_df.label == 1).sum()}")
    print(f"  Legit      (0)  : {(combined_df.label == 0).sum()}")
    print(f"\n  By source:")
    for src, count in combined_df.groupby("source").size().items():
        label = combined_df[combined_df.source == src]["label"].iloc[0]
        tag = "🟠" if label == 1 else "🔵"
        print(f"    {tag} {src:<25} {count} headlines")

    # save  
    combined_df.to_csv(out_path, index=False)
    print(f"\n Saved to {out_path}")

    # progress bar toward target
    TARGET = 1000
    progress = min(len(combined_df) / TARGET, 1.0)
    filled = int(progress * 30)
    bar = "█" * filled + "░" * (30 - filled)
    print(f"\n  Progress to {TARGET} headlines:")
    print(f"  [{bar}] {len(combined_df)}/{TARGET}")
    if len(combined_df) >= TARGET:
        print(" Target reached! Your dataset is ready for ML.")
    else:
        print(f" Run again to collect more.")

    # preview  
    print("\n Sample Headlines")
    print("\nClickbait:")
    for h in combined_df[combined_df.label == 1]["headline"].head(3).values:
        print(f"  • {h}")
    print("\nLegit news:")
    for h in combined_df[combined_df.label == 0]["headline"].head(3).values:
        print(f"  • {h}")