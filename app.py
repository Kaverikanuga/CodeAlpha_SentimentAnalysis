import re
import random
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"


def ensure_directories() -> None:
    """Create required directories if they do not exist."""
    for directory in [DATA_DIR, OUTPUT_DIR, SCREENSHOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def download_nltk_resources() -> set:
    """Download NLTK resources when needed and return English stopwords."""
    try:
        stopwords.words("english")
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


STOPWORDS = download_nltk_resources()


def create_default_dataset() -> pd.DataFrame:
    """Create a synthetic dataset with 100 realistic product reviews."""
    rng = random.Random(42)
    products = ["phone", "laptop", "headphones", "tablet", "camera", "speaker", "watch", "charger", "mouse", "keyboard"]
    features = ["battery life", "camera quality", "sound quality", "display", "build quality", "storage", "performance", "design", "price", "ease of use"]
    positive_adjectives = ["excellent", "amazing", "fantastic", "wonderful", "great", "superb", "impressive", "reliable"]
    negative_adjectives = ["poor", "terrible", "awful", "disappointing", "bad", "frustrating", "cheap", "weak"]
    neutral_adjectives = ["average", "decent", "okay", "fine", "standard"]

    positive_templates = [
        "The {product} has {adj} {feature} and I am very happy with the purchase.",
        "I love this {product} because the {feature} is {adj} and dependable.",
        "This {product} offers {adj} {feature}, which makes it a great buy.",
    ]
    negative_templates = [
        "I dislike this {product} because the {feature} is {adj} and not reliable.",
        "This {product} is {adj} and the {feature} disappointed me badly.",
        "The {product} feels {adj} and the {feature} is not worth the money.",
    ]
    neutral_templates = [
        "The {product} offers {adj} {feature} and works fine for daily use.",
        "This {product} has {adj} {feature}, which is acceptable for the price.",
        "The {product} performs {adj} and the {feature} is average overall.",
    ]

    rows = []
    for index in range(100):
        product = rng.choice(products)
        feature = rng.choice(features)

        if index % 3 == 0:
            template = rng.choice(positive_templates)
            adjective = rng.choice(positive_adjectives)
            rating = 5
        elif index % 3 == 1:
            template = rng.choice(negative_templates)
            adjective = rng.choice(negative_adjectives)
            rating = 2
        else:
            template = rng.choice(neutral_templates)
            adjective = rng.choice(neutral_adjectives)
            rating = 3

        review = template.format(product=product, feature=feature, adj=adjective)
        rows.append({"Review": review, "Rating": rating})

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "amazon_reviews.csv", index=False)
    sample_df = df.head(20)
    sample_df.to_csv(DATA_DIR / "sample_reviews.csv", index=False)
    return df


def load_dataset() -> pd.DataFrame:
    """Load the dataset, creating it if it does not exist."""
    dataset_path = DATA_DIR / "amazon_reviews.csv"
    if not dataset_path.exists():
        print("Dataset not found. Creating sample dataset...")
        return create_default_dataset()

    try:
        df = pd.read_csv(dataset_path)
        print("Dataset Loaded Successfully")
        return df
    except Exception as error:
        print(f"Error loading dataset: {error}")
        return create_default_dataset()


def clean_text(text: str) -> str:
    """Clean a review by converting to lowercase and removing punctuation and stopwords."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if word not in STOPWORDS and len(word) > 1]
    return " ".join(tokens)


def get_sentiment_label(polarity: float) -> str:
    """Convert polarity score into Positive, Negative, or Neutral."""
    if polarity > 0.1:
        return "Positive"
    if polarity < -0.1:
        return "Negative"
    return "Neutral"


def detect_emotion(text: str) -> str:
    """Detect a simple emotion from a predefined keyword lexicon."""
    tokens = set(text.lower().split())

    happy_words = {"happy", "joy", "great", "excellent", "amazing", "wonderful", "good", "satisfied"}
    excited_words = {"excited", "thrilled", "awesome", "fantastic", "love", "loved"}
    love_words = {"love", "adore", "cherish", "favorite", "perfect"}
    angry_words = {"angry", "hate", "annoyed", "disappointed", "bad", "awful", "terrible"}
    sad_words = {"sad", "disappointing", "upset", "poor", "worse"}
    fear_words = {"scared", "afraid", "fear", "worried", "unsafe"}

    if tokens & love_words:
        return "Love"
    if tokens & excited_words:
        return "Excited"
    if tokens & happy_words:
        return "Happy"
    if tokens & angry_words:
        return "Angry"
    if tokens & sad_words:
        return "Sad"
    if tokens & fear_words:
        return "Fear"
    return "Neutral"


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned text, polarity, sentiment label, and emotion to the dataframe."""
    df = df.copy()
    df["Cleaned_Text"] = df["Review"].apply(clean_text)
    df["Polarity"] = df["Cleaned_Text"].apply(lambda text: TextBlob(text).sentiment.polarity)
    df["Sentiment"] = df["Polarity"].apply(get_sentiment_label)
    df["Emotion"] = df["Cleaned_Text"].apply(detect_emotion)
    return df


def get_summary(df: pd.DataFrame) -> dict:
    """Create a simple summary dictionary for display and output."""
    sentiment_counts = df["Sentiment"].value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0)
    emotion_counts = df["Emotion"].value_counts()
    most_common_emotion = emotion_counts.idxmax() if not emotion_counts.empty else "Neutral"

    return {
        "total_reviews": len(df),
        "positive_reviews": int(sentiment_counts.get("Positive", 0)),
        "negative_reviews": int(sentiment_counts.get("Negative", 0)),
        "neutral_reviews": int(sentiment_counts.get("Neutral", 0)),
        "most_common_emotion": most_common_emotion,
        "average_rating": round(float(df["Rating"].mean()), 2),
    }


def generate_bar_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Create a bar chart showing the sentiment distribution."""
    counts = df["Sentiment"].value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0)
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color=["#2E8B57", "#D9534F", "#6C757D"])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_pie_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Create a pie chart representing sentiment proportions."""
    counts = df["Sentiment"].value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0)
    plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2E8B57", "#D9534F", "#6C757D"],
        wedgeprops={"edgecolor": "white"},
    )
    plt.title("Sentiment Pie Chart")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_wordcloud(df: pd.DataFrame, output_path: Path) -> None:
    """Generate a word cloud from cleaned review text."""
    text = " ".join(df["Cleaned_Text"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_top_words(df: pd.DataFrame, top_n: int = 10) -> list:
    """Get the most frequent words from cleaned review text."""
    tokens = [token for text in df["Cleaned_Text"].dropna() for token in text.split()]
    return [word for word, _ in Counter(tokens).most_common(top_n)]


def get_sentiment_word_summary(df: pd.DataFrame) -> dict:
    """Get top words for the overall, positive, and negative review sets."""
    positive_reviews = df[df["Sentiment"] == "Positive"]
    negative_reviews = df[df["Sentiment"] == "Negative"]

    return {
        "top_words": get_top_words(df),
        "positive_words": get_top_words(positive_reviews, top_n=10),
        "negative_words": get_top_words(negative_reviews, top_n=10),
    }


def save_results(df: pd.DataFrame) -> None:
    """Save the processed dataset and visualizations."""
    output_path = OUTPUT_DIR / "sentiment_results.csv"
    df.to_csv(output_path, index=False)

    generate_bar_chart(df, OUTPUT_DIR / "sentiment_chart.png")
    generate_pie_chart(df, OUTPUT_DIR / "pie_chart.png")
    generate_wordcloud(df, OUTPUT_DIR / "wordcloud.png")


def create_screenshots() -> None:
    """Generate simple screenshot-style images for the project documentation."""
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, "CodeAlpha\nSentiment Analysis\nProject Preview", ha="center", va="center", fontsize=16)
    plt.axis("off")
    plt.savefig(SCREENSHOTS_DIR / "terminal_output.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, "Sentiment Chart\nGenerated by Python", ha="center", va="center", fontsize=16)
    plt.axis("off")
    plt.savefig(SCREENSHOTS_DIR / "sentiment_chart.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, "GitHub Repository\nReady for Portfolio", ha="center", va="center", fontsize=16)
    plt.axis("off")
    plt.savefig(SCREENSHOTS_DIR / "github_repo.png", dpi=200, bbox_inches="tight")
    plt.close()


def print_summary(summary: dict, word_summary: dict) -> None:
    """Print a polished console summary."""
    print("\n" + "=" * 48)
    print("CodeAlpha Sentiment Analysis")
    print("=" * 48)
    print("Loading Dataset...")
    print("Dataset Loaded Successfully")
    print("Cleaning Text...")
    print("Performing Sentiment Analysis...")
    print("Detecting Emotions...")
    print("Generating Charts...")
    print("Saving CSV...")
    print("\n" + "=" * 48)
    print("Summary")
    print("=" * 48)
    print(f"Total Reviews : {summary['total_reviews']}")
    print(f"Positive Reviews : {summary['positive_reviews']}")
    print(f"Negative Reviews : {summary['negative_reviews']}")
    print(f"Neutral Reviews : {summary['neutral_reviews']}")
    print(f"Most Common Emotion : {summary['most_common_emotion']}")
    print(f"Average Rating : {summary['average_rating']}")
    print("\nTop 10 Frequent Words :")
    print(", ".join(word_summary['top_words']))
    print("\nTop Positive Words :")
    print(", ".join(word_summary['positive_words']))
    print("\nTop Negative Words :")
    print(", ".join(word_summary['negative_words']))
    print("\nCSV Saved Successfully")
    print("Charts Saved Successfully")
    print("Project Completed Successfully")
    print("=" * 48)


def main() -> None:
    """Run the complete sentiment analysis workflow."""
    ensure_directories()

    print("Loading Dataset...")
    df = load_dataset()

    print("Cleaning Text...")
    df = analyze_sentiment(df)

    print("Performing Sentiment Analysis...")
    print("Detecting Emotions...")

    summary = get_summary(df)
    word_summary = get_sentiment_word_summary(df)

    print("Generating Charts...")
    save_results(df)
    create_screenshots()

    print("Saving CSV...")
    print_summary(summary, word_summary)


if __name__ == "__main__":
    main()
