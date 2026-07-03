"""
50_senegal_v2_nlp.py
=====================
NLP companion to volet 2 (decentralised teacher management).

Pipeline (R -> Python -> R, file-based exchange, same pattern as Projet 20):
  1. Read the perception corpus written by the R notebook
       files/projet_50/verbatims.csv  (columns: id, ia_id, g_ia, texte)
  2. Multilingual sentence embeddings (sentence-transformers)
  3. Topic extraction with BERTopic  -> the "bottlenecks" named by the data
  4. Multilingual sentiment scoring   -> perceived quality of management
  5. Write results back for R to read
       files/projet_50/verbatims_nlp.csv (id, ia_id, g_ia, topic, topic_label, sentiment)
       files/projet_50/topic_info.csv    (Topic, Count, Name)

Rationale for using Python here (rather than R): topic modelling on sentence
embeddings (sentence-transformers + BERTopic + HDBSCAN) and transformer-based
multilingual sentiment are more mature and simpler in Python. Everything else in
the series stays in R.

Dependencies:
  pip install sentence-transformers bertopic transformers torch pandas
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path("files/projet_50")
IN_CSV   = DATA_DIR / "verbatims.csv"
OUT_NLP  = DATA_DIR / "verbatims_nlp.csv"
OUT_TOP  = DATA_DIR / "topic_info.csv"

EMB_MODEL       = "paraphrase-multilingual-MiniLM-L12-v2"          # light, FR-capable
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"  # 1..5 stars


def load_corpus(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["texte"] = df["texte"].fillna("").astype(str)
    print(f"Loaded {len(df)} verbatims from {path}")
    return df


def embed(texts):
    """Multilingual sentence embeddings."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL)
    return model.encode(texts, show_progress_bar=True)


def topics(texts, embeddings):
    """BERTopic on precomputed embeddings. Returns (topics, model)."""
    from bertopic import BERTopic
    # min_topic_size kept modest: the corpus is small and templated.
    model = BERTopic(language="multilingual", min_topic_size=20, verbose=False)
    topic_ids, _ = model.fit_transform(texts, embeddings=embeddings)
    return topic_ids, model


def sentiment(texts):
    """Map 1..5 star multilingual sentiment onto a [-1, 1] score."""
    from transformers import pipeline
    clf = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, truncation=True)
    scores = []
    for i in range(0, len(texts), 64):                     # batch to bound memory
        batch = texts[i:i + 64]
        for r in clf(batch):
            stars = int(r["label"][0])                     # "4 stars" -> 4
            scores.append((stars - 3) / 2.0)               # 1->-1 ... 5->+1
    return scores


def main():
    if not IN_CSV.exists():
        raise SystemExit(f"Corpus introuvable : {IN_CSV}. Lance d'abord le chunk R 'verbatims-gen'.")

    df = load_corpus(IN_CSV)
    texts = df["texte"].tolist()

    print("Embedding ...")
    emb = embed(texts)

    print("Topic modelling ...")
    topic_ids, topic_model = topics(texts, emb)
    info = topic_model.get_topic_info()                    # Topic, Count, Name, ...
    label_map = dict(zip(info["Topic"], info["Name"]))

    print("Sentiment ...")
    df["sentiment"]   = sentiment(texts)
    df["topic"]       = topic_ids
    df["topic_label"] = df["topic"].map(label_map)

    df[["id", "ia_id", "g_ia", "topic", "topic_label", "sentiment"]].to_csv(OUT_NLP, index=False)
    info.to_csv(OUT_TOP, index=False)
    print(f"Wrote {OUT_NLP} and {OUT_TOP}")
    print(info.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
