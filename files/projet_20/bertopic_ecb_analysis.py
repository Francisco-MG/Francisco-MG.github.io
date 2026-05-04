#!/usr/bin/env python3
"""
BERTopic analysis of ECB monetary policy statements.
Produces CSV/PNG/JSON outputs consumed by the Quarto portfolio (project 20).

Now also computes Cv coherence (Röder et al., 2015) for LDA, STM, and
BERTopic on the same corpus — providing a comparable, standard-literature
metric across all three modelling approaches.

Requirements:
    pip install bertopic sentence-transformers pandas matplotlib gensim

Usage:
    python bertopic_ecb_analysis.py \\
        --corpus files/corpus_ecb.csv \\
        --lda-terms files/lda_top_terms.csv \\
        --stm-terms files/stm_top_terms.csv \\
        --outdir files/results_bertopic/

Author: Francisco Martin-Gomez
Date: 2025-05-14
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NR_TOPICS       = 5
MIN_TOPIC_SIZE  = 3
SEED            = 2026


# ── Tokenisation (must match R preprocessing for fair Cv comparison) ─────────

POLICY_STOPWORDS = {
    "ecb", "governing", "council", "euro", "area",
    "decided", "continue", "remains", "remain",
    "percent", "quarter", "year", "basis", "points",
}

EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "in",
    "to", "for", "on", "at", "by", "with", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "this", "that",
    "these", "those", "it", "its", "we", "our", "us", "they", "their", "them",
    "he", "she", "his", "her", "i", "me", "my", "you", "your",
}


def tokenise(text: str) -> list:
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens
            if t not in EN_STOPWORDS
            and t not in POLICY_STOPWORDS
            and len(t) > 2]


def load_corpus(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Corpus loaded: {len(df)} documents, {df['phase'].nunique()} phases")
    print(f"  Period: {df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def fit_bertopic(docs):
    print(f"\n-- Embedding with {EMBEDDING_MODEL} --")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(docs, show_progress_bar=True)
    print(f"  Embeddings: {embeddings.shape}")

    print(f"\n-- Fitting BERTopic (nr_topics={NR_TOPICS}) --")
    model = BERTopic(
        embedding_model=embedder,
        nr_topics=NR_TOPICS,
        min_topic_size=MIN_TOPIC_SIZE,
        verbose=True,
        calculate_probabilities=True,
    )
    topics, probs = model.fit_transform(docs, embeddings)

    info = model.get_topic_info()
    n_real = len(info[info["Topic"] != -1])
    n_outlier = (np.array(topics) == -1).sum()
    print(f"  Topics: {n_real} (+1 outlier with {n_outlier} docs)")

    return model, embeddings, topics, probs


# ── Cv coherence (Roder et al., 2015) ────────────────────────────────────────

def compute_cv_coherence(topics_terms, tokenised_corpus, dictionary,
                          label="model"):
    filtered = []
    for terms in topics_terms:
        present = [t for t in terms if t in dictionary.token2id]
        if len(present) >= 2:
            filtered.append(present)

    if not filtered:
        print(f"  X {label}: no valid topics for coherence")
        return {"mean": None, "per_topic": [], "n_valid": 0}

    cm = CoherenceModel(
        topics=filtered,
        texts=tokenised_corpus,
        dictionary=dictionary,
        coherence="c_v",
        topn=10,
    )
    per_topic = cm.get_coherence_per_topic()
    mean_cv = float(np.mean(per_topic))

    print(f"  {label}: Cv = {mean_cv:.4f} (n_valid={len(filtered)})")
    return {
        "mean": round(mean_cv, 4),
        "per_topic": [round(c, 4) for c in per_topic],
        "n_valid": len(filtered),
    }


def load_topic_terms_csv(path):
    if not Path(path).exists():
        return []
    df = pd.read_csv(path)
    return df.groupby("topic")["term"].apply(list).tolist()


def extract_bertopic_terms(model, top_n=10):
    info = model.get_topic_info()
    topics_terms = []
    for topic_id in info["Topic"]:
        if topic_id == -1:
            continue
        terms = [w for w, _ in model.get_topic(topic_id)[:top_n]]
        topics_terms.append(terms)
    return topics_terms


# ── Export ───────────────────────────────────────────────────────────────────

def export_topic_info(model, outdir):
    info = model.get_topic_info()
    info.to_csv(outdir / "bertopic_topic_info.csv", index=False)
    print(f"  -> bertopic_topic_info.csv ({len(info)} topics)")

    rows = []
    for topic_id in info["Topic"]:
        if topic_id == -1:
            continue
        terms = model.get_topic(topic_id)
        for rank, (word, score) in enumerate(terms[:10], 1):
            rows.append({"topic": topic_id, "rank": rank,
                         "term": word, "score": round(score, 4)})
    pd.DataFrame(rows).to_csv(outdir / "bertopic_top_terms.csv", index=False)
    print(f"  -> bertopic_top_terms.csv ({len(rows)} pairs)")


def export_doc_topics(corpus, topics, probs, outdir):
    doc_topics = corpus[["doc_id", "date", "phase"]].copy()
    doc_topics["topic"] = topics
    doc_topics["max_prob"] = probs.max(axis=1) if probs.ndim > 1 else probs
    doc_topics.to_csv(outdir / "bertopic_doc_topics.csv", index=False)
    print(f"  -> bertopic_doc_topics.csv ({len(doc_topics)} docs)")


def plot_topic_barchart(model, outdir):
    try:
        fig = model.visualize_barchart(top_n_topics=5, n_words=8)
        fig.write_image(str(outdir / "bertopic_barchart.png"),
                        width=900, height=600, scale=2)
        print("  -> bertopic_barchart.png")
    except Exception as e:
        print(f"  ! Barchart export failed (kaleido needed?): {e}")


def plot_topic_timeline(corpus, topics, outdir):
    df = corpus[["date", "phase"]].copy()
    df["topic"] = topics
    df["quarter"] = df["date"].dt.to_period("Q")
    props = (df.groupby(["quarter", "topic"])
              .size().unstack(fill_value=0))
    props = props.div(props.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]

    x = range(len(props))
    x_labels = [str(q) for q in props.index]

    bottom = np.zeros(len(props))
    for i, col in enumerate(props.columns):
        if col == -1:
            continue
        c = colors[i % len(colors)]
        ax.bar(x, props[col].values, bottom=bottom, color=c,
               label=f"Topic {col}", alpha=0.85, width=1.0)
        bottom += props[col].values

    ax.set_xlabel("Quarter")
    ax.set_ylabel("Topic Proportion")
    ax.set_title("BERTopic - Topic Prevalence Over Time (ECB Statements)")
    ax.legend(loc="upper right", fontsize=8)

    tick_positions = list(range(0, len(x_labels), 4))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([x_labels[i] for i in tick_positions],
                       rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(outdir / "bertopic_timeline.png", dpi=150)
    plt.close()
    print("  -> bertopic_timeline.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BERTopic + Cv coherence on ECB statements")
    parser.add_argument("--corpus", default="files/corpus_ecb.csv")
    parser.add_argument("--lda-terms", default="files/lda_top_terms.csv")
    parser.add_argument("--stm-terms", default="files/stm_top_terms.csv")
    parser.add_argument("--outdir", default="files/results_bertopic")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Auto-resolve LDA/STM term paths if defaults were not overridden ─────
    # When the user passes only --corpus, look for the companion CSVs next to
    # the corpus itself (the Scenario B convention: one folder per project).
    corpus_dir = Path(args.corpus).parent
    DEFAULT_LDA = parser.get_default("lda_terms")
    DEFAULT_STM = parser.get_default("stm_terms")

    if args.lda_terms == DEFAULT_LDA and not Path(DEFAULT_LDA).exists():
        candidate = corpus_dir / "lda_top_terms.csv"
        if candidate.exists():
            args.lda_terms = str(candidate)

    if args.stm_terms == DEFAULT_STM and not Path(DEFAULT_STM).exists():
        candidate = corpus_dir / "stm_top_terms.csv"
        if candidate.exists():
            args.stm_terms = str(candidate)

    print(f"  LDA terms: {args.lda_terms}")
    print(f"  STM terms: {args.stm_terms}")

    # ── 1. Load corpus and tokenise (shared base for Cv) ────────────────────
    corpus = load_corpus(args.corpus)
    tokenised = [tokenise(text) for text in corpus["text"].tolist()]
    dictionary = Dictionary(tokenised)
    print(f"  Vocabulary size: {len(dictionary)}")

    # ── 2. Fit BERTopic ─────────────────────────────────────────────────────
    model, embeddings, topics, probs = fit_bertopic(corpus["text"].tolist())

    # ── 3. Export BERTopic results ──────────────────────────────────────────
    print("\n-- Exporting BERTopic results --")
    export_topic_info(model, outdir)
    export_doc_topics(corpus, topics, probs, outdir)
    plot_topic_barchart(model, outdir)
    plot_topic_timeline(corpus, topics, outdir)

    # ── 4. Compute Cv coherence for the THREE models ────────────────────────
    print("\n-- Computing Cv coherence (Roder et al., 2015) --")

    bert_terms = extract_bertopic_terms(model, top_n=10)
    coh_bert = compute_cv_coherence(bert_terms, tokenised, dictionary, "BERTopic")

    coh_lda = {"mean": None, "per_topic": [], "n_valid": 0}
    if Path(args.lda_terms).exists():
        coh_lda = compute_cv_coherence(
            load_topic_terms_csv(args.lda_terms),
            tokenised, dictionary, "LDA")
    else:
        print(f"  ! LDA terms not found: {args.lda_terms}")

    coh_stm = {"mean": None, "per_topic": [], "n_valid": 0}
    if Path(args.stm_terms).exists():
        coh_stm = compute_cv_coherence(
            load_topic_terms_csv(args.stm_terms),
            tokenised, dictionary, "STM")
    else:
        print(f"  ! STM terms not found: {args.stm_terms}")

    # ── 5. Export coherence scores ──────────────────────────────────────────
    with open(outdir / "coherence_scores.json", "w") as f:
        json.dump({
            "metric":      "Cv (Roder et al. 2015)",
            "topn":        10,
            "n_documents": len(corpus),
            "vocab_size":  len(dictionary),
            "models": {"LDA": coh_lda, "STM": coh_stm, "BERTopic": coh_bert},
        }, f, indent=2)
    print("  -> coherence_scores.json")

    # ── 6. Summary ──────────────────────────────────────────────────────────
    summary = {
        "n_documents":      len(corpus),
        "n_topics":         len(model.get_topic_info()) - 1,
        "n_outliers":       int((np.array(topics) == -1).sum()),
        "embedding_model":  EMBEDDING_MODEL,
        "cv_lda":           coh_lda["mean"],
        "cv_stm":           coh_stm["mean"],
        "cv_bertopic":      coh_bert["mean"],
    }
    with open(outdir / "bertopic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  -> bertopic_summary.json")

    print(f"\nDone. Cv: LDA={coh_lda['mean']} | "
          f"STM={coh_stm['mean']} | BERTopic={coh_bert['mean']}")


if __name__ == "__main__":
    main()
