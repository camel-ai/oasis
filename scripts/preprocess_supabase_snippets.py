#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "configs"
LEXICON_DIR = CONFIG_DIR / "lexicons"

ED_KEYWORDS = {
    "fast", "fasting", "restrict", "calorie", "scale", "bodycheck", "bones", "ana", "thinspo"
}
INCEL_KEYWORDS = {
    "females", "beta", "stacy", "chad", "blackpill", "looksmax", "hypergamy", "simp"
}
MISINFO_KEYWORDS = {
    "plandemic", "hoax", "toxic shot", "lab leak", "censorship", "vax", "chemtrail", "sheeple"
}
CONSPIRACY_KEYWORDS = {
    "deep state", "mkultra", "psyop", "puppet", "shadow", "coverup", "false flag", "ufo"
}
SUPPORT_KEYWORDS = {
    "stay strong", "recovery", "healing", "therapy", "reach out", "proud of you", "check in"
}

LABELS = [
    "benign",
    "recovery_support",
    "eating_disorder_risk",
    "incel_misogyny",
    "misinformation",
    "conspiracy",
]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def assign_label(text: str, reason: str, confidence: float) -> str:
    lowered = text.lower()
    reason_lower = reason.lower()
    if any(keyword in lowered for keyword in ED_KEYWORDS):
        return "eating_disorder_risk"
    if "self-harm" in reason_lower or any(keyword in lowered for keyword in SUPPORT_KEYWORDS):
        return "recovery_support"
    if any(keyword in lowered for keyword in INCEL_KEYWORDS) or "harassment" in reason_lower or "hate" in reason_lower:
        return "incel_misogyny"
    if any(keyword in lowered for keyword in MISINFO_KEYWORDS) or "misinfo" in reason_lower:
        return "misinformation"
    if any(keyword in lowered for keyword in CONSPIRACY_KEYWORDS) or "conspiracy" in reason_lower:
        return "conspiracy"
    if confidence < 0.65:
        return "benign"
    return "incel_misogyny"


def extract_spans(text: str, max_len: int = 200) -> List[str]:
    spans: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = clean_text(sentence)
        if 6 <= len(sentence.split()) <= max_len:
            spans.append(sentence)
    if not spans:
        spans.append(clean_text(text))
    return spans


def embed_texts(texts: List[str], model_name: str) -> np.ndarray | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    return np.asarray(embeddings)


def cluster_rows(matrix: np.ndarray, cluster_count: int) -> np.ndarray:
    if cluster_count <= 1:
        return np.zeros(matrix.shape[0], dtype=int)
    kmeans = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=42,
        batch_size=min(256, matrix.shape[0]),
        n_init=10,
    )
    return kmeans.fit_predict(matrix)


def build_lexicon_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    lexicon: Dict[str, List[str]] = {}
    for label in LABELS:
        entries = df[df["label"] == label]["content_clean"].tolist()
        phrases: List[str] = []
        for entry in entries:
            for span in extract_spans(entry):
                if len(phrases) >= 250:
                    break
                phrases.append(span)
            if len(phrases) >= 250:
                break
        lexicon[label] = phrases
    return lexicon


def compute_priors(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    priors: Dict[str, Dict[str, float]] = {}
    for label, group in df.groupby("label"):
        scores = group["confidence_norm"].to_numpy()
        if scores.size == 0:
            priors[label] = {"mean": 0.5, "std": 0.1, "min": 0.3, "max": 0.7}
            continue
        priors[label] = {
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std(ddof=0)), 4),
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
        }
    return priors


def summarize_clusters(df: pd.DataFrame, tfidf_matrix, feature_names: List[str]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for cluster_id, group in df.groupby("cluster_id"):
        mask = df["cluster_id"].values == cluster_id
        cluster_matrix = tfidf_matrix[mask]
        centroid = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        top_idx = centroid.argsort()[-10:][::-1]
        keywords = [feature_names[i] for i in top_idx if centroid[i] > 0]
        summary.append(
            {
                "cluster_id": int(cluster_id),
                "count": int(len(group)),
                "top_keywords": keywords[:8],
                "label_distribution": group["label"].value_counts(normalize=True).round(3).to_dict(),
            }
        )
    return summary


def write_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in df.to_dict("records"):
            payload = {
                "message_id": record.get("message_id"),
                "content": record.get("content_clean"),
                "label": record.get("label"),
                "confidence": record.get("confidence_norm"),
                "cluster_id": int(record.get("cluster_id", 0)),
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Supabase flagged snippets for RAG + lexicons")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_DIR / "Supabase Snippet Flagged Messages.csv"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=str(CONFIG_DIR / "rag" / "supabase_snippets.jsonl"),
    )
    parser.add_argument(
        "--lexicon-json",
        type=str,
        default=str(LEXICON_DIR / "supabase_lexicons.json"),
    )
    parser.add_argument(
        "--priors-json",
        type=str,
        default=str(LEXICON_DIR / "label_priors.json"),
    )
    parser.add_argument(
        "--cluster-summary",
        type=str,
        default=str(LEXICON_DIR / "supabase_cluster_summary.json"),
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap before dedupe (0=all)")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--max-clusters", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Supabase CSV not found at {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    df["content_clean"] = df["content"].map(clean_text)
    df = df[df["content_clean"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["content_clean"])
    df["confidence_norm"] = df.get("confidence", 70).fillna(70).astype(float) / 100.0
    df["reason_norm"] = df.get("reason", "").fillna("")
    df["label"] = df.apply(
        lambda row: assign_label(row["content_clean"], str(row["reason_norm"]), float(row["confidence_norm"])),
        axis=1,
    )

    tfidf = TfidfVectorizer(max_features=2048, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df["content_clean"])
    embeddings = embed_texts(df["content_clean"].tolist(), args.embedding_model)
    if embeddings is None:
        clustering_matrix = tfidf_matrix.toarray()
    else:
        clustering_matrix = embeddings
    cluster_count = max(1, min(args.max_clusters, clustering_matrix.shape[0] // 20 or 1))
    df["cluster_id"] = cluster_rows(clustering_matrix, cluster_count)

    lexicon_map = build_lexicon_map(df)
    priors = compute_priors(df)
    cluster_summary = summarize_clusters(df, tfidf_matrix, tfidf.get_feature_names_out())

    write_jsonl(df, Path(args.output_jsonl))
    LEXICON_DIR.mkdir(parents=True, exist_ok=True)
    with Path(args.lexicon_json).open("w", encoding="utf-8") as fh:
        json.dump(lexicon_map, fh, indent=2, ensure_ascii=False)
    with Path(args.priors_json).open("w", encoding="utf-8") as fh:
        json.dump(priors, fh, indent=2, ensure_ascii=False)
    with Path(args.cluster_summary).open("w", encoding="utf-8") as fh:
        json.dump(cluster_summary, fh, indent=2, ensure_ascii=False)

    print(f"Processed {len(df)} unique snippets; clusters={cluster_count}")
    print(f"Lexicons written to {args.lexicon_json}")
    print(f"Priors written to {args.priors_json}")


if __name__ == "__main__":
    main()
