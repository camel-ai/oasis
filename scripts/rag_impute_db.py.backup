import sqlite3
import pandas as pd
import yaml
import os
import re
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = "../data/oasis_mvp_gemini.db"
PHRASE_BANK_PATH = "../data/label_tokens_static_bank.yaml"
JSONL_OUT = "../out/posts_imputed_rag.jsonl"

IMPUTED_COL =  "text_rag_imputed"

TOKEN_PATTERN = re.compile(r"<(LBL:[A-z0-9_]+)>")

def load_phrase_bank(path):
    """"Load label_tokens_static_bank.yaml and return, We are building mini tfidf model for each token or label for the phrase bank"""
    with open(path, "r", encoding= "utf-8") as f:
        raw = yaml.safe_load(f)

    bank = {}
    for token, phrases in raw.items():
        bank[token] = phrases or []

    vectorizers = {}
    matrices = {}

    for token,phrases in bank.items():
        if not phrases:
            continue
        vec = TfidfVectorizer()
        X  = vec.fit_transform(phrases)
        vectorizers[token] = vec
        matrices[token] = X

    return bank, vectorizers, matrices
    

# def choose_phrase_rag(token, bank, vectorizers, matrices, context, post_id):

#     phrases = bank.get(token, [])
#     if not phrases:
#         return f"<{token}"

#     vec = vectorizers.get(token)
#     X = matrices.get(token)

#     if vec is None or X is None:
#         # Fallback: deterministic hash choice
#         h = hashlib.md5(f"{post_id}-{token}".encode()).hexdigest()
#         return phrases[int(h, 16) % len(phrases)]
    
#     try: 

#         context_vec = vec.transform([context])
#         sims = cosine_similarity(context_vec, X)[0]
#         best_idx = sims.argmax()
#         return phrases[best_idx]
#     except:
#         h = hashlib.md5(f"{post_id}-{token}".encode()).hexdigest()
#         return phrases[int(h, 16) % len(phrases)]


def choose_phrase_rag(token, bank, vectorizers, matrices, context, post_id):
    phrases = bank.get(token, [])
    if not phrases:
        print(f"[DEBUG] No phrases for {token} — returning placeholder")
        return f"<{token}>"

    vec = vectorizers.get(token)
    X = matrices.get(token)

    if vec is None or X is None:
        print(f"[DEBUG] NO TF-IDF for {token} — using deterministic hash fallback")
        h = hashlib.md5(f"{post_id}-{token}".encode()).hexdigest()
        return phrases[int(h, 16) % len(phrases)]

    try:
        context_vec = vec.transform([context])
        sims = cosine_similarity(context_vec, X)[0]
        best_idx = sims.argmax()
        
        print(f"[DEBUG] USED TF-IDF for {token} — best idx = {best_idx}, phrase = {phrases[best_idx]}")
        
        return phrases[best_idx]

    except Exception as e:
        print(f"[DEBUG] ERROR in TF-IDF for {token}: {e} — using hash fallback")
        h = hashlib.md5(f"{post_id}-{token}".encode()).hexdigest()
        return phrases[int(h, 16) % len(phrases)]


def rag_impute_text(text, post_id,bank ,vectorizers, matrices):
    if not text:
        return text
    
    def repl(match):
        full = match.group(0)
        token = match.group(1)

        phrase = choose_phrase_rag(token, bank, vectorizers, matrices, text, post_id)
        return phrase

    return TOKEN_PATTERN.sub(repl, text)

def main():
    os.makedirs(os.path.dirname(JSONL_OUT) or ".", exist_ok=True)

    # 1) Load phrase bank + TF-IDF indices
    print(f"Loading phrase bank from {PHRASE_BANK_PATH} ...")
    bank, vectorizers, matrices = load_phrase_bank(PHRASE_BANK_PATH)
    print("Tokens loaded:", list(bank.keys()))

    # 2) Connect to DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 3) Ensure post table has our imputed column
    cur.execute("PRAGMA table_info(post)")
    cols = [r[1] for r in cur.fetchall()]
    if IMPUTED_COL not in cols:
        print(f"Adding column {IMPUTED_COL} to post table ...")
        cur.execute(f"ALTER TABLE post ADD COLUMN {IMPUTED_COL} TEXT")
        conn.commit()

    # 4) Read all posts
    df = pd.read_sql_query("SELECT post_id, content FROM post ORDER BY post_id", conn)
    print(f"Found {len(df)} posts in DB")

    # 5) Impute + write back + export JSONL
    out_path = JSONL_OUT
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            pid = int(row["post_id"])
            original = row["content"]
            new_text = rag_impute_text(original, pid, bank, vectorizers, matrices)

            # update DB
            cur.execute(
                f"UPDATE post SET {IMPUTED_COL}=? WHERE post_id=?",
                (new_text, pid),
            )

            record = {"post_id": pid, "text_rag_imputed": new_text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    conn.commit()
    conn.close()
    print(f"✅ RAG imputation complete. Wrote {out_path}")

if __name__ == "__main__":
    import json
    main()
