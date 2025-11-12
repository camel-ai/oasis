import sqlite3, pandas as pd, yaml, re, hashlib, json, os

DB_PATH = "data/oasis_mvp_gemini.db"
PHRASE_BANK_PATH = "../data/label_tokens_static_bank.yaml"
IMPUTED_TABLE = "posts_imputed"
JSONL_OUT = "../out/posts_mvp_imputed.jsonl"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
tables = [t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print("\n=== TABLE FOUND ===")
for t in tables:
    print("-", t)

for t in tables:
    try:
        df = pd.read_sql_query(f"SELECT * FROM {t} LIMIT 5", conn)
        print(f"\n=== PREVIEW TABLE {t} ===")
        print(df.head())
    except Exception as e:
        print(f"COULD NOT FIND {t}: {e}")


def find_text_columns(conn):
    text_cols = []
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [r[1] for r in cur.fetchall()]
        for c in cols:
            try: 
                sample = cur.execute(f"SELECT {c} FROM {t} LIMIT 5").fetchall()
                if any("<LBL:" in str(x) for row in sample for x in row):
                    text_cols.append((t, c))
            except:
                continue
    return text_cols

text_cols = find_text_columns(conn)
print(f"\n ==COLUMNS LIKELY CONTAINER PLACEHOLDER ")
for t, c in text_cols:
    print(f"TABLES: {t}, COLUMS: {c}") 


def load_phrase_bank(path):
    with open(path, "r", encoding="utf-8") as f:
        bank = yaml.safe_load(f)

    norm = {}
    for k, v in bank.items():
        key = k.replace("<", "").replace(">", "")
        norm[key] = v
    return norm

def deterministic_choice(token, post_id, phrases):
    if not phrases: return f"<{token}>"
    h = hashlib.md5(f"{post_id}-{token}".encode()).hexdigest()
    return phrases[int(h, 16) % len(phrases)]
 
def impute(text, post_id, bank):
    def repl(match):
        token = match.group(0)[1:-1]
        phrases = bank.get(token, [])
        return deterministic_choice(token, post_id, phrases)
    return re.sub(r"<LBL:[A-Z0-9_]+>", repl, text or "")

# Pick first detected table+column
if text_cols:
    table, col = text_cols[0]
    print(f"\nImputing placeholders in {table}.{col} ...")

    # Load phrase bank
    bank = load_phrase_bank(PHRASE_BANK_PATH)
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    if "text_imputed" not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN text_imputed TEXT")

    cur.execute(f"SELECT post_id, {col} FROM {table}")
    rows = cur.fetchall()

    os.makedirs(os.path.dirname(JSONL_OUT) or ".", exist_ok=True)
    with open(JSONL_OUT, "w", encoding="utf-8") as f:
        for pid, text in rows:
            new_text = impute(text, str(pid), bank)
            conn.execute(f"UPDATE {table} SET text_imputed=? WHERE post_id=?", (new_text, pid))
            f.write(json.dumps({"post_id": pid, "text": new_text}, ensure_ascii=False) + "\n")
    conn.commit()
    print(f"✅ Imputation complete — updated {table}.text_imputed and wrote {JSONL_OUT}")
else:
    print("⚠️ No columns with placeholders found.")