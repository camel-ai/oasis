# posts.jsonl Dataset

## Pipeline Overview
1. **Persona synthesis** (`oasis/scripts/build_persona_rag.py`)
   - Clusters `oasis/data/personality.csv` seeds with MiniBatchKMeans (12 clusters).
   - Generates 130 personas (100 primary + 30 backup) with explicit `allowed_labels`, style cues, fallback utterances, and variant IDs.
   - Outputs CSVs plus a refreshed persona RAG corpus under `oasis/data/rag_corpus/`.
2. **Supabase snippet preprocessing** (`oasis/scripts/preprocess_supabase_snippets.py`)
   - Deduplicates *Supabase Snippet Flagged Messages.csv*, samples 11,420 rows, embeds with `all-MiniLM-L6-v2`, clusters into 20 topics, and saves lexicons + label priors in `oasis/configs/lexicons/`.
3. **Post generation with placeholder substitution** (`oasis/scripts/build_posts_dataset.py`)
   - Loads the 100 primary personas, plans 1,000 posts (exactly 200 harmful, 40 multi-label harmful) with realistic thread structures.
   - Supports Gemini generation (`--use-gemini`) but defaults to an offline composer that blends persona RAG snippets with Supabase spans.
   - Replaces placeholder tokens via nearest-class lexicons, assigns labels/features, and exports `posts.jsonl` plus `oasis/configs/lexicons/posts_dataset_metrics.json`.
4. **Validation** (`oasis/scripts/validate_posts_dataset.py`)
   - Confirms totals, harmful ratios, persona coverage, and split distributions; emits QA metrics JSON.

## Key Metrics
- Total posts: **1,000**
- Harmful posts: **200** (20% of corpus)
- Multi-label harmful posts: **40** (20% of harmful slice)
- Users represented: **100** unique personas
- Split distribution: train 750 / val 150 / test 100
- Label counts:
  - Benign 480
  - Recovery/Support 320
  - Conspiracy 50
  - Misinformation 55
  - Eating-disorder risk 67
  - Incel/misogyny 68
- Placeholder coverage: 100% of labeled samples had spans resolved from Supabase lexicons.

## Running the Pipeline
```bash
# 1. Personas
python oasis/scripts/build_persona_rag.py

# 2. Supabase preprocessing (optional arguments: --max-rows, --max-clusters)
python oasis/scripts/preprocess_supabase_snippets.py --max-rows 12000 --max-clusters 20

# 3. Dataset build (offline mode)
python oasis/scripts/build_posts_dataset.py --output posts.jsonl
# or enable Gemini
python oasis/scripts/build_posts_dataset.py --use-gemini --temperature 0.8 --output posts.jsonl

# 4. Validation
python oasis/scripts/validate_posts_dataset.py --input posts.jsonl --output oasis/configs/lexicons/posts_dataset_metrics.json
```

## Notes & Assumptions
- Harmful placeholders are always substituted with real Supabase spans to maintain lexical realism.
- Fallback offline composer stitches persona RAG snippets with lexicon phrases so the dataset can be regenerated without external APIs.
- Gemini support is lazy-imported; installing the full `oasis` runtime plus `google-genai` is required for live LLM calls.
- `posts.jsonl` aligns with the schema described in the goal statement (thread metadata, provenance, probabilities, etc.).
