# Business Entity Resolution — Team Pure Soul Reproducible Pipeline

This package produces the official competition outputs:
- `output/matching_results.tsv`: final entity matches scored on the leaderboard (5,819,900 accepted links)
- `output/candidate_pairs.tsv`: blocking candidate universe (99,695,890 candidate pairs, SHA-256: `168dd4fde9c2b76a028c731590080877401fb91357e9d8d7a1cbf7f351f3fe32`)

**Team Name:** Pure Soul  
**Final Production Model:** Phase 5/6 Rank-Gated Entity Model (V5)  
**Validation Scores:** DEV Macro $F_{0.5} = \mathbf{0.986641}$ | CONF Macro $F_{0.5} = \mathbf{0.986421}$ (Precision: 99.71%)

---

## 1. Environment & Dependencies

- Python 3.10+ (Windows or Linux, 16 GB+ RAM)
- Optional NVIDIA GPU for accelerated candidate generation (supports CUDA 12.x; CPU fallback supported)

```bash
python -m venv .venv
# Activate environment:
# Windows:
.venv\Scripts\activate
# Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2. Data Layout

Place the competition data files in the following structure:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

---

## 3. End-to-End Pipeline Execution

The pipeline executes through three stages:

### Step 1: Candidate Generation (Blocking)
Performs normalization (covering 9 Indic scripts, accent folding, legal suffixes), followed by sparse TF-IDF character 3-gram and address token indexing:
```bash
cd src
python prepare_data.py
python build_normalized.py train test
python candidates.py train test
```
*Outputs:* Exactly 10 candidate $S_1$ entities per query record (`cache/v2test_cands/`), capturing 98.70% of true links with an oracle ceiling of 0.996092.

### Step 2: Feature Engineering & Model Inference
Computes the complete 48-feature domain suite (combining string distances, address number logic, Indic phonetics, spurious conflict recovery, cluster decay rates, and probability margins) and executes the **Rank-Gated Entity Filter**:
```bash
python predict_v5_rank_gated.py
```
*Decision Rule:*
- **Rank 1 Candidate:** accepted if $p_{\text{set}} \ge 0.70$
- **Rank 2+ Candidate:** strictly gated at $p_{\text{set}} \ge 0.80$
- **One-S1-per-query constraint:** enforced across all records

*Outputs Generated:*
- `output/matching_results.tsv` (1,732,544 rows, SHA-256: `9e3244fe25c9b21919b880f770b0b2ea09a9817c05798d09ae3391c1cd6fcaac`)
- `output/candidate_pairs.tsv` (1,732,544 rows, SHA-256: `168dd4fde9c2b76a028c731590080877401fb91357e9d8d7a1cbf7f351f3fe32`)

### Step 3: Format & Compliance Verification
Run the official challenge validator:
```bash
cd ../..
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```
*Status:* **PASS — no blocking issues found. Safe to submit.**

---

## 4. Key Source Code Manifest (`src/`)

| Script / Module | Purpose |
| :--- | :--- |
| `predict_v5_rank_gated.py` | Complete end-to-end production test inference pipeline for V5 Rank-Gated Model |
| `normalization.py` | Universal Unicode Indic transliteration (9 scripts), accent folding, address cleaning |
| `retrieval.py` | TF-IDF character 3-gram & address token indexing, GPU CountSketch projection |
| `candidates.py` | Scaled blocking generator producing top-10 candidates per query |
| `features.py` | Pairwise string similarities, token overlap, and premise number extraction |
| `build_features.py` | Multiprocess batch feature generator |
| `indic_dictionary.py` | Learns and applies 526 domain phonetic transliteration mappings |
| `predict.py` | Formatter ensuring 100% strict subset constraints and tab-delimited grouped TSVs |
| `v2/` | Intermediate modular components and validation scoring suites |
