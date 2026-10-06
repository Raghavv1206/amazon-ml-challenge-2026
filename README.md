# Amazon ML Challenge 2026: Business Entity Resolution

**Team Name:** Pure Soul  
**Final Production Model:** Phase 5/6 Rank-Gated Entity Model (V5)  
**Final Submission Archive:** `Pure_Soul_submission.zip`  
**Official Metric (Held-out splits):**  
- **DEV Fold Macro $F_{0.5}$:** `0.986641` (Precision: 99.71%, Recall: 96.67%)
- **CONF Fold Macro $F_{0.5}$:** `0.986421` (Precision: 99.71%, Recall: 96.65%)
- **Official Validator:** **PASS (Exit code 0, `--check-ids` verified)**

---

## 1. Problem Overview

Given business records across three noisy, heterogeneous data sources ($S_1$, $S_2$, $S_3$), resolve all matching records from $S_2$ and $S_3$ to their corresponding reference entity in $S_1$. The challenge evaluates submissions using **Macro $F_{0.5}$**, where False Positives carry a **$4\times$ heavier penalty** than False Negatives ($5\text{TP} / [5\text{TP} + 4\text{FP} + \text{FN}]$).

Singletons ($S_1$ entities with 0 matches) receive a full 1.0 credit when correctly predicted as empty, but drop to 0.0 upon receiving even one false match.

---

## 2. Final Submission Structure

The final competition archive `Pure_Soul_submission.zip` conforms strictly to the official challenge specification:

```
Pure_Soul_submission.zip
├── output/
│   ├── matching_results.tsv   # Final entity matches (1,732,544 rows, 5,819,900 accepted links)
│   └── candidate_pairs.tsv    # Blocking candidate universe (99,695,890 pairs)
├── code/
│   └── business_entity_resolution/
│       ├── src/               # Full source code for blocking, features, and inference
│       ├── models/            # Trained LightGBM booster (model_phase56.txt)
│       ├── README.md          # Exact end-to-end reproduction guide
│       └── requirements.txt   # Pinned dependencies
└── Documentation_template.md  # Detailed methodology and error analysis write-up
```

---

## 3. End-to-End Pipeline Summary

1. **Normalization & Phonetic Transliteration:**
   - Universal Unicode transliteration table for 9 Indic scripts into Latin representations.
   - Learned phonetic dictionary (526 token mappings) capturing phonetic variations (`praivet→private`, `injiniyaring→engineering`).
   - Suffix and noise stripping (legal designations, accents, `DBA`, leet typos).
2. **GPU-Accelerated Candidate Blocking:**
   - Per-country sparse TF-IDF over name character 3-grams and address tokens.
   - GPU CountSketch projection generating the top-10 candidate entities per query.
   - Achieves **98.70% candidate recall** with a candidate oracle ceiling of **0.996092**.
3. **48-Feature Domain Engine:**
   - Rapidfuzz string distance metrics (token sort, token set, partial ratios).
   - Numeric & premise matching (exact street numbers, conflict detection, premise equality).
   - Cluster topology and stopping dynamics (S1 rank, max probability, margins, decay rates).
   - Domain flags (branch distractors, multi-tenant complexes, catalog uniqueness).
4. **Rank-Gated Decision Logic:**
   - **Rank 1:** $p_{\text{set}} \ge 0.70$
   - **Rank 2+:** $p_{\text{set}} \ge 0.80$ (pruning distractor links that degrade Macro $F_{0.5}$)
   - Strict one-S1-per-query enforcement.

---

## 4. Verification and Reproducibility

To re-verify the final outputs against the official test set:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

### Cryptographic Hashes
- `matching_results.tsv`: `9e3244fe25c9b21919b880f770b0b2ea09a9817c05798d09ae3391c1cd6fcaac`
- `candidate_pairs.tsv`: `168dd4fde9c2b76a028c731590080877401fb91357e9d8d7a1cbf7f351f3fe32`
- `Pure_Soul_submission.zip`: Verified compliant archive.
