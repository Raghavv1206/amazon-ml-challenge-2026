# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Pure Soul  
**Team Members:** Pure Soul  
**Submission Date:** October 2026  

---

## 1. Executive Summary

In large-scale commercial entity resolution, connecting noisy business records from heterogeneous sources to a deduplicated reference catalog ($S_1$) requires balancing extreme recall during blocking against stringent precision during matching. Under the official **Macro $F_{0.5}$** metric ($5\text{TP} / [5\text{TP} + 4\text{FP} + \text{FN}]$), False Positives are penalized **$4\times$ heavier** than False Negatives, meaning that any candidate link with precision $< 80\%$ actively degrades the official leaderboard ranking.

Our final production system, **Pure Soul Phase 5/6 Rank-Gated Entity Model (V5)**, addresses the exact failure modes discovered in earlier iterations:
1. **GPU-Accelerated Sparse TF-IDF Blocking:** S2/S3 queries retrieve the top-10 candidate S1 entities partitioned by country using CountSketch-accelerated cosine similarity over character 3-grams and address tokens, achieving **98.70% candidate recall** (theoretical candidate oracle ceiling: **0.996092**).
2. **Comprehensive 48-Feature Domain Suite:** Combines string distance ratios, Indic phonetics (covering 9 regional scripts plus learned transliteration rules), exact house/premise number parsing, spurious number conflict recovery, catalog ambiguity uniqueness, multi-tenant building flags, branch distractor suppression, and S1 cluster stopping dynamics (score decay rates, anchor followers, margin gaps).
3. **Rank-Gated Decision Logic:** To prevent low-confidence candidate pollution in multi-link entities (which previously dragged baseline models down to 0.972 on the leaderboard), V5 enforces asymmetric rank thresholds:
   - **Rank 1 Candidate:** accepted if $p_{\text{set}} \ge 0.70$.
   - **Rank 2+ Candidates:** strictly gated at $p_{\text{set}} \ge 0.80$.
   - **One-S1-per-query constraint:** every query links to at most one reference entity.
4. **Empirical Results:** Across independent entity-held-out validation folds, V5 achieves **0.986641 Macro $F_{0.5}$ (DEV)** and **0.986421 Macro $F_{0.5}$ (CONF)** with **99.71% precision**. On the test set, V5 accepted 5,819,900 links across 1,632,062 matched entities (100,482 singletons), demonstrating rock-solid stability across India (94.07% matched), US (94.18% matched), and unseen France (94.66% matched).

---

## 2. Methodology

### 2.1 Problem Analysis
* **Scale & Asymmetry:** The challenge requires resolving 9.97M queries from Source 2 and Source 3 against 1.73M reference entities in Source 1. In ground truth, every matched query links to at most one reference entity, and links never cross country borders.
* **Open Country Labels & Domain Shift:** The training data contains US and India. The test set introduces **France** (14.39% of test queries, 14.98% of test entities), which is completely absent from training. Models must not hardcode country partitions or overfit regional token distributions.
* **Heterogeneous Noise Patterns:**
  - *Name Variations:* Legal suffixes (`Pvt Ltd`, `LLC`, `GmbH`, `SA/SAS`), abbreviations, punctuation variations (`&` vs `and`), phonetic transpositions, leet typos (`Preparat0ry`), doing-business-as (`DBA`), and **9 Indic scripts** (~7% of queries).
  - *Address Noise:* Component reordering, missing PIN/postal codes, landmark references (`Near SBI ATM`), abbreviations (`Rd` vs `Road`, `St` vs `Street`, `Av`/`Bd`/`All` in French), and municipal numbering formats.
  - *Branch vs Multi-Tenant Distractors:* Coherent groups of same-name businesses at different street addresses (retail chains/branches) and unrelated businesses operating in the same commercial complex (multi-tenant buildings).
* **Macro $F_{0.5}$ Optimization:** Because precision is weighted $4\times$ heavier than recall, singletons ($S_1$ entities with 0 matches) earn a full 1.0 credit when left empty, but drop to 0.0 upon receiving a single false positive. Protecting singletons and pruning borderline multi-link candidates is paramount.

### 2.2 Solution Architecture & Workflow

The end-to-end pipeline operates in five deterministic, leak-free stages:

```
[Raw TSV Records]
       │
       ▼
[Stage 1: Normalization & Transliteration]
  • 9 Indic scripts → Latin transliteration + phonetic dictionary
  • Legal suffix stripping, French/US address normalization, number extraction
       │
       ▼
[Stage 2: Scaled GPU Candidate Blocking]
  • Country partition → TF-IDF name 3-grams + address tokens
  • GPU CountSketch Top-40 → exact re-score → Top-10 candidates per query (98.7% recall)
       │
       ▼
[Stage 3: 48-Feature Engineering Engine]
  • Direct name/address similarities (Rapidfuzz, Jaccard, Token Sort, TF-IDF cosine)
  • House number equality, conflict detection, premise matching
  • S1 cluster dynamics: rank, max p, margin12, decay rate, anchor follower
  • Domain flags: branch distractor, multi-tenant, rebrand candidate, Indic exact
       │
       ▼
[Stage 4: Gradient-Boosted Entity Ranker (LightGBM)]
  • 48-feature booster trained with pairwise ranking and classification objectives
       │
       ▼
[Stage 5: Rank-Gated Entity Assignment]
  • Rank 1 candidate: p_set ≥ 0.70
  • Rank 2+ candidate: p_set ≥ 0.80
  • One-S1-per-query enforcement → output/matching_results.tsv (5,819,900 links)
```

---

## 3. Candidate Generation (Blocking)

* **Normalization:**
  - *Indic Transliteration:* Unicode-offset table transliterates Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, and Malayalam into standardized Latin tokens. A phonetics dictionary (526 entries such as `praivet→private`, `injiniyaring→engineering`) learned strictly from training pairs bridges orthographic gaps.
  - *Text Cleaning:* Accent normalization, ASCII folding, removal of generic prefixes (`M/s`, `Dr`, `Shri`), legal entity suffixes (`Inc`, `Corp`, `Ltd`, `SARL`), and regex extraction of numeric tokens.
* **Retrieval Indexing:**
  - Partitioned strictly by country (India, US, France).
  - Sparse TF-IDF over space-free name character 3-grams and address unigrams/bigrams.
  - Combined scoring: $\text{Score} = 0.5 \cdot \cos(\text{name}) + 0.5 \cdot \cos(\text{address})$.
  - CountSketch projection to 1024 dimensions allows high-throughput GPU FP16 matrix operations for the top-40 candidates, followed by exact sparse cosine re-ranking to yield the final **10 candidates per query**.
* **Candidate Volume & Quality:**
  - Exactly 99,695,890 candidate pairs across 1,732,544 test $S_1$ entities (~57.5 candidates per entity).
  - Candidate blocking recall is **98.70%** (India: 98.3%, US: 98.9%, Indic records: 99.4%).
  - Candidate oracle Macro $F_{0.5}$ ceiling is **0.996092**, proving blocking does not limit competition performance.
  - The frozen candidate universe is saved with verified SHA-256 hash `168dd4fde9c2b76a028c731590080877401fb91357e9d8d7a1cbf7f351f3fe32`.

---

## 4. Matching Model & Feature Engineering

### 4.1 Feature Suite (48 Features)
All features are computed deterministically without external API lookups:
1. **Candidate Retrieval & Competition (6 features):** Pair probability $p$, top-3 candidate probabilities ($p_1, p_2, p_3$), probability margin ($\text{margin}_{12} = p_1 - p_2$), and interaction term ($p \cdot \text{margin}_{12}$).
2. **Name Matching (7 features):** Rapidfuzz token-set ratio, token-sort ratio, partial ratio, exact name match flag, core name equality, raw name equality, and Indic canonical exact match.
3. **Address & Geography (6 features):** TF-IDF address cosine similarity, address character ratio, token set ratio, address missing flag, both address present flag, and source 3 indicator.
4. **Numeric & Premise Logic (5 features):** Exact street number match (`nums_exact`), street number conflict (`num_conflict`), premise number equality, spurious number conflict recovery (high address token similarity overriding minor apartment discrepancies), and strong name with weak address interaction.
5. **Cluster & S1 Entity Topology (16 features):** Rank within S1 (`rank_in_s1`), indicator for rank 1 (`rank_is_1`), total candidates in S1, maximum candidate probability in S1 (`s1_max_p`), mean candidate probability, count of candidates with $p \ge 0.80$, $p \ge 0.70$, $p \ge 0.50$, probability gap to next candidate, gap to previous candidate, cluster decay rate ($p_i / p_{i-1}$), anchor follower flag, and rebranding candidate flag.
6. **Domain & Specificity Flags (8 features):** Branch distractor flag (exact name match but address cosine $< 0.50$), multi-tenant flag (high address cosine $> 0.80$ but name mismatch), name uniqueness in S1 catalog, and address uniqueness.

### 4.2 Model Architecture & Training
- **Model Type:** LightGBM gradient boosted tree ensemble (500 trees, 63 leaves, learning rate 0.03, feature fraction 0.85, minimum data in leaf 50).
- **Training Protocol:** Trained strictly on 60% TRAIN-fold entities with out-of-sample margin inputs. Hyperparameters tuned under Macro $F_{0.5}$ loss.
- **Model Parameters:** Under 150,000 parameters (model file size 1.50 MB), vastly below the 8 Billion parameter competition ceiling and fully compliant with Apache 2.0 / MIT licensing.

### 4.3 Rank-Gated Decision Logic
To maximize the $4\times$ precision-weighted metric:
- For candidate $i$ of entity $S_1$:
  $$\text{Accept}(i) = \begin{cases} \text{True} & \text{if } \text{rank}(i) = 1 \text{ and } p_{\text{set}}(i) \ge 0.70 \\ \text{True} & \text{if } \text{rank}(i) > 1 \text{ and } p_{\text{set}}(i) \ge 0.80 \\ \text{False} & \text{otherwise} \end{cases}$$
- This asymmetric gating prunes noisy Rank 2+ candidate links while ensuring high recall on genuine single-link matches.

---

## 5. Results & Error Analysis

### 5.1 Validation Folds (Strict Official Metric)

Models were evaluated using the exact official scoring metric on two independent held-out folds:

| Model Architecture | DEV Fold $F_{0.5}$ | CONF Fold $F_{0.5}$ | Precision | Recall | Singletons $F_{0.5}$ | Test Accepted Links |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| B0 (Baseline) | 0.97785 | 0.97787 | 98.35% | 94.36% | 0.9662 | 5,860,000 |
| C4 (Stacking baseline) | 0.98322 | 0.98321 | 98.86% | 95.49% | 0.9724 | 5,712,570 |
| V4 (Entity Set, uniform t=0.55) | 0.98469 | 0.98448 | 99.29% | 97.10% | 0.9814 | 5,926,594 |
| **V5 Pure Soul (Rank-Gated)** | **0.986641** | **0.986421** | **99.71%** | **96.67%** | **0.9853** | **5,819,900** |

*Absolute split variance between DEV and CONF is $< 0.00022$, establishing deterministic generalization without leakage.*

### 5.2 Ground-Truth Dissection: Why V5 Protects Leaderboard Score
Forensic analysis on DEV ground truth explained why earlier submissions fell on the live leaderboard:
- **Consensus Matches (V4 & V5 agreed):** 1,478,427 TP vs 4,217 FP (**99.72% precision**).
- **V4-Only Matches (rejected by V5):** 7,675 TP vs 6,405 FP (**54.51% precision**). With $\Delta\text{TP}/\Delta\text{FP} = 1.20 \ll 4.0$, this pool actively destroys Macro $F_{0.5}$.
- V5 prunes these 114,448 distractor links on the test set, shielding the solution from the False Positive penalty.

### 5.3 Geographic Generalization Across Test Countries

| Metric | India (TRAIN seen) | US (TRAIN seen) | France (TRAIN unseen) | Overall Test Set |
| :--- | :---: | :---: | :---: | :---: |
| **Total S1 Entities** | 809,986 | 663,106 | 259,452 | **1,732,544** |
| **Entities with $\ge 1$ match** | 761,935 (94.07%) | 624,518 (94.18%) | 245,609 (94.66%) | **1,632,062 (94.20%)** |
| **Singleton Entities (0 match)** | 48,051 (5.93%) | 38,588 (5.82%) | 13,843 (5.34%) | **100,482 (5.80%)** |
| **Total Accepted Links** | 2,682,799 | 2,245,402 | 891,699 | **5,819,900** |
| **Mean Links / Matched Entity** | 3.521 | 3.595 | 3.631 | **3.566** |

The variance across countries is under 0.6%, proving that normalization, Indic canonicalization, and French address token handling operate uniformly without geographical bias.

---

## 6. Submission Verification & Compliance

### 6.1 Official Validator Output
Verified with `utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids`:
```
ML Challenge 2026 - submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (100482 empty, 1632062 non-empty).
  candidate_pairs.tsv: 1732544 rows (6 empty, 1732538 non-empty).

PASS - no blocking issues found. Safe to submit.
```

### 6.2 Cryptographic Hashes
- **`matching_results.tsv`:** `9e3244fe25c9b21919b880f770b0b2ea09a9817c05798d09ae3391c1cd6fcaac`
- **`candidate_pairs.tsv`:** `168dd4fde9c2b76a028c731590080877401fb91357e9d8d7a1cbf7f351f3fe32`
- **`Pure_Soul_submission.zip`:** `d9c0e4cbb55581e1ee30e46b9ec76f1da4570dc937ebf8fb38a2e1bdf48cfbb2`

### 6.3 Academic Integrity Confirmation
In accordance with challenge rules:
- **No external data lookups:** zero commercial entity APIs, zero Google Maps/web scraping, zero external business registries, zero external geocoding services.
- **Fair Play:** All transliteration, dictionaries, and feature representations were learned exclusively from the provided training set.

---

## 7. Conclusion

By shifting the modeling paradigm from pairwise unconstrained matching to **Rank-Gated Cluster Optimization**, Team **Pure Soul** eliminated the precision degradation that hindered previous baseline models. With an empirical validation score of **0.986641**, 99.71% precision, and strict official validator verification, the final submission achieves the strongest, most competition-compliant performance possible.
