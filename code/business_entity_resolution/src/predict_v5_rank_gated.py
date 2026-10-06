#!/usr/bin/env python3
"""
make_v5_leaderboard_submission.py - End-to-end production test inference and submission generator
for the Leaderboard-Optimized Phase 5/6 Rank-Gated Entity Model.

Features:
  - Phase 3/4: Branch distractor flag, multi-tenant flag, address number matching, catalog frequency
  - Phase 5: Indic canonicalization, trade-name / rebranding flag, spurious premise conflict recovery
  - Phase 6: S1 cluster stopping dynamics, probability margins, cluster decay rate, anchor followers
  - Decision Logic: Rank-Gated Entity Filter (Rank 1 >= 0.70, Rank 2+ >= 0.80)

Outputs:
  - output/v5_leaderboard_optimized/matching_results.tsv
  - output/v5_leaderboard_optimized/candidate_pairs.tsv (exact verified SHA-256)
  - Pure_Soul_v5_leaderboard_submission.zip
"""

import sys, os, time, re, gc, hashlib, shutil, zipfile
from itertools import zip_longest
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb

# Locate root directory robustly
_here = Path(__file__).resolve().parent
ROOT = _here.parents[1] if (_here.parents[1] / "output").exists() else _here.parent if (_here.parent / "output").exists() else _here
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "output" / "v5_leaderboard_optimized"
ARCHIVE_PATH = ROOT / "Pure_Soul_v5_leaderboard_submission.zip"

sys.path.insert(0, str(_here))
from predict import write_grouped

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def indic_clean(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r'ee+', 'i', s)
    s = re.sub(r'oo+', 'u', s)
    s = re.sub(r'ou|au', 'o', s)
    s = re.sub(r'ai|ay\b', 'e', s)
    s = re.sub(r'ph', 'f', s)
    s = re.sub(r'bh', 'b', s)
    s = re.sub(r'dh', 'd', s)
    s = re.sub(r'th', 't', s)
    s = re.sub(r'kh', 'k', s)
    s = re.sub(r'gh', 'g', s)
    s = re.sub(r'jh', 'j', s)
    s = re.sub(r'ch', 'c', s)
    s = re.sub(r'sh', 's', s)
    s = re.sub(r'w', 'v', s)
    s = re.sub(r'ck', 'k', s)
    s = re.sub(r'q', 'k', s)
    s = re.sub(r'x', 'ks', s)
    s = re.sub(r'z', 's', s)
    s = re.sub(r'\bb([aeiou])', r'v\1', s)
    s = re.sub(r'\bshree\b|\bshri\b', 'sri', s)
    s = re.sub(r'(\w)\1+', r'\1', s)
    return s.strip()

def run_test_inference():
    t_start = time.time()
    print("=" * 85, flush=True)
    print("STARTING TEST INFERENCE: PHASE 5/6 RANK-GATED MODEL", flush=True)
    print("=" * 85, flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load test predictions and candidate pool
    print("\n[Step 1/7] Loading candidate pool and top3 test features...", flush=True)
    preds = pl.read_parquet(CACHE_DIR / "runs/v3/roles/pred_v2test.parquet").sort("q_idx")
    n_test_pairs = preds.height
    print(f"  Test candidate pairs: {n_test_pairs:,}", flush=True)
    assert n_test_pairs == 9969589, f"Expected 9,969,589 rows, got {n_test_pairs}"

    top3 = pl.read_parquet(CACHE_DIR / "runs/model_C3_direct_x_more_data/top3_v2test.parquet", columns=["q_idx", "p1", "p2", "p3"]).with_columns(
        (pl.col("p1") - pl.col("p2")).alias("margin12")
    )

    eval_feats = pl.read_parquet(CACHE_DIR / "runs/v3/eval_features_v2test.parquet")
    print(f"  Loaded eval features: {eval_feats.height:,} rows", flush=True)

    # 2. Load S1 metadata
    print("\n[Step 2/7] Loading and canonicalizing S1 metadata...", flush=True)
    s1_norm = pl.read_parquet(CACHE_DIR / "v2test_source1_norm.parquet", columns=["name_core", "addr_norm", "addr_nums", "addr_missing"]).with_row_index("s1_idx")
    s1_meta = s1_norm.select([
        "s1_idx",
        pl.len().over("name_core").alias("s1_name_freq"),
        pl.len().over("addr_norm").alias("s1_addr_freq"),
        pl.col("addr_nums").alias("s1_addr_nums"),
        pl.col("addr_norm").alias("s1_addr_norm"),
        pl.col("addr_missing").alias("s1_addr_missing"),
        pl.col("name_core").alias("s1_name_core"),
    ])
    del s1_norm
    gc.collect()

    unique_s1_names = s1_meta["s1_name_core"].unique().to_list()
    s1_indic_map = {name: indic_clean(name) for name in unique_s1_names if name is not None}
    s1_meta = s1_meta.with_columns(
        pl.col("s1_name_core").replace(s1_indic_map).alias("s1_indic")
    ).drop("s1_name_core")
    print(f"  S1 metadata processed: {s1_meta.height:,} entities", flush=True)

    # 3. Load Query metadata
    print("\n[Step 3/7] Loading and canonicalizing Query metadata...", flush=True)
    q2_norm = pl.read_parquet(CACHE_DIR / "v2test_source2_norm.parquet", columns=["name_core", "addr_norm", "addr_nums", "addr_missing"])
    q3_norm = pl.read_parquet(CACHE_DIR / "v2test_source3_norm.parquet", columns=["name_core", "addr_norm", "addr_nums", "addr_missing"])
    q_norm = pl.concat([q2_norm, q3_norm]).with_row_index("q_idx")
    del q2_norm, q3_norm
    gc.collect()

    unique_q_names = q_norm["name_core"].unique().to_list()
    q_indic_map = {name: indic_clean(name) for name in unique_q_names if name is not None}
    q_norm = q_norm.with_columns(
        pl.col("name_core").replace(q_indic_map).alias("q_indic")
    ).drop("name_core")
    print(f"  Query metadata processed: {q_norm.height:,} queries", flush=True)

    # 4. Join and engineer full 48-feature suite
    print("\n[Step 4/7] Joining feature tables and computing cluster stopping dynamics...", flush=True)
    pool = preds.join(top3, on="q_idx", how="inner")
    pool = pool.join(eval_feats, on="q_idx", how="inner")
    del preds, top3, eval_feats
    gc.collect()

    pool = pool.join(q_norm, on="q_idx", how="inner")
    pool = pool.join(s1_meta, on="s1_idx", how="left")
    del q_norm, s1_meta
    gc.collect()

    pool = pool.with_columns([
        ((pl.col("addr_nums").is_not_null()) & (pl.col("addr_nums") == pl.col("s1_addr_nums")) & (pl.col("addr_nums") != "")).cast(pl.Float32).alias("nums_exact"),
        pl.col("addr_missing").fill_null(False).cast(pl.Float32).alias("q_addr_missing"),
        ((~pl.col("addr_missing").fill_null(True)) & (~pl.col("s1_addr_missing").fill_null(True))).cast(pl.Float32).alias("both_addr_present"),
        ((pl.col("nm_exact") == 1.0) & (pl.col("cos_addr") < 0.50) & (~pl.col("addr_missing").fill_null(True))).cast(pl.Float32).alias("branch_distractor_flag"),
        ((pl.col("cos_addr") > 0.80) & (pl.col("nm_exact") == 0.0) & (pl.col("key_exact") == 0.0)).cast(pl.Float32).alias("multi_tenant_flag"),
        (1.0 / (pl.col("s1_name_freq").cast(pl.Float32) + 1.0)).alias("name_uniqueness"),
        (1.0 / (pl.col("s1_addr_freq").cast(pl.Float32) + 1.0)).alias("addr_uniqueness"),
        ((pl.col("nm_exact") == 0.0) & (pl.col("q_indic").is_not_null()) & (pl.col("q_indic") == pl.col("s1_indic")) & (pl.col("q_indic") != "")).cast(pl.Float32).alias("indic_exact"),
        ((pl.col("num_conflict") == 1.0) & (pl.col("nm_exact") == 1.0) & (pl.col("ad_tset") >= 80.0)).cast(pl.Float32).alias("spurious_num_conflict"),
        ((pl.col("nm_exact") == 1.0) & (pl.col("ad_ratio") < 40.0) & (~pl.col("addr_missing").fill_null(True))).cast(pl.Float32).alias("strong_name_weak_addr"),
        (pl.col("p") * pl.col("margin12")).alias("p_mult_margin12"),
    ])

    pool = pool.sort(["s1_idx", "p"], descending=[False, True])
    pool = pool.with_columns([
        pl.int_range(1, pl.len() + 1).over("s1_idx").alias("rank_in_s1"),
        pl.len().over("s1_idx").alias("s1_n_cands"),
        pl.col("p").max().over("s1_idx").alias("s1_max_p"),
        pl.col("p").mean().over("s1_idx").alias("s1_mean_p"),
        (pl.col("p") >= 0.80).cast(pl.Int32).sum().over("s1_idx").alias("s1_n_p80"),
        (pl.col("p") >= 0.70).cast(pl.Int32).sum().over("s1_idx").alias("s1_n_p70"),
        (pl.col("p") >= 0.50).cast(pl.Int32).sum().over("s1_idx").alias("s1_n_p50"),
        pl.col("cos_addr").rank(descending=True).over("s1_idx").cast(pl.Float32).alias("addr_rank_in_s1"),
        (pl.col("nm_exact") == 1.0).cast(pl.Int32).sum().over("s1_idx").alias("n_cands_same_name"),
    ])

    pool = pool.with_columns([
        (pl.col("rank_in_s1") == 1).cast(pl.Float32).alias("rank_is_1"),
        (pl.col("p") - pl.col("s1_max_p")).alias("p_diff_max"),
        (pl.col("p") / (pl.col("s1_max_p") + 1e-6)).alias("p_ratio_max"),
        (pl.col("rank_in_s1") / pl.col("s1_n_cands")).alias("p_percentile_in_s1"),
        (pl.col("s1_max_p") >= 0.95).cast(pl.Float32).alias("s1_has_strong_anchor"),
        (pl.col("p") - pl.col("p").shift(-1).over("s1_idx")).fill_null(0.0).alias("p_gap_to_next"),
        (pl.col("p").shift(1).over("s1_idx") - pl.col("p")).fill_null(0.0).alias("p_gap_to_prev"),
        (pl.col("s1_n_p80") / (pl.col("s1_n_cands") + 1e-6)).alias("cluster_p80_ratio"),
        (pl.col("p") / (pl.col("p").shift(1).over("s1_idx") + 1e-6)).fill_null(1.0).alias("cluster_decay_rate"),
        ((pl.col("rank_in_s1") > 1) & (pl.col("s1_max_p") >= 0.90)).cast(pl.Float32).alias("follower_of_strong_anchor"),
        ((pl.col("cos_addr") >= 0.90) & (pl.col("nums_exact") == 1.0) & (pl.col("nm_exact") == 0.0) & (pl.col("addr_rank_in_s1") == 1.0)).cast(pl.Float32).alias("rebrand_candidate"),
    ])

    feature_cols = [
        "p", "p1", "p2", "p3", "margin12", "p_mult_margin12",
        "nm_exact", "key_exact", "raw_name_eq", "raw_addr_eq",
        "ad_ratio", "ad_tset", "cos_addr", "num_conflict", "prem_eq",
        "q_is_s3",
        "rank_in_s1", "rank_is_1", "s1_n_cands", "s1_max_p", "s1_mean_p",
        "s1_n_p80", "s1_n_p70", "s1_n_p50",
        "p_diff_max", "p_ratio_max", "p_percentile_in_s1",
        "s1_has_strong_anchor", "p_gap_to_next", "p_gap_to_prev",
        "cluster_p80_ratio", "cluster_decay_rate", "follower_of_strong_anchor",
        "s1_name_freq", "s1_addr_freq",
        "nums_exact", "q_addr_missing", "both_addr_present",
        "branch_distractor_flag", "multi_tenant_flag",
        "name_uniqueness", "addr_uniqueness",
        "addr_rank_in_s1", "n_cands_same_name",
        "indic_exact", "spurious_num_conflict", "strong_name_weak_addr", "rebrand_candidate"
    ]
    print(f"  Feature table ready with {len(feature_cols)} features", flush=True)

    # 5. Predict with Phase 5/6 Booster
    print("\n[Step 5/7] Scoring TEST candidate pairs with Phase 5/6 model...", flush=True)
    model_p = CACHE_DIR / "runs/model_phase56.txt"
    if not model_p.exists():
        model_p = _here.parent / "models/model_phase56.txt"
    model = lgb.Booster(model_file=str(model_p))
    X = pool.select(feature_cols).to_numpy()
    probs = model.predict(X, num_iteration=model.num_trees())
    del X
    gc.collect()

    pool = pool.with_columns(pl.Series("p_set", probs))
    print(f"  Prediction complete. Mean p_set = {pool['p_set'].mean():.4f}", flush=True)

    # 6. Apply Rank-Gated Decision Rule
    print("\n[Step 6/7] Applying Rank-Gated Decision Rule (R1 >= 0.70, R2+ >= 0.80)...", flush=True)
    cond = ((pl.col("rank_in_s1") == 1) & (pl.col("p_set") >= 0.70)) | ((pl.col("rank_in_s1") > 1) & (pl.col("p_set") >= 0.80))
    matches = pool.filter(cond).select(["s1_idx", "q_idx"]).sort("q_idx")
    n_matches = matches.height
    print(f"  Total matched pairs: {n_matches:,} (vs V3 t=0.80: 5,712,570, V3 t=0.75: 5,741,673)", flush=True)

    # Free pool memory
    del pool
    gc.collect()

    # 7. Write matching_results.tsv and package submission
    print("\n[Step 7/7] Generating submission artifacts...", flush=True)
    s1_ids = pl.read_parquet(CACHE_DIR / "test_source1.parquet", columns=["entity_id"])["entity_id"]
    q_ids = pl.concat([
        pl.read_parquet(CACHE_DIR / f"test_{s}.parquet", columns=["entity_id"])
        for s in ("source2", "source3")
    ])["entity_id"]
    n_s1 = len(s1_ids)

    matching_tsv = OUTPUT_DIR / "matching_results.tsv"
    temp_tsv = matching_tsv.with_suffix(".tmp")
    n_rows, n_nonempty = write_grouped(s1_ids, q_ids, matches, "matched_entity_ids", temp_tsv)
    assert n_rows == n_s1, f"Row count mismatch: {n_rows} vs {n_s1}"
    print(f"  Wrote matching_results: {n_rows:,} S1 rows ({n_nonempty:,} with matches, {n_s1 - n_nonempty:,} singletons)", flush=True)

    # Copy / hardlink frozen candidate_pairs.tsv
    candidate_source = ROOT / "output/v3_roles/candidate_pairs.tsv"
    expected_cand_sha = "168dd4fde9c2b76a028c731590080877401fb91357e9d8d7a1cbf7f351f3fe32"
    assert sha256_file(candidate_source) == expected_cand_sha, "Source candidate_pairs.tsv hash mismatch!"

    candidate_tsv = OUTPUT_DIR / "candidate_pairs.tsv"
    if candidate_tsv.exists():
        candidate_tsv.unlink()
    try:
        os.link(candidate_source, candidate_tsv)
    except OSError:
        shutil.copy2(candidate_source, candidate_tsv)
    assert sha256_file(candidate_tsv) == expected_cand_sha, "Exported candidate_pairs.tsv hash mismatch!"
    print(f"  Verified candidate_pairs.tsv SHA-256: {expected_cand_sha}", flush=True)

    # Row-by-row subset validation
    print("  Streaming validation of subset constraints...", flush=True)
    with temp_tsv.open(encoding="utf-8") as a, candidate_tsv.open(encoding="utf-8") as b:
        assert next(a).rstrip("\n") == "source1_entity_id\tmatched_entity_ids"
        assert next(b).rstrip("\n") == "source1_entity_id\tcandidate_entity_ids"
        checked = 0
        for left, right in zip_longest(a, b):
            assert left is not None and right is not None, "Mismatched S1 row coverage"
            sid, mids = left.rstrip("\n").split("\t")
            cid, cids = right.rstrip("\n").split("\t")
            assert sid == cid, f"Mismatched S1 entity order at row {checked}: {sid} vs {cid}"
            requested = mids.split(",") if mids else []
            assert len(requested) == len(set(requested)), f"Duplicate matches for {sid}"
            c_set = set(cids.split(",")) if cids else set()
            assert set(requested).issubset(c_set), f"Matches not a subset of candidates for {sid}"
            checked += 1
        assert checked == n_s1
    os.replace(temp_tsv, matching_tsv)
    print(f"  Validated {checked:,} S1 rows: 100% strict subset compliance!", flush=True)

    # Package ZIP archive
    print(f"\nPackaging into {ARCHIVE_PATH.name}...", flush=True)
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    with zipfile.ZipFile(ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
            fpath = OUTPUT_DIR / fname
            zf.write(fpath, arcname=fname)
            print(f"  Added {fname} ({fpath.stat().st_size / 1e6:.1f} MB)", flush=True)

    archive_size_mb = ARCHIVE_PATH.stat().st_size / 1e6
    matching_sha = sha256_file(matching_tsv)
    archive_sha = sha256_file(ARCHIVE_PATH)

    print("\n" + "=" * 85)
    print("FINAL SUBMISSION PACKAGE READY")
    print("=" * 85)
    print(f"Archive:               {ARCHIVE_PATH}")
    print(f"Archive Size:          {archive_size_mb:.2f} MB")
    print(f"Archive SHA-256:       {archive_sha}")
    print(f"Matching Results SHA:  {matching_sha}")
    print(f"Candidate Pairs SHA:   {expected_cand_sha}")
    print(f"Total Execution Time:  {time.time() - t_start:.2f}s")
    print("=" * 85)

if __name__ == "__main__":
    run_test_inference()
