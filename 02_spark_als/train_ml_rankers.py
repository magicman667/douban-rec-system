"""Train and benchmark non-DL ML rankers (ALS_Dot, L2-LR, FM, Spark YARN GBT, GBDT, GBDT+LR)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODULE_M3 = Path(__file__).resolve().parents[1] / "03_weaviate_stream_eval"
if str(MODULE_M3) not in sys.path:
    sys.path.insert(0, str(MODULE_M3))

from ml_ranker import FEATURE_COLS, FEATURE_LABELS_ZH, FactorizationMachine, encode_fm_matrix


ML_DIR = Path("/srv/douban-data/outputs/full-v1/ml_ranker")
ANALYSIS_SPARK = Path(__file__).resolve().parents[1] / "analysis" / "02_spark"

MODEL_ORDER = [
    ("ALS_Dot", "纯 ALS 协同得分 (课堂基础)", "Spark MLlib ALS"),
    ("L2-LR", "L2 逻辑回归 (全特征线性)", "scikit-learn LR"),
    ("FM", "FM 分解机 (k=16 二阶交叉)", "NumPy AdaGrad FM"),
    ("Spark_YARN_GBT", "Spark 分布式 GBT (2 Worker)", "Spark YARN MLlib"),
    ("GBDT", "梯度提升树 (Hist-GBDT)", "scikit-learn HistGBDT"),
    ("GBDT+LR", "GBDT + LR (Facebook 架构)", "Trees + Sparse LR"),
]


def compute_gauc(y_true: np.ndarray, y_score: np.ndarray, user_ids: np.ndarray) -> tuple[float, int]:
    """Compute impression-weighted grouped per-user ROC-AUC (GAUC)."""
    order = np.argsort(user_ids, kind="mergesort")
    u = user_ids[order]
    y = y_true[order]
    s = y_score[order]
    splits = np.flatnonzero(np.diff(u)) + 1
    starts = np.concatenate(([0], splits))
    ends = np.concatenate((splits, [len(u)]))

    weighted_sum = 0.0
    weight_total = 0
    valid_users = 0
    for st, en in zip(starts, ends):
        n = en - st
        if n < 2:
            continue
        yi = y[st:en]
        n_pos = int(yi.sum())
        if n_pos == 0 or n_pos == n:
            continue
        si = s[st:en]
        ranks = rankdata(si, method="average")
        auc = (float(ranks[yi == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / float(n_pos * (n - n_pos))
        weighted_sum += n * auc
        weight_total += n
        valid_users += 1
    return (weighted_sum / weight_total if weight_total > 0 else 0.5), valid_users


def evaluate_predictions(
    model_key: str,
    display_name: str,
    engine: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    user_ids: np.ndarray,
    train_sec: float,
    yarn_app_id: str = "-",
    raw_score: np.ndarray | None = None,
) -> tuple[dict, list[dict]]:
    rank_score = raw_score if raw_score is not None else probs
    roc = float(roc_auc_score(y_true, rank_score))
    gauc, g_users = compute_gauc(y_true, rank_score, user_ids)
    pr = float(average_precision_score(y_true, rank_score))
    p_clip = np.clip(probs, 1e-6, 1.0 - 1e-6)
    ll = float(log_loss(y_true, p_clip))
    acc = float(np.mean((p_clip >= 0.5) == (y_true == 1)))

    fpr, tpr, _ = roc_curve(y_true, rank_score)
    idx = np.unique(np.round(np.linspace(0, len(fpr) - 1, 65)).astype(int))
    roc_rows = [
        {"model": model_key, "display_name": display_name, "fpr": round(float(fpr[i]), 5), "tpr": round(float(tpr[i]), 5)}
        for i in idx
    ]

    summary = {
        "model": model_key,
        "display_name": display_name,
        "engine": engine,
        "roc_auc": round(roc, 6),
        "gauc": round(gauc, 6),
        "pr_auc": round(pr, 6),
        "logloss": round(ll, 6),
        "accuracy": round(acc, 6),
        "train_sec": round(float(train_sec), 2),
        "yarn_app_id": yarn_app_id,
        "gauc_users": g_users,
    }
    print(
        f"[eval] {model_key:<14} | ROC-AUC={roc:.5f} | GAUC={gauc:.5f} ({g_users:,}u) | "
        f"PR-AUC={pr:.5f} | LogLoss={ll:.5f} | Acc={acc:.4f} | {train_sec:.1f}s"
    )
    return summary, roc_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-dir", type=Path, default=ML_DIR)
    parser.add_argument("--out-csv-dir", type=Path, default=ANALYSIS_SPARK)
    args = parser.parse_args()

    con = duckdb.connect()
    cols_sql = ", ".join(FEATURE_COLS)
    print("[train] Loading stage2_train & stage2_test parquets...")
    df_tr = con.execute(
        f"SELECT user_id, movie_id, y, main_genre, user_type_label, {cols_sql} FROM read_parquet('{args.ml_dir / 'stage2_train.parquet'}')"
    ).fetchdf()
    df_te = con.execute(
        f"SELECT user_id, movie_id, y, main_genre, user_type_label, {cols_sql} FROM read_parquet('{args.ml_dir / 'stage2_test.parquet'}')"
    ).fetchdf()

    X_tr = df_tr[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_tr = df_tr["y"].to_numpy(dtype=np.int32)
    X_te = df_te[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_te = df_te["y"].to_numpy(dtype=np.int32)
    u_te = df_te["user_id"].to_numpy(dtype=np.int32)
    print(f"[train] stage2_train={X_tr.shape}, pos_rate={y_tr.mean():.4f} | stage2_test={X_te.shape}, pos_rate={y_te.mean():.4f}")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    all_metrics: list[dict] = []
    all_rocs: list[dict] = []
    meta_map = {k: (d, e) for k, d, e in MODEL_ORDER}

    # 1. Baseline: Pure ALS_Dot (Platt calibrated for logloss/acc, exact dot score for ranking)
    t0 = time.time()
    als_platt = LogisticRegression(max_iter=200)
    als_platt.fit(X_tr[:, [0]], y_tr)
    als_probs = als_platt.predict_proba(X_te[:, [0]])[:, 1]
    m, r = evaluate_predictions("ALS_Dot", *meta_map["ALS_Dot"], y_te, als_probs, u_te, time.time() - t0, raw_score=X_te[:, 0])
    all_metrics.append(m)
    all_rocs.extend(r)

    # 2. L2 Logistic Regression (21 dense features)
    t0 = time.time()
    lr_full = LogisticRegression(C=0.5, max_iter=400, solver="lbfgs")
    lr_full.fit(X_tr_s, y_tr)
    lr_probs = lr_full.predict_proba(X_te_s)[:, 1]
    m, r = evaluate_predictions("L2-LR", *meta_map["L2-LR"], y_te, lr_probs, u_te, time.time() - t0)
    all_metrics.append(m)
    all_rocs.extend(r)

    # 3. Factorization Machine (k=16, 21 dense + 15 top genres + 3 user archetypes)
    top_genres = df_tr["main_genre"].value_counts().head(15).index.astype(str).tolist()
    user_types = ["稳妥防雷派", "均衡观影派", "猎奇探索派"]
    X_fm_tr = encode_fm_matrix(X_tr_s, df_tr["main_genre"].astype(str).tolist(), df_tr["user_type_label"].astype(str).tolist(), top_genres, user_types)
    X_fm_te = encode_fm_matrix(X_te_s, df_te["main_genre"].astype(str).tolist(), df_te["user_type_label"].astype(str).tolist(), top_genres, user_types)
    t0 = time.time()
    fm = FactorizationMachine(n_factors=16, lr=0.06, l2_w=1e-4, l2_v=2e-4, epochs=10, batch_size=8192, seed=42)
    fm.fit(X_fm_tr, y_tr, val_data=(X_fm_te[:80000], y_te[:80000]))
    fm_sec = time.time() - t0
    fm_probs = fm.predict_proba(X_fm_te)
    m, r = evaluate_predictions("FM", *meta_map["FM"], y_te, fm_probs, u_te, fm_sec)
    all_metrics.append(m)
    all_rocs.extend(r)

    # 4. Spark YARN Distributed GBTClassifier
    spark_json = args.ml_dir / "spark_gbt_metrics.json"
    spark_preds_dir = args.ml_dir / "spark_gbt_preds.parquet"
    if spark_json.exists() and spark_preds_dir.exists():
        sp_meta = json.loads(spark_json.read_text(encoding="utf-8"))
        sp_df = con.execute(
            f"SELECT user_id, y, prob FROM read_parquet('{spark_preds_dir}/*.parquet')"
        ).fetchdf()
        m, r = evaluate_predictions(
            "Spark_YARN_GBT",
            *meta_map["Spark_YARN_GBT"],
            sp_df["y"].to_numpy(dtype=np.int32),
            sp_df["prob"].to_numpy(dtype=np.float64),
            sp_df["user_id"].to_numpy(dtype=np.int32),
            float(sp_meta["train_sec"]),
            yarn_app_id=str(sp_meta["yarn_app_id"]),
        )
        all_metrics.append(m)
        all_rocs.extend(r)
    else:
        print("[warn] Spark YARN GBT outputs not found, skipping Spark_YARN_GBT row.")

    # 5. HistGradientBoostingClassifier (full stage2_train)
    t0 = time.time()
    gbdt = HistGradientBoostingClassifier(
        max_iter=240,
        max_depth=6,
        learning_rate=0.06,
        l2_regularization=2.0,
        min_samples_leaf=100,
        random_state=42,
    )
    gbdt.fit(X_tr, y_tr)
    gbdt_sec = time.time() - t0
    gbdt_probs = gbdt.predict_proba(X_te)[:, 1]
    m, r = evaluate_predictions("GBDT", *meta_map["GBDT"], y_te, gbdt_probs, u_te, gbdt_sec)
    all_metrics.append(m)
    all_rocs.extend(r)

    # 6. GBDT + LR (Facebook ADKDD'14 architecture: disjoint A/B split -> leaf OHE + scaled dense -> L2 LR)
    t0 = time.time()
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y_tr))
    half = len(perm) // 2
    idx_a = perm[:half][:280000]
    idx_b = perm[half:]

    gbdt_trees = GradientBoostingClassifier(
        n_estimators=60,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        max_features="sqrt",
        random_state=42,
    )
    gbdt_trees.fit(X_tr[idx_a], y_tr[idx_a])

    leaves_b = gbdt_trees.apply(X_tr[idx_b])[:, :, 0]
    ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    X_b_sp = sparse.hstack([ohe.fit_transform(leaves_b), sparse.csr_matrix(X_tr_s[idx_b])]).tocsr()
    gbdt_lr_model = LogisticRegression(C=0.15, max_iter=300, solver="lbfgs")
    gbdt_lr_model.fit(X_b_sp, y_tr[idx_b])
    glr_sec = time.time() - t0

    leaves_te = gbdt_trees.apply(X_te)[:, :, 0]
    X_te_sp = sparse.hstack([ohe.transform(leaves_te), sparse.csr_matrix(X_te_s)]).tocsr()
    glr_probs = gbdt_lr_model.predict_proba(X_te_sp)[:, 1]
    m, r = evaluate_predictions("GBDT+LR", *meta_map["GBDT+LR"], y_te, glr_probs, u_te, glr_sec)
    all_metrics.append(m)
    all_rocs.extend(r)

    # 7. Permutation feature importance on GBDT (100K sealed test sample)
    print("[train] Computing permutation feature importance on 100K sealed test rows...")
    sub_idx = rng.choice(len(y_te), size=min(100000, len(y_te)), replace=False)
    perm_res = permutation_importance(
        gbdt,
        X_te[sub_idx],
        y_te[sub_idx],
        scoring="roc_auc",
        n_repeats=5,
        random_state=42,
        n_jobs=4,
    )
    imp_rows = []
    for i, feat in enumerate(FEATURE_COLS):
        imp_rows.append({
            "feature": feat,
            "label_zh": FEATURE_LABELS_ZH.get(feat, feat),
            "importance": round(float(max(0.0, perm_res.importances_mean[i])), 6),
            "std": round(float(perm_res.importances_std[i]), 6),
        })
    imp_df = pd.DataFrame(imp_rows).sort_values("importance", ascending=False).reset_index(drop=True)
    print("[train] Top 10 GBDT features:\n", imp_df.head(10).to_string(index=False))

    # 8. Persist production joblib bundle & contract CSVs
    bundle = {
        "feature_cols": FEATURE_COLS,
        "top_genres": top_genres,
        "user_types": user_types,
        "scaler": scaler,
        "lr": lr_full,
        "fm": fm,
        "gbdt": gbdt,
        "gbdt_lr_trees": gbdt_trees,
        "gbdt_lr_ohe": ohe,
        "gbdt_lr_lr": gbdt_lr_model,
        "metrics_summary": all_metrics,
    }
    bundle_path = args.ml_dir / "ml_ranker_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    print(f"[train] Saved bundle -> {bundle_path} ({bundle_path.stat().st_size / 1e6:.2f} MB)")

    args.out_csv_dir.mkdir(parents=True, exist_ok=True)
    met_df = pd.DataFrame(all_metrics).drop(columns=["gauc_users"])
    met_df.to_csv(args.out_csv_dir / "ml_ranker_metrics.csv", index=False)
    imp_df.to_csv(args.out_csv_dir / "ml_feature_importance.csv", index=False)
    pd.DataFrame(all_rocs).to_csv(args.out_csv_dir / "ml_roc_curves.csv", index=False)
    print(f"[train] Exported 3 CSVs into {args.out_csv_dir}")

    # Quick serving latency sanity check
    loaded = joblib.load(bundle_path)
    sample_batch = X_te[:50]
    sample_s = loaded["scaler"].transform(sample_batch)
    t_lat = []
    for _ in range(40):
        ts = time.perf_counter()
        lf = loaded["gbdt_lr_trees"].apply(sample_batch)[:, :, 0]
        sp = sparse.hstack([loaded["gbdt_lr_ohe"].transform(lf), sparse.csr_matrix(sample_s)]).tocsr()
        _ = loaded["gbdt_lr_lr"].predict_proba(sp)[:, 1]
        t_lat.append((time.perf_counter() - ts) * 1000.0)
    print(f"[train] Live GBDT+LR batch inference (50 candidates): P50={np.median(t_lat):.2f}ms, P95={np.percentile(t_lat, 95):.2f}ms")


if __name__ == "__main__":
    main()
