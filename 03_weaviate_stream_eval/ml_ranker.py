"""Non-DL classical ML rankers (Factorization Machine, GBDT, GBDT+LR) for online TDC-Rec scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
from scipy import sparse

FEATURE_COLS = [
    "als_dot",
    "als_cos",
    "user_als_norm",
    "movie_als_norm",
    "bayesian_rating",
    "z_genre_score",
    "controversy_score",
    "log_rating_count",
    "log_douban_votes",
    "movie_year",
    "user_mean_rating",
    "user_pos_rate",
    "user_rating_std",
    "user_tolerance_tau",
    "user_log_history",
    "genre_affinity",
    "genre_max_affinity",
    "rating_minus_user_mean",
    "excess_risk",
    "risk_margin",
    "c_times_tau",
]

FEATURE_LABELS_ZH = {
    "als_dot": "ALS 协同内积",
    "als_cos": "ALS 协同余弦",
    "user_als_norm": "用户协同活跃模长",
    "movie_als_norm": "电影协同流行模长",
    "bayesian_rating": "电影贝叶斯口碑",
    "z_genre_score": "同题材相对分 Z",
    "controversy_score": "耦合分歧度 C",
    "log_rating_count": "建模评分样本量 (log)",
    "log_douban_votes": "豆瓣全站热度 (log)",
    "movie_year": "电影上映年份",
    "user_mean_rating": "用户历史平均打分",
    "user_pos_rate": "用户历史好评率",
    "user_rating_std": "用户打分严苛波动",
    "user_tolerance_tau": "用户分歧容忍度 τ",
    "user_log_history": "用户观影总数量 (log)",
    "genre_affinity": "主题材偏好度",
    "genre_max_affinity": "全题材最大契合度",
    "rating_minus_user_mean": "影片口碑 − 用户均分",
    "excess_risk": "超额分歧风险 max(0, C−τ)",
    "risk_margin": "分歧安全余量 (τ−C)",
    "c_times_tau": "分歧 × 容忍度交互",
}


class FactorizationMachine:
    """Rendle (2010) 2-way Factorization Machine trained via AdaGrad on binary logloss."""

    def __init__(
        self,
        n_factors: int = 16,
        lr: float = 0.05,
        l2_w: float = 1e-4,
        l2_v: float = 2e-4,
        epochs: int = 10,
        batch_size: int = 8192,
        seed: int = 42,
    ) -> None:
        self.n_factors = int(n_factors)
        self.lr = float(lr)
        self.l2_w = float(l2_w)
        self.l2_v = float(l2_v)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.w0: float = 0.0
        self.w: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def _raw_logits(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None and self.V is not None
        xv = X @ self.V
        x2v2 = (X * X) @ (self.V * self.V)
        inter = 0.5 * np.sum(xv * xv - x2v2, axis=1)
        return self.w0 + (X @ self.w) + inter

    def fit(self, X: np.ndarray, y: np.ndarray, val_data: tuple[np.ndarray, np.ndarray] | None = None) -> "FactorizationMachine":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)

        self.w0 = float(np.log(np.clip(y.mean(), 1e-4, 1.0 - 1e-4) / (1.0 - np.clip(y.mean(), 1e-4, 1.0 - 1e-4))))
        self.w = np.zeros(d, dtype=np.float64)
        self.V = rng.normal(0.0, 0.02 / np.sqrt(max(1, self.n_factors)), size=(d, self.n_factors)).astype(np.float64)

        g2_w0 = 1e-8
        g2_w = np.full(d, 1e-8, dtype=np.float64)
        g2_v = np.full((d, self.n_factors), 1e-8, dtype=np.float64)
        self.loss_history_ = []

        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            ep_loss = 0.0
            batches = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = X[idx].astype(np.float64, copy=False)
                yb = y[idx].astype(np.float64, copy=False)
                bs = float(len(yb))

                xv = xb @ self.V
                x2 = xb * xb
                x2v2 = x2 @ (self.V * self.V)
                logits = self.w0 + (xb @ self.w) + 0.5 * np.sum(xv * xv - x2v2, axis=1)
                logits = np.clip(logits, -30.0, 30.0)
                probs = 1.0 / (1.0 + np.exp(-logits))

                eps = 1e-7
                ep_loss += -float(np.mean(yb * np.log(probs + eps) + (1.0 - yb) * np.log(1.0 - probs + eps)))
                batches += 1

                err = (probs - yb) / bs
                grad_w0 = float(np.sum(err))
                grad_w = (xb.T @ err) + self.l2_w * self.w
                grad_v = (xb.T @ (err[:, None] * xv)) - ((x2.T @ err)[:, None] * self.V) + self.l2_v * self.V

                g2_w0 += grad_w0 * grad_w0
                g2_w += grad_w * grad_w
                g2_v += grad_v * grad_v

                self.w0 -= self.lr * grad_w0 / np.sqrt(g2_w0)
                self.w -= self.lr * grad_w / np.sqrt(g2_w)
                self.V -= self.lr * grad_v / np.sqrt(g2_v)

            mean_loss = ep_loss / max(1, batches)
            self.loss_history_.append(mean_loss)
            msg = f"  [FM epoch {epoch + 1:02d}/{self.epochs}] train_logloss={mean_loss:.5f}"
            if val_data is not None:
                from sklearn.metrics import roc_auc_score
                vx, vy = val_data
                vp = self.predict_proba(vx)
                msg += f", val_auc={roc_auc_score(vy, vp):.5f}"
            print(msg)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X64 = np.asarray(X, dtype=np.float64)
        logits = np.clip(self._raw_logits(X64), -30.0, 30.0)
        return np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-6, 1.0 - 1e-6)


def encode_fm_matrix(
    X_dense_scaled: np.ndarray,
    main_genres: Sequence[str],
    user_types: Sequence[str],
    top_genres: Sequence[str],
    known_user_types: Sequence[str],
) -> np.ndarray:
    """Build dense + one-hot feature matrix for Factorization Machine."""
    n = len(main_genres)
    g_map = {g: i for i, g in enumerate(top_genres)}
    u_map = {u: i for i, u in enumerate(known_user_types)}
    oh_g = np.zeros((n, len(top_genres)), dtype=np.float32)
    oh_u = np.zeros((n, len(known_user_types)), dtype=np.float32)
    for row_idx, g in enumerate(main_genres):
        gi = g_map.get(str(g or ""))
        if gi is not None:
            oh_g[row_idx, gi] = 1.0
    for row_idx, u in enumerate(user_types):
        ui = u_map.get(str(u or ""))
        if ui is not None:
            oh_u[row_idx, ui] = 1.0
    return np.hstack([X_dense_scaled.astype(np.float32, copy=False), oh_g, oh_u])


@dataclass
class MLRankerEngine:
    """Thread-safe online batch ML scoring engine for GBDT+LR, GBDT, and FM."""

    bundle: dict
    movie_als_map: dict[int, tuple[np.ndarray, float]]
    user_als_path: Path
    user_stats_path: Path
    meta_votes_year: dict[int, tuple[float, float]]
    _user_cache: dict[int, tuple[np.ndarray | None, float, tuple[float, float, float]]] = field(default_factory=dict, repr=False)

    @classmethod
    def load(
        cls,
        bundle_path: Path,
        user_stats_path: Path,
        als_factors_path: Path,
        meta_extra_csv: Path,
    ) -> "MLRankerEngine":
        import csv
        import duckdb

        bundle = joblib.load(bundle_path)
        con = duckdb.connect()
        m_df = con.execute(
            f"SELECT id AS movie_id, features FROM read_parquet('{als_factors_path}/factor_type=movie/*.parquet')"
        ).fetchdf()
        movie_als_map: dict[int, tuple[np.ndarray, float]] = {}
        for mid, feat in zip(m_df["movie_id"], m_df["features"]):
            vec = np.asarray(feat, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            movie_als_map[int(mid)] = (vec, norm)

        meta_votes_year: dict[int, tuple[float, float]] = {}
        if meta_extra_csv.exists():
            with meta_extra_csv.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        mid = int(row["movie_id"])
                        yr = float(row.get("year") or 0)
                        votes = float(row.get("douban_votes") or 0)
                        meta_votes_year[mid] = (votes, yr if yr > 1880 else 2010.0)
                    except (TypeError, ValueError, KeyError):
                        continue

        return cls(
            bundle=bundle,
            movie_als_map=movie_als_map,
            user_als_path=als_factors_path / "factor_type=user",
            user_stats_path=user_stats_path,
            meta_votes_year=meta_votes_year,
        )

    def _lookup_user(self, user_id: int) -> tuple[np.ndarray | None, float, tuple[float, float, float]]:
        uid = int(user_id)
        cached = self._user_cache.get(uid)
        if cached is not None:
            return cached

        import duckdb

        con = duckdb.connect()
        row_als = con.execute(
            f"SELECT features FROM read_parquet('{self.user_als_path}/*.parquet') WHERE id = ? LIMIT 1",
            [uid],
        ).fetchone()
        u_vec: np.ndarray | None = None
        u_norm = 0.0
        if row_als and row_als[0] is not None:
            u_vec = np.asarray(row_als[0], dtype=np.float32)
            u_norm = float(np.linalg.norm(u_vec))

        row_st = con.execute(
            f"SELECT user_mean_rating, user_pos_rate, user_rating_std FROM read_parquet('{self.user_stats_path}') WHERE user_id = ? LIMIT 1",
            [uid],
        ).fetchone()
        if row_st:
            stats = (float(row_st[0]), float(row_st[1]), float(row_st[2]))
        else:
            stats = (3.42, 0.46, 0.95)
        result = (u_vec, u_norm, stats)
        if len(self._user_cache) >= 4096:
            self._user_cache.clear()
        self._user_cache[uid] = result
        return result

    def _build_matrix(
        self,
        user_id: int,
        profile: Mapping[str, object],
        movies: Sequence[Mapping[str, object]],
    ) -> tuple[np.ndarray, list[str], list[str], list[list[str]]]:
        u_vec, u_norm, (u_mean, u_pos, u_std) = self._lookup_user(int(user_id))
        tau = float(profile.get("tolerance_tau") or 0.39)
        u_cnt = int(profile.get("rating_count") or 1)
        u_log_hist = float(np.log1p(max(0, u_cnt)))
        user_type = str(profile.get("user_type_label") or "均衡观影派")
        prefs = profile.get("genre_preferences") or {}
        if not isinstance(prefs, Mapping):
            prefs = {}

        rows: list[list[float]] = []
        main_genres: list[str] = []
        user_types: list[str] = []
        reasons_list: list[list[str]] = []

        for m in movies:
            mid = int(m["movie_id"])
            m_entry = self.movie_als_map.get(mid)
            if u_vec is not None and m_entry is not None:
                m_vec, m_norm = m_entry
                als_dot = float(np.dot(u_vec, m_vec))
                als_cos = als_dot / (u_norm * m_norm + 1e-8)
            else:
                als_dot, als_cos, m_norm = 0.0, 0.0, (m_entry[1] if m_entry is not None else 0.0)

            bay = float(m.get("bayesian_rating") or 3.2)
            zg = float(m.get("z_genre_score") or 0.0)
            c = float(m.get("controversy_score") or 0.0)
            rc = int(m.get("rating_count") or 0)
            votes_meta, year_meta = self.meta_votes_year.get(mid, (0.0, 2010.0))
            votes = float(m.get("douban_votes") or votes_meta or 0.0)
            yr_raw = float(m.get("year") or 0.0)
            yr = yr_raw if yr_raw > 1880 else year_meta

            mg = str(m.get("main_genre") or "未知")
            gs = [str(g) for g in (m.get("genres") or ([mg] if mg else []))]
            g_aff = float(prefs.get(mg, 0.0))
            g_max = max([float(prefs.get(g, 0.0)) for g in gs], default=g_aff)

            excess = max(0.0, c - tau)
            margin = tau - c
            rows.append([
                als_dot,
                als_cos,
                u_norm,
                m_norm,
                bay,
                zg,
                c,
                float(np.log1p(max(0, rc))),
                float(np.log1p(max(0.0, votes))),
                yr,
                u_mean,
                u_pos,
                u_std,
                tau,
                u_log_hist,
                g_aff,
                g_max,
                bay - u_mean,
                excess,
                margin,
                c * tau,
            ])
            main_genres.append(mg)
            user_types.append(user_type)

            tags: list[str] = []
            if bay >= 3.8 or zg >= 0.4:
                tags.append("口碑领先")
            if als_dot >= 0.6 or als_cos >= 0.35:
                tags.append("协同高匹配")
            if g_max >= 0.15:
                tags.append("题材契合")
            if excess > 0.05:
                tags.append("高分歧预警")
            elif margin >= 0.05:
                tags.append("低分歧安全")
            if not tags:
                tags.append("综合均衡")
            reasons_list.append(tags[:3])

        return np.asarray(rows, dtype=np.float32), main_genres, user_types, reasons_list

    def _predict_probs(
        self,
        X_dense: np.ndarray,
        main_genres: Sequence[str],
        user_types: Sequence[str],
        model_name: str,
    ) -> np.ndarray:
        scaler = self.bundle["scaler"]
        X_scaled = scaler.transform(X_dense)

        key = str(model_name or "GBDT+LR").upper()
        if "FM" in key:
            X_fm = encode_fm_matrix(
                X_scaled,
                main_genres,
                user_types,
                self.bundle["top_genres"],
                self.bundle["user_types"],
            )
            return self.bundle["fm"].predict_proba(X_fm)
        if key == "GBDT":
            return np.clip(self.bundle["gbdt"].predict_proba(X_dense)[:, 1], 1e-6, 1.0 - 1e-6)

        # Default: Facebook ADKDD'14 GBDT + LR
        leaves = self.bundle["gbdt_lr_trees"].apply(X_dense)[:, :, 0]
        X_ohe = self.bundle["gbdt_lr_ohe"].transform(leaves)
        X_sp = sparse.hstack([X_ohe, sparse.csr_matrix(X_scaled)]).tocsr()
        return np.clip(self.bundle["gbdt_lr_lr"].predict_proba(X_sp)[:, 1], 1e-6, 1.0 - 1e-6)

    def score_candidates(
        self,
        user_id: int,
        profile: Mapping[str, object],
        movies: Sequence[Mapping[str, object]],
        model_name: str = "GBDT+LR",
        popularity_weight: float = 0.0,
    ) -> list[dict]:
        if not movies:
            return []
        X_dense, main_genres, user_types, reasons_list = self._build_matrix(int(user_id), profile, movies)
        probs = self._predict_probs(X_dense, main_genres, user_types, model_name=model_name)
        tau = float(profile.get("tolerance_tau") or 0.39)

        scored: list[dict] = []
        for idx, m in enumerate(movies):
            item = dict(m)
            prob = float(probs[idx])
            c = float(item.get("controversy_score") or 0.0)
            if tau > 0.50:
                excess = 0.0
                explore = 0.26 * c
            else:
                excess = max(0.0, c - tau)
                explore = 0.0
            rc = int(item.get("rating_count") or 0)
            pop_bonus = float(popularity_weight) * float(np.log1p(max(0, rc)) / 6.0)
            # Keep ML predicted match probability honest while applying asymmetric landmine safety
            rank_score = prob - 0.65 * (excess**1.5) + explore + pop_bonus

            item["ml_prob"] = round(prob, 4)
            item["ml_model"] = str(model_name)
            item["ml_reasons"] = reasons_list[idx]
            item["ml_rank_score"] = round(rank_score, 5)
            scored.append(item)

        return scored

    def score_single_movie(
        self,
        user_id: int,
        profile: Mapping[str, object],
        movie: Mapping[str, object],
        model_name: str = "GBDT+LR",
    ) -> dict | None:
        try:
            res = self.score_candidates(int(user_id), profile, [movie], model_name=model_name)
            return res[0] if res else None
        except Exception:
            logging.exception("ML single scoring failed for uid=%s", user_id)
            return None
