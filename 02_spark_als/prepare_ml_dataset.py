"""Prepare leakage-free chronological Stage-2 datasets (21 dense features) for ML rankers."""

import argparse
import os
from pathlib import Path

import duckdb
import numpy as np
from sklearn.metrics import roc_auc_score


TRAIN_PARQUET = Path(
    os.getenv("DOUBAN_TRAIN_PARQUET", "/srv/douban-data/outputs/full-v1/parquet/train_ratings")
)
TEST_PARQUET = Path(
    os.getenv("DOUBAN_TEST_PARQUET", "/srv/douban-data/outputs/full-v1/parquet/test_ratings")
)
MOVIES_PARQUET = Path(
    os.getenv("DOUBAN_MOVIES_PARQUET", "/srv/douban-data/outputs/full-v1/parquet/movies")
)
META_EXTRA_CSV = Path(
    os.getenv("DOUBAN_META_EXTRA", "/srv/douban-data/outputs/full-v1/movie_meta_extra.csv")
)
MEMBER2_ARTIFACTS = Path(
    os.getenv(
        "DOUBAN_MEMBER2_ARTIFACTS",
        "/srv/douban-data/outputs/member2/full-v1-20261119-205432/artifacts",
    )
)
OUT_DIR = Path(os.getenv("DOUBAN_ML_DIR", "/srv/douban-data/outputs/full-v1/ml_ranker"))

PRIOR = 20.0
HIGH_CONTROVERSY = 0.50

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


def _parquet_glob(path: Path, pattern: str = "*.parquet") -> str:
    if path.is_file():
        return str(path)
    return str(path / pattern)


def run_split(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    train_glob = _parquet_glob(TRAIN_PARQUET)
    movies_glob = _parquet_glob(MOVIES_PARQUET)
    train_s1 = out_dir / "train_s1.parquet"
    train_s2 = out_dir / "train_s2.parquet"

    print("[split] Splitting train_ratings chronologically per user (75% s1 / 25% s2)...")
    con.execute(f"""
        COPY (
            WITH ranked AS (
                SELECT rating_id, user_id, movie_id, rating, rating_time,
                       row_number() OVER (PARTITION BY user_id ORDER BY rating_time, rating_id) AS rn,
                       count(*) OVER (PARTITION BY user_id) AS cnt
                FROM read_parquet('{train_glob}')
            )
            SELECT rating_id, user_id, movie_id, rating, rating_time
            FROM ranked
            WHERE cnt = 1 OR rn <= greatest(1, (cnt * 3) // 4)
        ) TO '{train_s1}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute(f"""
        COPY (
            WITH ranked AS (
                SELECT rating_id, user_id, movie_id, rating, rating_time,
                       row_number() OVER (PARTITION BY user_id ORDER BY rating_time, rating_id) AS rn,
                       count(*) OVER (PARTITION BY user_id) AS cnt
                FROM read_parquet('{train_glob}')
            )
            SELECT rating_id, user_id, movie_id, rating, rating_time
            FROM ranked
            WHERE cnt >= 2 AND rn > greatest(1, (cnt * 3) // 4)
        ) TO '{train_s2}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    c_s1, u_s1 = con.execute(f"SELECT count(*), count(DISTINCT user_id) FROM read_parquet('{train_s1}')").fetchone()
    c_s2, u_s2 = con.execute(f"SELECT count(*), count(DISTINCT user_id) FROM read_parquet('{train_s2}')").fetchone()
    print(f"[split] train_s1={c_s1:,} rows ({u_s1:,} users), train_s2={c_s2:,} rows ({u_s2:,} users)")

    s1_movies = out_dir / "s1_movie_features.parquet"
    print("[split] Computing Stage-1 movie statistics (train_s1 only)...")
    con.execute(f"""
        COPY (
            WITH counts AS (
                SELECT movie_id,
                       count(*)::BIGINT AS rating_count,
                       sum(rating)::DOUBLE AS rating_sum,
                       sum(rating * rating)::DOUBLE AS rating_square_sum,
                       sum(CASE WHEN rating = 1 THEN 1 ELSE 0 END)::DOUBLE AS r1,
                       sum(CASE WHEN rating = 2 THEN 1 ELSE 0 END)::DOUBLE AS r2,
                       sum(CASE WHEN rating = 3 THEN 1 ELSE 0 END)::DOUBLE AS r3,
                       sum(CASE WHEN rating = 4 THEN 1 ELSE 0 END)::DOUBLE AS r4,
                       sum(CASE WHEN rating = 5 THEN 1 ELSE 0 END)::DOUBLE AS r5
                FROM read_parquet('{train_s1}')
                GROUP BY movie_id
            ),
            obs AS (
                SELECT m.movie_id, coalesce(m.main_genre, '未知') AS main_genre, m.genres,
                       coalesce(c.rating_count, 0) AS n,
                       coalesce(c.rating_sum, 0.0) AS rating_sum,
                       coalesce(c.rating_square_sum, 0.0) AS rating_square_sum,
                       coalesce(c.r1, 0.0) AS r1, coalesce(c.r2, 0.0) AS r2,
                       coalesce(c.r3, 0.0) AS r3, coalesce(c.r4, 0.0) AS r4,
                       coalesce(c.r5, 0.0) AS r5
                FROM read_parquet('{movies_glob}') m
                LEFT JOIN counts c USING(movie_id)
            ),
            g_agg AS (
                SELECT main_genre,
                       sum(n)::DOUBLE AS gn,
                       sum(rating_sum)::DOUBLE AS gsum,
                       sum(rating_square_sum)::DOUBLE AS gsq
                FROM obs
                GROUP BY main_genre
            ),
            global_stats AS (
                SELECT avg(rating)::DOUBLE AS gmean FROM read_parquet('{train_s1}')
            ),
            calc AS (
                SELECT o.*, g.gn,
                       CASE WHEN g.gn > 0 THEN g.gsum / g.gn ELSE global_stats.gmean END AS gmean,
                       sqrt(greatest(0.0,
                           CASE WHEN g.gn > 0 THEN g.gsq / g.gn ELSE global_stats.gmean * global_stats.gmean END
                           - pow(CASE WHEN g.gn > 0 THEN g.gsum / g.gn ELSE global_stats.gmean END, 2)
                       )) AS gstd,
                       CASE WHEN o.n > 0 THEN o.rating_sum / o.n ELSE global_stats.gmean END AS fmean,
                       CASE WHEN o.n > 0 THEN o.r1 / o.n ELSE 0.0 END AS p1,
                       CASE WHEN o.n > 0 THEN o.r2 / o.n ELSE 0.0 END AS p2,
                       CASE WHEN o.n > 0 THEN o.r3 / o.n ELSE 0.0 END AS p3,
                       CASE WHEN o.n > 0 THEN o.r4 / o.n ELSE 0.0 END AS p4,
                       CASE WHEN o.n > 0 THEN o.r5 / o.n ELSE 0.0 END AS p5
                FROM obs o
                JOIN g_agg g USING(main_genre)
                CROSS JOIN global_stats
            ),
            terms AS (
                SELECT *,
                       (rating_sum + {PRIOR} * gmean) / (n + {PRIOR}) AS bayesian_rating,
                       CASE WHEN n > 0 THEN sqrt(greatest(0.0, rating_square_sum / n - fmean * fmean)) / 2.0 ELSE 0.0 END AS std_norm,
                       4.0 * (p1 + p2) * (p4 + p5) AS pol,
                       -(
                           (CASE WHEN p1 > 0 THEN p1 * ln(p1) ELSE 0.0 END)
                         + (CASE WHEN p2 > 0 THEN p2 * ln(p2) ELSE 0.0 END)
                         + (CASE WHEN p3 > 0 THEN p3 * ln(p3) ELSE 0.0 END)
                         + (CASE WHEN p4 > 0 THEN p4 * ln(p4) ELSE 0.0 END)
                         + (CASE WHEN p5 > 0 THEN p5 * ln(p5) ELSE 0.0 END)
                       ) / ln(5.0) AS ent
                FROM calc
            )
            SELECT movie_id,
                   n AS rating_count,
                   bayesian_rating,
                   CASE WHEN gstd > 0 THEN (bayesian_rating - gmean) / gstd ELSE 0.0 END AS z_genre_score,
                   CASE WHEN n > 0 THEN (n / (n + {PRIOR})) * (0.40 * std_norm + 0.35 * pol + 0.25 * ent) ELSE 0.0 END AS controversy_score
            FROM terms
        ) TO '{s1_movies}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    s1_users = out_dir / "s1_user_stats.parquet"
    print("[split] Computing Stage-1 user profiles & smoothed stats (train_s1 only)...")
    con.execute(f"""
        COPY (
            WITH global_stats AS (
                SELECT avg(rating)::DOUBLE AS gm,
                       avg(CASE WHEN rating >= 4 THEN 1.0 ELSE 0.0 END)::DOUBLE AS gp
                FROM read_parquet('{train_s1}')
            ),
            watched AS (
                SELECT t.user_id, t.rating, m.genres, mf.controversy_score
                FROM read_parquet('{train_s1}') t
                JOIN read_parquet('{movies_glob}') m USING(movie_id)
                JOIN read_parquet('{s1_movies}') mf USING(movie_id)
            ),
            base AS (
                SELECT user_id,
                       count(*)::BIGINT AS user_rating_count,
                       avg(rating)::DOUBLE AS raw_mean,
                       sum(rating)::DOUBLE AS rating_sum,
                       sum(CASE WHEN rating >= 4 THEN 1.0 ELSE 0.0 END)::DOUBLE AS pos_sum,
                       coalesce(stddev_pop(rating), 0.95)::DOUBLE AS raw_std,
                       avg(CASE WHEN rating >= 4 THEN controversy_score END) AS pos_c,
                       avg(CASE WHEN controversy_score >= {HIGH_CONTROVERSY} THEN rating END) AS div_r
                FROM watched
                GROUP BY user_id
            ),
            tau_calc AS (
                SELECT b.user_id, b.user_rating_count,
                       (b.rating_sum + 5.0 * g.gm) / (b.user_rating_count + 5.0) AS user_mean_rating,
                       (b.pos_sum + 5.0 * g.gp) / (b.user_rating_count + 5.0) AS user_pos_rate,
                       (b.raw_std * b.user_rating_count + 2.0 * 0.95) / (b.user_rating_count + 2.0) AS user_rating_std,
                       CASE
                           WHEN b.pos_c IS NOT NULL THEN least(1.0, b.pos_c + greatest(0.0, coalesce((b.div_r - b.raw_mean) / 4.0, 0.0)))
                           ELSE 0.39
                       END AS user_tolerance_tau
                FROM base b
                CROSS JOIN global_stats g
            ),
            g_counts AS (
                SELECT user_id, genre, count(*)::DOUBLE AS cnt
                FROM (
                    SELECT user_id, unnest(genres) AS genre
                    FROM watched
                    WHERE rating >= 4
                )
                GROUP BY user_id, genre
            ),
            g_ratios AS (
                SELECT user_id,
                       to_json(map_from_entries(list(struct_pack(k := genre, v := cnt / sum_cnt)))) AS genre_pref_json
                FROM (
                    SELECT user_id, genre, cnt,
                           sum(cnt) OVER (PARTITION BY user_id) AS sum_cnt
                    FROM g_counts
                )
                GROUP BY user_id
            )
            SELECT t.user_id,
                   t.user_rating_count,
                   t.user_mean_rating,
                   t.user_pos_rate,
                   t.user_rating_std,
                   t.user_tolerance_tau,
                   CASE
                       WHEN t.user_tolerance_tau < 0.28 THEN '稳妥防雷派'
                       WHEN t.user_tolerance_tau > 0.50 THEN '猎奇探索派'
                       ELSE '均衡观影派'
                   END AS user_type_label,
                   coalesce(g.genre_pref_json, '{{}}') AS genre_pref_json
            FROM tau_calc t
            LEFT JOIN g_ratios g USING(user_id)
        ) TO '{s1_users}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    full_users = out_dir / "full_user_rating_stats.parquet"
    print("[split] Computing full-train user rating stats (for test & online inference, 0 test leakage)...")
    con.execute(f"""
        COPY (
            WITH global_stats AS (
                SELECT avg(rating)::DOUBLE AS gm,
                       avg(CASE WHEN rating >= 4 THEN 1.0 ELSE 0.0 END)::DOUBLE AS gp
                FROM read_parquet('{train_glob}')
            ),
            base AS (
                SELECT user_id,
                       count(*)::BIGINT AS cnt,
                       sum(rating)::DOUBLE AS rating_sum,
                       sum(CASE WHEN rating >= 4 THEN 1.0 ELSE 0.0 END)::DOUBLE AS pos_sum,
                       coalesce(stddev_pop(rating), 0.95)::DOUBLE AS raw_std
                FROM read_parquet('{train_glob}')
                GROUP BY user_id
            )
            SELECT b.user_id,
                   (b.rating_sum + 5.0 * g.gm) / (b.cnt + 5.0) AS user_mean_rating,
                   (b.pos_sum + 5.0 * g.gp) / (b.cnt + 5.0) AS user_pos_rate,
                   (b.raw_std * b.cnt + 2.0 * 0.95) / (b.cnt + 2.0) AS user_rating_std
            FROM base b
            CROSS JOIN global_stats g
        ) TO '{full_users}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print("[split] Done.")


def _build_split_features(
    con: duckdb.DuckDBPyConnection,
    ratings_glob: str,
    user_als_glob: str,
    movie_als_glob: str,
    movie_feat_glob: str,
    user_stats_sql: str,
    movies_meta_glob: str,
    meta_extra_csv: Path,
    dest_parquet: Path,
) -> None:
    con.execute(f"""
        COPY (
            WITH u_als AS (
                SELECT id AS user_id, features AS uf,
                       sqrt(list_dot_product(features, features)) AS u_norm
                FROM read_parquet('{user_als_glob}')
            ),
            m_als AS (
                SELECT id AS movie_id, features AS mf,
                       sqrt(list_dot_product(features, features)) AS m_norm
                FROM read_parquet('{movie_als_glob}')
            ),
            meta AS (
                SELECT CAST(movie_id AS INTEGER) AS movie_id,
                       coalesce(TRY_CAST(year AS DOUBLE), 0.0) AS meta_year,
                       coalesce(TRY_CAST(douban_votes AS DOUBLE), 0.0) AS douban_votes
                FROM read_csv('{meta_extra_csv}', header=true)
            ),
            u_prof AS (
                {user_stats_sql}
            ),
            raw_join AS (
                SELECT r.user_id,
                       r.movie_id,
                       CASE WHEN r.rating >= 4 THEN 1 ELSE 0 END AS y,
                       (u.uf IS NOT NULL AND m.mf IS NOT NULL) AS has_als,
                       CASE WHEN u.uf IS NOT NULL AND m.mf IS NOT NULL
                            THEN list_dot_product(u.uf, m.mf) ELSE 0.0 END AS als_dot,
                       coalesce(u.u_norm, 0.0) AS user_als_norm,
                       coalesce(m.m_norm, 0.0) AS movie_als_norm,
                       coalesce(mf.bayesian_rating, 3.2) AS bayesian_rating,
                       coalesce(mf.z_genre_score, 0.0) AS z_genre_score,
                       coalesce(mf.controversy_score, 0.0) AS controversy_score,
                       ln(1.0 + coalesce(mf.rating_count, 0)::DOUBLE) AS log_rating_count,
                       ln(1.0 + coalesce(mt.douban_votes, 0.0)) AS log_douban_votes,
                       CASE WHEN coalesce(mt.meta_year, mm.year, 0) > 1880
                            THEN coalesce(mt.meta_year, mm.year)::DOUBLE ELSE 2010.0 END AS movie_year,
                       coalesce(up.user_mean_rating, 3.42) AS user_mean_rating,
                       coalesce(up.user_pos_rate, 0.46) AS user_pos_rate,
                       coalesce(up.user_rating_std, 0.95) AS user_rating_std,
                       coalesce(up.user_tolerance_tau, 0.39) AS user_tolerance_tau,
                       ln(1.0 + coalesce(up.user_rating_count, 1)::DOUBLE) AS user_log_history,
                       coalesce(mm.main_genre, '未知') AS main_genre,
                       mm.genres AS genres,
                       coalesce(up.user_type_label, '均衡观影派') AS user_type_label,
                       try_cast(coalesce(up.genre_pref_json, '{{}}')::JSON AS MAP(VARCHAR, DOUBLE)) AS gmap
                FROM read_parquet('{ratings_glob}') r
                LEFT JOIN u_als u USING(user_id)
                LEFT JOIN m_als m USING(movie_id)
                LEFT JOIN read_parquet('{movie_feat_glob}') mf USING(movie_id)
                LEFT JOIN read_parquet('{movies_meta_glob}') mm USING(movie_id)
                LEFT JOIN meta mt USING(movie_id)
                LEFT JOIN u_prof up USING(user_id)
            )
            SELECT user_id,
                   movie_id,
                   y,
                   has_als,
                   main_genre,
                   user_type_label,
                   als_dot::FLOAT AS als_dot,
                   (als_dot / (user_als_norm * movie_als_norm + 1e-8))::FLOAT AS als_cos,
                   user_als_norm::FLOAT AS user_als_norm,
                   movie_als_norm::FLOAT AS movie_als_norm,
                   bayesian_rating::FLOAT AS bayesian_rating,
                   z_genre_score::FLOAT AS z_genre_score,
                   controversy_score::FLOAT AS controversy_score,
                   log_rating_count::FLOAT AS log_rating_count,
                   log_douban_votes::FLOAT AS log_douban_votes,
                   movie_year::FLOAT AS movie_year,
                   user_mean_rating::FLOAT AS user_mean_rating,
                   user_pos_rate::FLOAT AS user_pos_rate,
                   user_rating_std::FLOAT AS user_rating_std,
                   user_tolerance_tau::FLOAT AS user_tolerance_tau,
                   user_log_history::FLOAT AS user_log_history,
                   coalesce(gmap[main_genre], 0.0)::FLOAT AS genre_affinity,
                   coalesce(list_max(list_transform(genres, g -> coalesce(gmap[g], 0.0))), 0.0)::FLOAT AS genre_max_affinity,
                   (bayesian_rating - user_mean_rating)::FLOAT AS rating_minus_user_mean,
                   greatest(0.0, controversy_score - user_tolerance_tau)::FLOAT AS excess_risk,
                   (user_tolerance_tau - controversy_score)::FLOAT AS risk_margin,
                   (controversy_score * user_tolerance_tau)::FLOAT AS c_times_tau
            FROM raw_join
        ) TO '{dest_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


def run_build(out_dir: Path) -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    train_s2 = out_dir / "train_s2.parquet"
    s1_movies = out_dir / "s1_movie_features.parquet"
    s1_users = out_dir / "s1_user_stats.parquet"
    s1_als_dir = out_dir / "s1_als_factors.parquet"
    if not s1_als_dir.exists():
        raise FileNotFoundError(f"Missing Stage-1 ALS factors at {s1_als_dir}. Run Spark s1_als first.")

    movies_glob = _parquet_glob(MOVIES_PARQUET)
    test_glob = _parquet_glob(TEST_PARQUET)
    stage2_train = out_dir / "stage2_train.parquet"
    stage2_test = out_dir / "stage2_test.parquet"

    print("[build] Building 21-feature stage2_train.parquet (from out-of-time train_s2)...")
    _build_split_features(
        con=con,
        ratings_glob=str(train_s2),
        user_als_glob=str(s1_als_dir / "factor_type=user" / "*.parquet"),
        movie_als_glob=str(s1_als_dir / "factor_type=movie" / "*.parquet"),
        movie_feat_glob=str(s1_movies),
        user_stats_sql=f"SELECT * FROM read_parquet('{s1_users}')",
        movies_meta_glob=movies_glob,
        meta_extra_csv=META_EXTRA_CSV,
        dest_parquet=stage2_train,
    )

    print("[build] Building 21-feature stage2_test.parquet (from sealed test_ratings + full-train stats)...")
    full_user_stats = out_dir / "full_user_rating_stats.parquet"
    full_profiles = MEMBER2_ARTIFACTS / "user_profiles.parquet"
    full_movies = MEMBER2_ARTIFACTS / "movie_features.parquet"
    full_als = MEMBER2_ARTIFACTS / "als_factors.parquet"
    test_user_sql = f"""
        SELECT p.user_id,
               p.rating_count AS user_rating_count,
               s.user_mean_rating,
               s.user_pos_rate,
               s.user_rating_std,
               p.tolerance_tau AS user_tolerance_tau,
               p.user_type_label,
               p.genre_pref_json
        FROM read_parquet('{full_profiles}/*.parquet') p
        LEFT JOIN read_parquet('{full_user_stats}') s USING(user_id)
    """
    _build_split_features(
        con=con,
        ratings_glob=test_glob,
        user_als_glob=str(full_als / "factor_type=user" / "*.parquet"),
        movie_als_glob=str(full_als / "factor_type=movie" / "*.parquet"),
        movie_feat_glob=_parquet_glob(full_movies),
        user_stats_sql=test_user_sql,
        movies_meta_glob=movies_glob,
        meta_extra_csv=META_EXTRA_CSV,
        dest_parquet=stage2_test,
    )

    # Sanity check out-of-time AUC alignment (zero leakage verification)
    for label, path in [("stage2_train (out-of-time s2)", stage2_train), ("stage2_test (sealed test)", stage2_test)]:
        total_n, als_n = con.execute(
            f"SELECT count(*), sum(CASE WHEN has_als THEN 1 ELSE 0 END) FROM read_parquet('{path}')"
        ).fetchone()
        samp = con.execute(
            f"SELECT y, als_dot, bayesian_rating, user_pos_rate FROM read_parquet('{path}') WHERE has_als USING SAMPLE 100000 ROWS (reservoir, 42)"
        ).fetchdf()
        auc_als = roc_auc_score(samp["y"], samp["als_dot"])
        auc_bay = roc_auc_score(samp["y"], samp["bayesian_rating"])
        auc_usr = roc_auc_score(samp["y"], samp["user_pos_rate"])
        print(
            f"[build] {label}: rows={total_n:,}, has_als={int(als_n):,} | "
            f"AUC(als_dot)={auc_als:.4f}, AUC(bayesian)={auc_bay:.4f}, AUC(user_pos_rate)={auc_usr:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["split", "build"])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.action == "split":
        run_split(args.out_dir)
    else:
        run_build(args.out_dir)


if __name__ == "__main__":
    main()
