"""Tune the strengthened fusion policy on the old pilot and test on fresh users."""

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import islice, product
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
import weaviate
from sentence_transformers import SentenceTransformer

from core import Movie, UserProfile, fuse_rankings
from fusion_policy import FUSION_BANDS, fusion_parameters, history_band, strengthened_weight


TOP_K = 10
HIGH_CONTROVERSY = 0.50
CONSERVATIVE_TOLERANCE = 0.28
DISCOUNTS = [1.0 / math.log2(rank + 1) for rank in range(1, TOP_K + 1)]
METRICS = ("ndcg_10", "recall_10", "landmine_10", "genre_err_10", "fill_rate_10", "mean_list_length")
METHODS = ("B2_als", "original", "strengthened", "tuned")
GRID = tuple(product((0.45, 0.475, 0.50, 0.525, 0.55, 0.60, 0.65, 0.70), (25, 60, 100)))
TUNING_KEYS = ("strengthened", *GRID)


def read_rows(path, columns, users=None, user_key="user_id", extra_filter=None):
    dataset = ds.dataset(path, format="parquet")
    condition = extra_filter
    if users is not None:
        selected = pc.field(user_key).isin(users)
        condition = selected if condition is None else condition & selected
    return dataset.to_table(columns=columns, filter=condition).to_pylist()


def collect_pairs(path, users, extra_filter=None):
    by_user = defaultdict(set)
    for row in read_rows(path, ["user_id", "movie_id"], users, extra_filter=extra_filter):
        by_user[row["user_id"]].add(row["movie_id"])
    return by_user


def load_catalog(path):
    columns = ["movie_id", "title", "main_genre", "genres", "rating_count",
               "bayesian_rating", "z_genre_score", "controversy_score"]
    return {row["movie_id"]: Movie.from_mapping(row) for row in read_rows(path, columns)}


def normalize(movie_ids, seen):
    return list(islice(dict.fromkeys(mid for mid in movie_ids if mid not in seen), TOP_K))


def ordering_metrics(movie_ids, relevant, conservative, movies):
    hits = [mid in relevant for mid in movie_ids]
    dcg = sum(discount for discount, hit in zip(DISCOUNTS, hits) if hit)
    ndcg = dcg / sum(DISCOUNTS[:len(relevant)])
    recall = sum(hits) / len(relevant)
    landmine = (
        sum(movies[mid].controversy_score >= HIGH_CONTROVERSY for mid in movie_ids) / TOP_K
        if conservative else math.nan
    )
    return ndcg, recall, landmine


def individual_metrics(movie_ids, relevant, profile, movies):
    ndcg, recall, landmine = ordering_metrics(
        movie_ids, relevant, profile.tolerance_tau < CONSERVATIVE_TOLERANCE, movies
    )
    preferences = profile.genre_preferences
    genre_error = math.nan
    if preferences:
        actual = Counter(genre for mid in movie_ids for genre in movies[mid].genres)
        total = actual.total()
        genre_error = (
            sum(abs(preferences.get(g, 0.0) - actual[g] / total) for g in preferences.keys() | actual.keys()) / 2
            if total else 1.0
        )
    return ndcg, recall, landmine, genre_error, float(len(movie_ids) == TOP_K), len(movie_ids)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_interval(values, rng, samples=2000):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None, None, None
    means = []
    for start in range(0, samples, 64):
        indices = rng.integers(0, len(values), size=(min(64, samples - start), len(values)), dtype=np.int32)
        means.extend(values[indices].mean(axis=1))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def select_settings(dev_results, output):
    settings = {}
    grid_rows = []
    for band in FUSION_BANDS:
        observations = np.array(dev_results[band])
        averages = np.nanmean(observations, axis=0)
        incumbent_ndcg, incumbent_recall, incumbent_landmine = averages[0]
        best_index = 0
        best_ndcg = incumbent_ndcg
        for index, (key, (ndcg, recall, landmine)) in enumerate(zip(TUNING_KEYS, averages, strict=True)):
            admissible = recall >= incumbent_recall and landmine <= incumbent_landmine
            if admissible and ndcg > best_ndcg + 1e-12:
                best_index = index
                best_ndcg = ndcg
            grid_rows.append({
                "rating_count_band": band, "users": len(observations),
                "policy": "strengthened" if key == "strengthened" else "fixed",
                "als_weight": "" if key == "strengthened" else key[0],
                "rrf_k": 60 if key == "strengthened" else key[1],
                "ndcg_10": ndcg, "recall_10": recall, "landmine_10": landmine,
                "admissible": bool(admissible),
            })
        winner = TUNING_KEYS[best_index]
        settings[band] = (
            {"policy": "strengthened"} if winner == "strengthened"
            else {"policy": "fixed", "als_weight": winner[0], "rrf_k": winner[1]}
        )
    write_csv(output / "dev_grid.csv", grid_rows)
    (output / "fusion_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=50000, help="0 uses all users outside both previous rounds")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--pilot-users", type=int, default=20000)
    parser.add_argument("--pilot-seed", type=int, default=42)
    parser.add_argument("--previous-users", type=int, default=40000)
    parser.add_argument("--previous-seed", type=int, default=43)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--http-port", type=int, default=18080)
    parser.add_argument("--grpc-port", type=int, default=50052)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    all_users = sorted(row["user_id"] for row in read_rows(args.artifacts / "eval_users", ["user_id"]))
    pilot = sorted(random.Random(args.pilot_seed).sample(all_users, args.pilot_users))
    pilot_set = set(pilot)
    outside_pilot = [uid for uid in all_users if uid not in pilot_set]
    previous = set(random.Random(args.previous_seed).sample(outside_pilot, args.previous_users))
    available = [uid for uid in outside_pilot if uid not in previous]
    sample = sorted(random.Random(args.seed).sample(available, args.users)) if args.users else available
    all_selected = sorted(pilot + sample)
    movies = load_catalog(args.artifacts / "movie_features.parquet")
    profiles = {row["user_id"]: UserProfile.from_mapping(row) for row in read_rows(
        args.artifacts / "user_profiles.parquet",
        ["user_id", "rating_count", "tolerance_tau", "user_type_label", "genre_pref_json"], all_selected,
    )}
    factors = {row["id"]: row["features"] for row in read_rows(
        args.artifacts / "als_factors.parquet" / "factor_type=user", ["id", "features"], all_selected, "id",
    )}
    seen = collect_pairs(args.snapshot / "train_ratings", all_selected)
    relevant = collect_pairs(args.snapshot / "test_ratings", all_selected, pc.field("rating") >= 4)
    b2 = defaultdict(list)
    for row in read_rows(args.artifacts / "als_top10.parquet", ["user_id", "rank", "movie_id"], sample):
        b2[row["user_id"]].append((row["rank"], row["movie_id"]))
    for uid in all_selected:
        relevant[uid].difference_update(seen[uid])
        if not relevant[uid] or uid not in profiles:
            raise ValueError(f"incomplete evaluation inputs for user {uid}")
    if any(len(b2[uid]) != TOP_K for uid in sample):
        raise ValueError("incomplete B2 rankings")
    print(f"loaded {len(pilot)} tuning and {len(sample)} fresh test users; {len(movies)} movies", flush=True)

    query_by_user = {
        uid: " ".join(sorted(profile.genre_preferences, key=profile.genre_preferences.get, reverse=True)[:5]) or "电影"
        for uid, profile in profiles.items()
    }
    queries = sorted(set(query_by_user.values()))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(args.model), device="cpu")
    vectors = model.encode(queries, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    print(f"encoded {len(queries)} original semantic queries in {time.monotonic() - started:.1f}s", flush=True)

    all_ratings = {method: [] for method in METHODS}
    coverage = {method: set() for method in METHODS}
    dev_results = defaultdict(list)

    with weaviate.connect_to_custom(
        http_host="127.0.0.1", http_port=args.http_port, http_secure=False,
        grpc_host="127.0.0.1", grpc_port=args.grpc_port, grpc_secure=False,
    ) as client, ThreadPoolExecutor(max_workers=args.workers) as pool:
        collection = client.collections.get("DoubanMovie")

        def near(vector, target):
            result = collection.query.near_vector(
                near_vector=vector, target_vector=target, limit=args.candidate_limit, return_properties=["movie_id"],
            )
            return [(movies[obj.properties["movie_id"]], 0.0) for obj in result.objects]

        def semantic_lookup(vector):
            return near(vector.tolist(), "semantic")

        semantic = dict(zip(queries, pool.map(semantic_lookup, vectors), strict=True))
        print(f"resolved semantic retrieval; tuning on {len(pilot)} users", flush=True)

        def rankings_for(uid):
            user_vector = factors.get(uid)
            als = near(user_vector, "collaborative") if user_vector is not None else []
            vector = semantic[query_by_user[uid]]
            profile = profiles[uid]
            excluded = seen[uid]
            cache = {}

            def rank(config):
                if config not in cache:
                    weight, rrf_k = config
                    ranked = fuse_rankings(als, vector, profile, als_weight=weight, rrf_k=rrf_k,
                                           excluded_movie_ids=excluded)
                    cache[config] = [item.movie.movie_id for item in ranked]
                return cache[config]

            return rank

        def tune_user(uid):
            profile = profiles[uid]
            rank = rankings_for(uid)
            conservative = profile.tolerance_tau < CONSERVATIVE_TOLERANCE
            rows = [ordering_metrics(rank(config), relevant[uid], conservative, movies)
                    for config in ((strengthened_weight(profile.rating_count), 60), *GRID)]
            return history_band(profile.rating_count), rows

        def batches(task, users, phase):
            for offset in range(0, len(users), 512):
                yield from pool.map(task, users[offset:offset + 512])
                done = min(offset + 512, len(users))
                if done % 2048 == 0 or done == len(users):
                    print(f"{phase} {done}/{len(users)} after {time.monotonic() - started:.1f}s", flush=True)

        for band, rows in batches(tune_user, pilot, "tuned"):
            dev_results[band].append(rows)
        settings = select_settings(dev_results, args.output)
        print(f"froze tuning settings: {settings}", flush=True)
        del dev_results

        def score_user(uid):
            ratings = profiles[uid].rating_count
            rank = rankings_for(uid)
            return uid, {
                "B2_als": normalize((mid for _, mid in sorted(b2[uid])), seen[uid]),
                "original": rank((0.5, 60)),
                "strengthened": rank((strengthened_weight(ratings), 60)),
                "tuned": rank(fusion_parameters(ratings, settings)),
            }

        for uid, lists in batches(score_user, sample, "scored blind"):
            for method, ids in lists.items():
                all_ratings[method].append(individual_metrics(ids, relevant[uid], profiles[uid], movies))
                coverage[method].update(ids)

    arrays = {method: np.array(rows) for method, rows in all_ratings.items()}
    rng = np.random.default_rng(args.seed)
    metrics = []
    by_history = []
    deltas = []
    labels = np.array([history_band(profiles[uid].rating_count) for uid in sample])
    for method, values in arrays.items():
        averages = dict(zip(METRICS, np.nanmean(values, axis=0).tolist(), strict=True))
        metrics.append({"method": method, **averages, "coverage_10": len(coverage[method]) / len(movies),
                        "distinct_movies": len(coverage[method])})
        for band in FUSION_BANDS:
            subset = values[labels == band]
            if len(subset):
                ndcg, recall = subset[:, :2].mean(axis=0)
                by_history.append({"method": method, "rating_count_band": band, "users": len(subset),
                                   "ndcg_10": ndcg, "recall_10": recall,
                                   "genre_err_10": np.nanmean(subset[:, 3])})
    for method, reference in (("strengthened", "original"), ("tuned", "strengthened"), ("tuned", "original")):
        differences = arrays[method] - arrays[reference]
        for band in ("all", *FUSION_BANDS):
            subset = differences if band == "all" else differences[labels == band]
            for index, metric in enumerate(METRICS[:4]):
                delta, low, high = paired_interval(subset[:, index], rng)
                deltas.append({"method": method, "reference": reference, "population": band, "metric": metric,
                               "delta": delta, "ci95_low": low, "ci95_high": high})
    write_csv(args.output / "metrics.csv", metrics)
    write_csv(args.output / "metrics_by_history.csv", by_history)
    write_csv(args.output / "paired_deltas.csv", deltas)
    metadata = {
        "evaluation_users": len(sample), "cohort_users": len(all_users), "seed": args.seed,
        "tuning_users": len(pilot), "pilot_seed": args.pilot_seed,
        "excluded_previous_users": len(previous), "previous_seed": args.previous_seed,
        "catalog_movies": len(movies), "candidate_limit_per_route": args.candidate_limit,
        "conservative_users": sum(profiles[uid].tolerance_tau < CONSERVATIVE_TOLERANCE for uid in sample),
        "long_history_users": int(sum(labels == "30+")),
        "fixed_risk_definition": "original controversy >= 0.50 and original tolerance < 0.28",
        "same_seen_filter": "remove seen movies before selecting Top 10 across all methods",
        "grid_candidates_per_band": len(GRID),
        "selection_rule": "max NDCG subject to recall and landmine rate at least as good as the strengthened baseline",
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "run_summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
