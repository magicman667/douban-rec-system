import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))

from live_backend import WeaviateRepository


class Collections:
    def get(self, _name):
        return object()


class Client:
    collections = Collections()


def test_semantic_cache_is_thread_safe_during_eviction():
    repository = WeaviateRepository(Client(), "x", {}, {}, lambda text: [float(len(text))])
    repository._near = lambda vector, target, limit: [(vector[0], float(limit))]
    keys = [f"query-{index}" for index in range(700)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda key: repository.semantic_candidates(key, 100), keys))
    assert len(results) == 700
    assert len(repository._semantic_cache) <= 512
