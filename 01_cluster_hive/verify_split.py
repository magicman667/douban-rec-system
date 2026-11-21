"""Validate identifiers, referential integrity and chronological splits."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def records(name):
    with (root / f"{name}.jsonl").open(encoding="utf-8") as handle:
        yield from map(json.loads, handle)

movies = {m["movie_id"] for m in records("movies")}
users = {}
seen = set()
counts = {}
for split in ("train", "test"):
    counts[split] = 0
    for rating in records(f"{split}_ratings"):
        user, rid = rating["user_id"], rating["rating_id"]
        if rid in seen or rating["movie_id"] not in movies or not 1 <= rating["rating"] <= 5:
            raise ValueError(f"Invalid rating ID: {rid}")
        seen.add(rid)
        key = rating["rating_time"], rid
        if split == "train":
            n, m, latest = users.get(user, (0, 0, key))
            users[user] = n + 1, m, max(latest, key)
        else:
            n, m, latest = users[user]
            if key <= latest:
                raise ValueError(f"Time split violated for user ID {user}")
            users[user] = n, m + 1, latest
        counts[split] += 1
for user, (train, test, _) in users.items():
    if train != max(1, (train + test) * 4 // 5):
        raise ValueError(f"Wrong train/test proportion for user ID {user}")
print(json.dumps(dict(movies=len(movies), users=len(users), **counts)))
