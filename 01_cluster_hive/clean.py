"""Build a reproducible full-data handoff from the original MovieData CSVs."""
import argparse
import csv
import json
import re
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

def rows(path, encoding):
    with path.open(encoding=encoding, errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)

def clean(value):
    return " ".join((value or "").split())

def integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def timestamp(value):
    try:
        return datetime.fromisoformat(value).isoformat(sep=" ", timespec="seconds")
    except (TypeError, ValueError):
        return None

def json_rows(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def csv_rows(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, help="New output directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists. Choose a new run directory.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10_000_000)
    stats = Counter()
    with tempfile.TemporaryDirectory(dir=args.output.parent) as work:
        stage = Path(work) / "publish"
        stage.mkdir()
        db = sqlite3.connect(Path(work) / "scratch.db")
        db.executescript("""
            CREATE TABLE movies(movie_id INTEGER PRIMARY KEY, title TEXT, payload TEXT);
            CREATE TABLE ratings(rating_id INTEGER PRIMARY KEY, user_md5 TEXT, movie_id INTEGER, rating INTEGER, rating_time TEXT,
              UNIQUE(user_md5,movie_id,rating_time,rating));
            CREATE TABLE comments(comment_id INTEGER PRIMARY KEY, movie_id INTEGER, rating INTEGER, votes INTEGER, content TEXT);
        """)
        for row in rows(args.source / "movies.csv", "utf-8-sig"):
            stats["movies_input"] += 1
            movie_id, title = integer(row["MOVIE_ID"]), clean(row["NAME"])
            if not movie_id or not title:
                continue
            movie = dict(movie_id=movie_id, title=title, cover_url=clean(row["COVER"]),
                         year=integer(row["YEAR"]), genres=list(dict.fromkeys(clean(x) for x in (row["GENRES"] or "").split("/") if clean(x))),
                         regions=list(dict.fromkeys(clean(x) for x in (row["REGIONS"] or "").split("/") if clean(x))),
                         tags=clean(row["TAGS"]), storyline=clean(row["STORYLINE"]))
            movie["main_genre"] = next(iter(movie["genres"]), "")
            db.execute("INSERT OR IGNORE INTO movies VALUES (?,?,?)", (movie_id,title,json.dumps(movie,ensure_ascii=False)))
        movie_ids = {row[0] for row in db.execute("SELECT movie_id FROM movies")}
        stats["movies_kept"] = len(movie_ids)
        for row in rows(args.source / "ratings.csv", "utf-8-sig"):
            stats["ratings_input"] += 1
            rid, mid, rating = map(integer,(row["RATING_ID"],row["MOVIE_ID"],row["RATING"]))
            user, when = clean(row["USER_MD5"]), timestamp(row["RATING_TIME"])
            if not rid or mid not in movie_ids or rating not in range(1,6) or not re.fullmatch(r"[0-9a-fA-F]{32}",user) or not when:
                stats["ratings_invalid"] += 1
                continue
            inserted = db.execute("INSERT OR IGNORE INTO ratings VALUES (?,?,?,?,?)",(rid,user.lower(),mid,rating,when))
            stats["ratings_duplicate"] += 1 - inserted.rowcount
        db.executescript("""
            CREATE TABLE users(user_id INTEGER PRIMARY KEY, user_md5 TEXT UNIQUE);
            INSERT INTO users(user_md5) SELECT DISTINCT user_md5 FROM ratings ORDER BY user_md5;
            CREATE INDEX ratings_order ON ratings(user_md5,rating_time,rating_id);
        """)
        train = (stage / "train_ratings.jsonl").open("w",encoding="utf-8")
        test = (stage / "test_ratings.jsonl").open("w",encoding="utf-8")
        fields = ("rating_id","user_id","movie_id","rating","rating_time")
        with train, test:
            for *item, ordinal, count in db.execute("""
                SELECT rating_id, user_id, movie_id, rating, rating_time,
                    row_number() OVER (PARTITION BY ratings.user_md5 ORDER BY rating_time,rating_id),
                    count(*) OVER (PARTITION BY ratings.user_md5)
                FROM ratings JOIN users USING(user_md5)
            """):
                is_train = ordinal <= max(1, count * 4 // 5)
                (train if is_train else test).write(json.dumps(dict(zip(fields,item)))+"\n")
                stats["train" if is_train else "test"] += 1
        for row in rows(args.source / "comments.csv", "gb18030"):
            stats["comments_input"] += 1
            cid, mid, rating, votes = map(integer,(row["COMMENT_ID"],row["MOVIE_ID"],row["RATING"],row["VOTES"]))
            body = clean(row["CONTENT"])
            if cid and mid in movie_ids and rating in (1,5) and body and "\ufffd" not in body:
                db.execute("INSERT OR IGNORE INTO comments VALUES (?,?,?,?,?)",(cid,mid,rating,max(0,votes or 0),body))
        db.commit()
        stats["users"] = db.execute("SELECT count(*) FROM users").fetchone()[0]
        stats["eligible_comments"] = db.execute("SELECT count(*) FROM comments").fetchone()[0]
        with (stage / "movies.jsonl").open("w",encoding="utf-8") as handle:
            for payload, in db.execute("SELECT payload FROM movies ORDER BY movie_id"):
                handle.write(payload+"\n")
        def reviews():
            active = None
            for mid, rating, content in db.execute("""
                SELECT movie_id,rating,content FROM (
                    SELECT *,row_number() OVER (PARTITION BY movie_id,rating ORDER BY votes DESC,comment_id) AS ordinal FROM comments
                ) WHERE ordinal<=5 ORDER BY movie_id,rating,ordinal
            """):
                if active is None or active["movie_id"] != mid:
                    if active: yield active
                    active = dict(movie_id=mid,pos_comments=[],neg_comments=[])
                active["pos_comments" if rating==5 else "neg_comments"].append(content)
            if active: yield active
        json_rows(stage / "movie_reviews.jsonl", reviews())
        csv_rows(stage / "user_mapping.csv",("user_id","user_md5"),db.execute("SELECT * FROM users"))
        counts = dict(db.execute("SELECT rating,count(*) FROM ratings GROUP BY rating"))
        total = sum(counts.values())
        if not total or not stats["test"]:
            raise ValueError("No usable full train/test data")
        csv_rows(stage / "rating_distribution.csv",("rating","count","ratio"),
                 ((rating,counts.get(rating,0),counts.get(rating,0)/total) for rating in range(1,6)))
        csv_rows(stage / "movie_rating_count.csv",("movie_id","title","rating_count"),db.execute("""
            SELECT movies.movie_id,title,coalesce(n,0) FROM movies
            LEFT JOIN (SELECT movie_id,count(*) n FROM ratings GROUP BY movie_id) ratings USING(movie_id)
            ORDER BY movies.movie_id
        """))
        with (stage / "wordcount_input.tsv").open("w",encoding="utf-8") as handle:
            for payload, in db.execute("SELECT payload FROM movies ORDER BY movie_id"):
                movie = json.loads(payload)
                for genre in movie["genres"]:
                    for tag in movie["tags"].split("/"):
                        if tag.strip(): handle.write(f"{genre}\t{tag.strip()}\n")
        (stage / "counts.json").write_text(json.dumps(stats,indent=2)+"\n",encoding="utf-8")
        db.close()
        stage.rename(args.output)

if __name__ == "__main__":
    main()
