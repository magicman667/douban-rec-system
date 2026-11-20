from __future__ import annotations

import argparse
import json

import weaviate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--grpc-port", type=int, default=50051)
    parser.add_argument("--collection", default="DoubanMovie")
    parser.add_argument("--expected", type=int)
    args = parser.parse_args()
    client = weaviate.connect_to_custom(
        http_host=args.host,
        http_port=args.http_port,
        http_secure=False,
        grpc_host=args.host,
        grpc_port=args.grpc_port,
        grpc_secure=False,
    )
    try:
        collection = client.collections.get(args.collection)
        count = collection.aggregate.over_all(total_count=True).total_count
        response = collection.query.fetch_objects(limit=5, include_vector=True)
        dimensions: dict[str, set[int]] = {}
        samples = []
        for obj in response.objects:
            for name, vector in (obj.vector or {}).items():
                dimensions.setdefault(name, set()).add(len(vector))
            samples.append({"movie_id": obj.properties.get("movie_id"), "title": obj.properties.get("title"), "vectors": sorted((obj.vector or {}).keys())})
        report = {
            "ready": client.is_ready(),
            "collection": args.collection,
            "object_count": count,
            "vector_dimensions": {name: sorted(values) for name, values in dimensions.items()},
            "samples": samples,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.expected is not None and count != args.expected:
            raise SystemExit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
