#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    traces = sorted(args.trace_dir.glob("*.json"))
    if not traces:
        raise SystemExit(f"No trace files found in {args.trace_dir}")

    scores = []
    for trace_path in traces:
        score = run_trace(trace_path, args.base_url, args.k)
        scores.append(score)
        print(f"{trace_path.name}: Recall@{args.k}={score:.3f}")

    average = sum(scores) / len(scores)
    print(f"Average Recall@{args.k}={average:.3f}")


def run_trace(trace_path: Path, base_url: str, k: int) -> float:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    expected = {name.casefold() for name in trace.get("expected", [])}
    if not expected:
        return 0.0

    history = []
    final_response = {}
    with httpx.Client(timeout=30) as client:
        for message in trace.get("messages", []):
            history.append(message)
            response = client.post(f"{base_url.rstrip('/')}/chat", json={"messages": history})
            response.raise_for_status()
            final_response = response.json()
            history.append({"role": "assistant", "content": final_response.get("reply", "")})

    recommended = [
        item.get("name", "").casefold()
        for item in final_response.get("recommendations", [])[:k]
        if item.get("name")
    ]
    hits = sum(1 for name in expected if name in recommended)
    return hits / len(expected)


if __name__ == "__main__":
    main()
