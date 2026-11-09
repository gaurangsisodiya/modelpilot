"""Write all xRouteBench test queries, shuffled (seed 0), as GuideLLM prompts: cache/live_prompts.jsonl."""

import json
from pathlib import Path

from modelpilot.data import load_split
from modelpilot.embeddings import query_table

OUT = Path("cache/live_prompts.jsonl")


def main() -> None:
    queries = query_table(load_split("test")).sample(frac=1.0, random_state=0)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as file:
        for text in queries["query"]:
            file.write(json.dumps({"prompt": text}) + "\n")
    print(f"Wrote {len(queries)} prompts to {OUT}")


if __name__ == "__main__":
    main()
