"""Extra detail on the live runs: error messages, empty replies, live vs offline token cost.

Reads results/v5_live/raw (GuideLLM JSONs + gateway log) and writes results/v5_live/live_details.md.
Does not touch live_results.md.
"""

import json
from pathlib import Path

import pandas as pd

from modelpilot.cost import add_cost
from modelpilot.data import load_candidates, load_split
from modelpilot.gateway_config import load_config

RAW = Path("results/v5_live/raw")
OUT = RAW.parent / "live_details.md"
SETUPS = {"small": "all-small", "large": "all-large", "auto": "routed"}


def benchmarks(setup: str) -> list[dict]:
    """GuideLLM benchmarks (one per concurrency) for a setup."""
    return json.loads((RAW / f"guidellm_{setup}.json").read_text())["benchmarks"]


def prompt_text(request: dict) -> str:
    """The user prompt GuideLLM sent, from its recorded request body."""
    content = json.loads(request["request_args"])["body"]["messages"][0]["content"]
    return content[0]["text"] if isinstance(content, list) else content


def error_lines() -> list[str]:
    lines = ["## Errored requests", "", "| Setup | Conc. | HTTP | Error (GuideLLM) | Failed after (s) |", "|---|---:|---:|---|---:|"]
    for setup, label in SETUPS.items():
        for bench in benchmarks(setup):
            for e in bench["requests"]["errored"]:
                info = e["info"]
                message = info["error"].split("\\n")[0].split(" for url")[0].removeprefix('HTTPStatusError("')
                status = message.split("'")[1].split()[0] if "'" in message else "?"
                seconds = (info["timings"]["resolve_end"] or 0) - (info["timings"]["request_start"] or 0)
                lines.append(f"| {label} | {bench['config']['strategy']['streams']} | {status} | {message} | {seconds:.2f} |")
    return lines


def empty_lines() -> list[str]:
    lines = [
        "## Empty replies (successful requests whose answer content is empty)", "",
        "| Setup | Conc. | Successful | Empty | Empty share | Empty and at 512-token cap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for setup, label in SETUPS.items():
        for bench in benchmarks(setup):
            ok = bench["requests"]["successful"]
            empty = [r for r in ok if not (r["output"] or "").strip()]
            capped = sum((r["output_tokens"] or 0) >= 512 for r in empty)
            lines.append(f"| {label} | {bench['config']['strategy']['streams']} | {len(ok)} | {len(empty)} | {len(empty) / len(ok):.1%} | {capped} |")
    return lines


def cost_lines() -> list[str]:
    candidates = load_candidates()
    backends = load_config(candidates).backends
    test = add_cost(load_split("test"), candidates)
    log = pd.DataFrame(json.loads(line) for line in (RAW / "gateway.jsonl").read_text().splitlines())
    log = log[log["status"] == 200]

    lines = [
        "## Live token cost vs offline cost per 1k requests, per backend", "",
        "Both use xRouteBench prices; only the token counts differ. Live = gateway log (status 200, all setups,",
        "max_tokens 512). Offline = xRouteBench test split. \"Same queries\" = offline cost of exactly the prompts",
        "sent in that backend's single-backend run (all-small / all-large), matched by prompt text.", "",
        "| Backend | Model | Live requests | Live in / out tok (mean) | Live $/1k | Offline in / out tok (mean, full test) | Offline $/1k (full test) | Offline $/1k (same queries) | Live / offline (same queries) |",
        "|---|---|---:|---|---:|---|---:|---:|---:|",
    ]
    for name in ("small", "large"):
        model = backends[name].router_model
        live = log[log["backend"] == name]
        offline = test[test["model_name"] == model]
        per_query = offline.groupby("query")["cost"].mean()
        sent = [prompt_text(r) for b in benchmarks(name) for r in b["requests"]["successful"]]
        matched = per_query.reindex(sent).dropna()
        same = matched.mean() * 1000
        live_cost = live["cost_usd"].mean() * 1000
        lines.append(
            f"| {name} | {model} | {len(live)} | {live['prompt_tokens'].mean():.0f} / {live['completion_tokens'].mean():.0f} | "
            f"{live_cost:.3f} | {offline['input_tokens'].mean():.0f} / {offline['output_tokens'].mean():.0f} | "
            f"{offline['cost'].mean() * 1000:.3f} | {same:.3f} ({len(matched)}/{len(sent)} matched) | {live_cost / same:.2f}x |"
        )
    return lines


def main() -> None:
    notes = [
        "",
        "Notes on errors: the gateway passes a backend's non-200 status straight through, and makes its own 502 only",
        "when it can't reach the backend. Neither GuideLLM nor the gateway log saved the response body, so the 502",
        "could be the gateway's 'backend unreachable' or a 502 from RunPod's proxy; the log can't tell them apart.",
    ]
    lines = ["# Live run details (results/v5_live)", "", *error_lines(), *notes, "", *empty_lines(), "", *cost_lines()]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
