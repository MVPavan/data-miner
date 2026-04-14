"""Compare results between two pipeline runs (e.g., after prompt version change)."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def compare_runs(job_a_dir: str, job_b_dir: str) -> None:
    """Compare traces from two pipeline runs.

    Reads all ``traces/*.json`` files from each job directory and prints a
    side-by-side breakdown of accept / reject / human_review counts per class.

    Parameters
    ----------
    job_a_dir:
        Path to the first job output directory (e.g. ``output/auto_annotation_v3/job_abc``).
    job_b_dir:
        Path to the second job output directory.
    """
    dir_a = Path(job_a_dir) / "traces"
    dir_b = Path(job_b_dir) / "traces"

    # Collect stats per evaluation group: stats[run][class][action] = count
    stats: dict[str, dict] = {
        "a": defaultdict(lambda: defaultdict(int)),
        "b": defaultdict(lambda: defaultdict(int)),
    }

    for label, traces_dir in [("a", dir_a), ("b", dir_b)]:
        if not traces_dir.exists():
            print(f"Warning: {traces_dir} not found")
            continue
        for trace_file in traces_dir.glob("*.json"):
            try:
                trace = json.loads(trace_file.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"Warning: could not parse {trace_file}: {exc}")
                continue
            annotations = trace.get("annotations", [])
            for ann in annotations:
                action = ann.get("action", "unknown")
                cls = ann.get("class_name", "unknown")
                stats[label][cls][action] += 1
                stats[label]["_total"][action] += 1

    # Compute the full class list from both runs (excluding the synthetic _total key)
    all_classes = sorted(
        set(list(stats["a"].keys()) + list(stats["b"].keys())) - {"_total"}
    )

    print(f"\nRun A: {job_a_dir}")
    print(f"Run B: {job_b_dir}")
    print(
        f"\n{'Class':<20} {'A accept':>10} {'B accept':>10} "
        f"{'A reject':>10} {'B reject':>10}"
    )
    print("-" * 65)

    for cls in all_classes:
        a_acc = stats["a"][cls].get("accept", stats["a"][cls].get("ACCEPT", 0))
        b_acc = stats["b"][cls].get("accept", stats["b"][cls].get("ACCEPT", 0))
        a_rej = stats["a"][cls].get("reject", stats["a"][cls].get("REJECT", 0))
        b_rej = stats["b"][cls].get("reject", stats["b"][cls].get("REJECT", 0))
        print(f"{cls:<20} {a_acc:>10} {b_acc:>10} {a_rej:>10} {b_rej:>10}")

    # Totals row per action
    print("-" * 65)
    for action in ["accept", "reject", "human_review"]:
        for variant in [action, action.upper()]:
            a_total = stats["a"]["_total"].get(variant, 0)
            b_total = stats["b"]["_total"].get(variant, 0)
            if a_total or b_total:
                diff = b_total - a_total
                pct = f"({diff:+d})" if diff != 0 else "(same)"
                print(
                    f"{'Total ' + action:<20} {a_total:>10} {b_total:>10} {pct:>10}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare annotation results between two pipeline runs",
        prog="python -m data_miner.auto_annotation_v3.compare",
    )
    parser.add_argument(
        "--job-a",
        required=True,
        help="Path to the first job output directory",
    )
    parser.add_argument(
        "--job-b",
        required=True,
        help="Path to the second job output directory",
    )
    args = parser.parse_args()
    compare_runs(args.job_a, args.job_b)


if __name__ == "__main__":
    main()
