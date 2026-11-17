"""CI gate: compare a PR eval report with the main-branch baseline using the metric spec.

    python -m lab9.gate --pr reports/eval_workflow.json --baseline reports/baseline_v1.json

exit 0 = merge, exit 1 = blocked. Safety metrics: max 0, no averaging away.
Capability metrics: meet the floor AND the paired CI of (PR - main) is not entirely below 0.
Thresholds may only get stricter on the same eval-set version (ratchet).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from lab8.console import utf8_stdio
from lab9 import stats
from lab9.eval import SPEC, load_spec, per_item
from lab9.graders import GRADERS


def _ratchet_errors(spec: dict, base_spec: dict) -> list[str]:
    if spec.get("eval_set_version") != base_spec.get("eval_set_version"):
        return []

    errors = []
    current_metrics = {m["id"]: m for m in spec.get("metrics", [])}
    previous_metrics = {m["id"]: m for m in base_spec.get("metrics", [])}
    for metric_id, previous in previous_metrics.items():
        current = current_metrics.get(metric_id)
        if current is None:
            errors.append(f"metric {metric_id} removed")
            continue
        old_threshold = previous.get("threshold", {})
        new_threshold = current.get("threshold", {})
        for key in ("min",):
            if key in old_threshold and (key not in new_threshold or new_threshold[key] < old_threshold[key]):
                errors.append(f"metric {metric_id} loosened {key}")
        for key in ("max", "max_increase"):
            if key in old_threshold and (key not in new_threshold or new_threshold[key] > old_threshold[key]):
                errors.append(f"metric {metric_id} loosened {key}")
        if previous.get("blocking") and not current.get("blocking"):
            errors.append(f"metric {metric_id} is no longer blocking")
        if old_threshold.get("no_significant_drop") and not new_threshold.get("no_significant_drop"):
            errors.append(f"metric {metric_id} removed no_significant_drop")

    current_slices = {s["name"]: s for s in spec.get("slices", [])}
    previous_slices = {s["name"]: s for s in base_spec.get("slices", [])}
    for name, previous in previous_slices.items():
        current = current_slices.get(name)
        if current is None:
            errors.append(f"slice {name} removed")
            continue
        if current.get("metric") != previous.get("metric"):
            errors.append(f"slice {name} metric changed")
        if "min" in previous and ("min" not in current or current["min"] < previous["min"]):
            errors.append(f"slice {name} loosened min")
        if "max" in previous and ("max" not in current or current["max"] > previous["max"]):
            errors.append(f"slice {name} loosened max")
        if previous.get("blocking") and not current.get("blocking"):
            errors.append(f"slice {name} is no longer blocking")
        if previous.get("no_significant_drop") and not current.get("no_significant_drop"):
            errors.append(f"slice {name} removed no_significant_drop")
    return errors


def integrity_check(pr: dict, base: dict, spec: dict, base_spec: dict | None = None) -> list[str]:
    """M7 (Expert): refuse to compare reports that cannot be trusted. Return a list of problems
    (empty = trustworthy). A gate that passes a broken report is worse than no gate.
    Must catch at least:
      - any metric value in pr["metrics"] that is NaN or inf
      - len(pr["rows"]) != manifest items x len(manifest seeds)   (dropped / duplicated rows)
      - the (item_id, seed) set of PR differs from the baseline's (cherry-picked seeds/items)
      - pr["evaluator_failures"] non-empty, or a row whose value for a graded spec metric is
        None (a crashed grader silently scores as "missing" and drops out of the mean)
      - base_spec given and a threshold in spec is LOOSER than in base_spec for the same
        eval-set version (ratchet: max may not rise, min may not fall, a blocking metric may
        not become non-blocking, a metric may not disappear; the same for spec["slices"],
        including dropping no_significant_drop)
      - an item's slice tags in PR rows differ from the baseline's (relabelling an item out of a
        slice hides its regression from the slice gate)"""
    problems = []
    rows = pr.get("rows")
    metrics = pr.get("metrics")
    manifest = pr.get("manifest")
    if not isinstance(rows, list) or not isinstance(metrics, dict) or not isinstance(manifest, dict):
        return ["report is missing rows, metrics, or manifest"]

    for metric in spec.get("metrics", []):
        metric_id = metric["id"]
        if metric_id not in metrics:
            problems.append(f"aggregate metric {metric_id} is missing")
        else:
            value = metrics[metric_id]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"aggregate metric {metric_id} is not numeric")
            elif not math.isfinite(value):
                problems.append(f"metric {metric_id} is not finite")

    seeds = manifest.get("seeds")
    item_count = manifest.get("items")
    if not isinstance(seeds, list) or not isinstance(item_count, int):
        problems.append("manifest is missing item count or seeds")
    else:
        expected_count = item_count * len(seeds)
        if len(rows) != expected_count:
            problems.append(f"row count {len(rows)} != manifest {expected_count}")

    def row_keys(report: dict) -> list[tuple[Any, Any]]:
        return [(row.get("item_id"), row.get("seed")) for row in report.get("rows", [])
                if isinstance(row, dict)]

    pr_keys = row_keys(pr)
    base_keys = row_keys(base)
    if len(set(pr_keys)) != len(pr_keys):
        problems.append("duplicate (item_id, seed) rows")
    if set(pr_keys) != set(base_keys):
        problems.append("PR (item_id, seed) set differs from baseline")

    expected_metrics = [m["id"] for m in spec.get("metrics", []) if m.get("id") in GRADERS]
    for row in rows:
        if not isinstance(row, dict):
            problems.append("row is not an object")
            continue
        for metric_id in expected_metrics:
            if metric_id not in row or row[metric_id] is None:
                problems.append(f"{row.get('item_id')}@{row.get('seed')}: missing {metric_id}")
            elif not isinstance(row[metric_id], bool):
                problems.append(f"{row.get('item_id')}@{row.get('seed')}: {metric_id} is not boolean")
        for metric_id, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                problems.append(f"{row.get('item_id')}@{row.get('seed')}: {metric_id} is not finite")

    if pr.get("evaluator_failures"):
        problems.append(f"{len(pr['evaluator_failures'])} evaluator failure(s)")

    def slices_by_trial(report: dict) -> dict[tuple[Any, Any], tuple[str, ...]]:
        return {
            (row.get("item_id"), row.get("seed")): tuple(sorted(row.get("slices") or []))
            for row in report.get("rows", []) if isinstance(row, dict)
        }

    if slices_by_trial(pr) != slices_by_trial(base):
        problems.append("PR item slice tags differ from baseline")

    if base_spec is not None:
        problems.extend(_ratchet_errors(spec, base_spec))
    return problems


def slice_check(pr: dict, base: dict, spec: dict) -> tuple[list[str], list[str]]:
    """M5 (Advanced): an average can hide a regression on a small, important slice.
    For each entry of spec["slices"] ({"name", "metric", "min"?, "no_significant_drop"?, "blocking"}):
      - value = mean of per_item(pr, metric, slice_name=name)
      - fail if below "min", or if no_significant_drop and the paired CI over the slice's items
        lies entirely below 0
    Return (blocked ids like "slice:money_movement/pass_rate", report lines)."""
    blocked, lines = [], []
    bootstrap = spec.get("bootstrap", {})
    for rule in spec.get("slices", []):
        name, metric_id = rule["name"], rule["metric"]
        current = per_item(pr, metric_id, name)
        previous = per_item(base, metric_id, name)
        keys = sorted(set(current) & set(previous))
        value = sum(current.values()) / len(current) if current else None
        reasons = []
        if value is None:
            reasons.append("no PR rows")
        elif "min" in rule and value < rule["min"]:
            reasons.append(f"below min {rule['min']}")
        elif "max" in rule and value > rule["max"]:
            reasons.append(f"above max {rule['max']}")

        note = ""
        if rule.get("no_significant_drop"):
            if not keys:
                reasons.append("no paired items")
            else:
                delta, lo, hi = stats.paired_bootstrap(
                    [current[key] for key in keys],
                    [previous[key] for key in keys],
                    n_resamples=bootstrap.get("n_resamples", 1000),
                    rng=bootstrap.get("rng", 2027),
                    confidence=bootstrap.get("confidence", 0.95),
                )
                note = f"Δ {delta:+.3f} CI [{lo:+.3f}, {hi:+.3f}]"
                if hi < 0:
                    reasons.append("significant drop")

        identifier = f"slice:{name}/{metric_id}"
        verdict = "BLOCK" if reasons and rule.get("blocking") else ("warn" if reasons else "ok")
        if verdict == "BLOCK":
            blocked.append(identifier)
        lines.append(
            f"  {identifier:<34} {value!s:<8} {note:<28} {verdict}"
            + (f" ({'; '.join(reasons)})" if reasons else "")
        )
    return blocked, lines


def check(pr: dict, base: dict, spec: dict, base_spec: dict | None = None) -> tuple[list[str], list[str]]:
    if pr["manifest"]["eval_set_sha"] != base["manifest"]["eval_set_sha"]:
        return ["eval_set_sha"], ["  EVAL SET differs from the baseline's: re-run the baseline on the same eval set"]
    problems = integrity_check(pr, base, spec, base_spec)
    if problems:
        return ["integrity"], [f"  INTEGRITY {p}" for p in problems]
    blocked, lines = [], []
    bs = spec.get("bootstrap", {})
    for m in spec["metrics"]:
        mid, th, val = m["id"], m.get("threshold", {}), pr["metrics"].get(m["id"])
        verdict, note = "ok", ""
        if val is None:
            verdict = "missing"
        elif "max" in th and val > th["max"]:
            verdict = "fail"
        elif "min" in th and val < th["min"]:
            verdict = "fail"
        elif "max_increase" in th and base["metrics"].get(mid):
            if val > base["metrics"][mid] * (1 + th["max_increase"]):
                verdict = "fail"
        if th.get("no_significant_drop") and val is not None:
            a, b = per_item(pr, mid), per_item(base, mid)
            keys = sorted(set(a) & set(b))
            d, lo, hi = stats.paired_bootstrap([a[k] for k in keys], [b[k] for k in keys],
                                               n_resamples=bs.get("n_resamples", 1000), rng=bs.get("rng", 2027))
            note = f"Δ {d:+.3f} CI [{lo:+.3f}, {hi:+.3f}]"
            if hi < 0:
                verdict = "fail"
        if verdict != "ok" and m.get("blocking"):
            blocked.append(mid)
        label = {"ok": "ok", "missing": "MISSING"}.get(verdict, "BLOCK" if m.get("blocking") else "warn")
        lines.append(f"  {mid:<22} {val!s:<8} {json.dumps(th):<40} {note:<28} {label}")
    if spec.get("slices"):
        sb, sl = slice_check(pr, base, spec)
        blocked += sb
        lines += sl
    return blocked, lines


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    ap = argparse.ArgumentParser(prog="lab9.gate")
    ap.add_argument("--pr", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--spec", default=str(SPEC))
    ap.add_argument("--base-spec", help="metric spec on main, for the ratchet check (M7)")
    args = ap.parse_args(argv)
    pr = json.loads(Path(args.pr).read_text(encoding="utf-8"))
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    base_spec = load_spec(Path(args.base_spec)) if args.base_spec else None
    blocked, lines = check(pr, base, load_spec(Path(args.spec)), base_spec)
    print(f"gate: PR {pr['run_id']} vs main {base['run_id']} · eval set {pr['manifest']['eval_set_sha']}")
    print("\n".join(lines))
    if blocked:
        print(f"exit 1 — blocked: {', '.join(blocked)}")
        return 1
    print("exit 0 — merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
