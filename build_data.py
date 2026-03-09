#!/usr/bin/env python3
"""Build consolidated data.json from multilingual evaluation results.

Reads results from results/{Language}/{model}/{checkpoint}/{task}/{partition}/
and aggregates across prompt/partition variants (p0, p1, p2).

Output: docs/data.json
"""

import json
import math
import os
import glob
import statistics
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = BASE_DIR / "docs" / "data.json"

SHOT_SETTINGS = ["0", "1", "5"]


# ── Per-language task configuration ──
# main_metric, random_baseline, max_performance, metric_scale, category

TASK_CONFIG = {
    # French tasks
    "fquad": {
        "pretty_name": "FQuAD",
        "main_metric": "exact",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "frabelebele": {
        "pretty_name": "Belebele (French)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "french_bench_grammar": {
        "pretty_name": "French Bench Grammar",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "french_bench_reading": {
        "pretty_name": "French Bench Reading",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "french_bench_vocabulary": {
        "pretty_name": "French Bench Vocabulary",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "french_xnli": {
        "pretty_name": "XNLI (French)",
        "main_metric": "acc",
        "random_baseline": 1 / 3,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "global_mmlu_french": {
        "pretty_name": "Global MMLU (French)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "include_french": {
        "pretty_name": "IncludeFrench",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "topic_based_nli": {
        "pretty_name": "Topic-Based NLI",
        "main_metric": "acc",
        "random_baseline": 1 / 3,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    # Spanish tasks
    "cocoteros_es": {
        "pretty_name": "Cocoteros (Spanish)",
        "main_metric": "bleu",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "generation & summarization",
    },
    "copa_es": {
        "pretty_name": "COPA (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "escola": {
        "pretty_name": "EsCoLA",
        "main_metric": "mcc",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "linguistic knowledge",
    },
    "flores_en-es": {
        "pretty_name": "FLORES (EN→ES)",
        "main_metric": "bleu",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "translation",
    },
    "global_mmlu_spanish": {
        "pretty_name": "Global MMLU (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "include_spanish": {
        "pretty_name": "IncludeSpanish",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "mgsm_direct_es": {
        "pretty_name": "MGSM Direct (Spanish)",
        "main_metric": "exact_match",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "math reasoning",
    },
    "openbookqa_es": {
        "pretty_name": "OpenBookQA (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "paws_es": {
        "pretty_name": "PAWS (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.5,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "spabelebele": {
        "pretty_name": "Belebele (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.25,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "veritasqa_es_gen": {
        "pretty_name": "TruthfulQA Gen (Spanish)",
        "main_metric": "bleu_max",
        "random_baseline": 0.0,
        "max_performance": 100.0,
        "metric_scale": "percent",
        "category": "world knowledge",
    },
    "veritasqa_es_mc1": {
        "pretty_name": "TruthfulQA MC1 (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.243,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "veritasqa_es_mc2": {
        "pretty_name": "TruthfulQA MC2 (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.154,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "world knowledge",
    },
    "xnli_es": {
        "pretty_name": "XNLI (Spanish)",
        "main_metric": "acc",
        "random_baseline": 1 / 3,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "xquad_es": {
        "pretty_name": "XQuAD (Spanish)",
        "main_metric": "exact_match",
        "random_baseline": 0.0,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
    "xstorycloze_es": {
        "pretty_name": "XStoryCloze (Spanish)",
        "main_metric": "acc",
        "random_baseline": 0.5,
        "max_performance": 1.0,
        "metric_scale": "unit",
        "category": "language understanding",
    },
}

# Model display names and colors
MODEL_CONFIG = {
    "ref_french_checkpoints": {
        "display_name": "HPLT French (reference)",
        "color": "#6366f1",
    },
    "synt_french_checkpoints": {
        "display_name": "HPLT French (synthetic)",
        "color": "#f43f5e",
    },
    "ref_spanish_checkpoints": {
        "display_name": "HPLT Spanish (reference)",
        "color": "#6366f1",
    },
    "synt_spanish_checkpoints": {
        "display_name": "HPLT Spanish (synthetic)",
        "color": "#f43f5e",
    },
}

# Metrics to exclude globally
EXCLUDED_METRICS = {"bleu_diff", "rouge1_diff", "rouge2_diff", "rougeL_diff"}


def find_latest_results_json(directory):
    """Find the newest results_*.json or results.json under directory."""
    # Try results.json first (simple format)
    simple = os.path.join(directory, "results.json")
    if os.path.isfile(simple):
        return simple
    # Try nested results_*.json
    pattern = os.path.join(directory, "**", "results_*.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    files.sort(key=lambda f: os.path.basename(f))
    return files[-1]


def _get_stderr(task_results, metric_name, metric_suffix, n_samples, metric_scale):
    """Get stderr for a metric from task results, estimating if missing."""
    stderr_key = f"{metric_name}_stderr,{metric_suffix}"
    se = task_results.get(stderr_key)
    if isinstance(se, (int, float)):
        return se
    if n_samples and n_samples > 1:
        val_key = f"{metric_name},{metric_suffix}"
        val = task_results.get(val_key)
        if isinstance(val, (int, float)):
            if metric_scale == "percent":
                p = max(0.0, min(1.0, val / 100.0))
                if 0 < p < 1:
                    return math.sqrt(p * (1 - p) / n_samples) * 100
            else:
                p = max(0.0, min(1.0, val))
                if 0 < p < 1:
                    return math.sqrt(p * (1 - p) / n_samples)
    return None


def extract_benchmark_scores(results_json_path, benchmark_name, task_config_entry=None):
    """Extract metrics from a single results JSON (one partition).

    Returns dict {metric_name: (value, stderr)} or None.
    """
    with open(results_json_path) as f:
        data = json.load(f)

    results = data.get("results", {})
    n_samples_dict = data.get("n-samples", {})
    metric_scale = (
        task_config_entry.get("metric_scale", "unit") if task_config_entry else "unit"
    )

    metrics = {}
    for task_key, task_results in results.items():
        # Match the benchmark name with partition suffix
        if not (
            task_key == benchmark_name
            or task_key.startswith(f"{benchmark_name}_p")
        ):
            # Also check for group-level results (global_mmlu)
            if not task_key.startswith(f"{benchmark_name}_"):
                continue
            # Skip subtask results (e.g., global_mmlu_french_business_p0)
            # Only keep the main group aggregate
            continue

        ns_entry = n_samples_dict.get(task_key, {})
        n_samples = ns_entry.get("effective") or ns_entry.get("original")

        for key, val in task_results.items():
            if key == "alias":
                continue
            # Handle both ",none" and other suffixes like ",remove_whitespace"
            if "," not in key:
                continue
            if "_stderr," in key:
                continue
            metric_name, metric_suffix = key.rsplit(",", 1)
            if metric_name in EXCLUDED_METRICS:
                continue
            if isinstance(val, (int, float)):
                se = _get_stderr(
                    task_results, metric_name, metric_suffix, n_samples, metric_scale
                )
                metrics[metric_name] = (val, se)

    # For group tasks (like global_mmlu), also check group-level results
    groups = data.get("groups", {})
    for group_key, group_results in groups.items():
        if not (
            group_key == benchmark_name
            or group_key.startswith(f"{benchmark_name}_p")
        ):
            continue
        for key, val in results.get(group_key, {}).items():
            if key == "alias":
                continue
            if "," not in key:
                continue
            if "_stderr," in key:
                continue
            metric_name, metric_suffix = key.rsplit(",", 1)
            if metric_name in EXCLUDED_METRICS:
                continue
            if isinstance(val, (int, float)):
                ns_entry = n_samples_dict.get(group_key, {})
                n_samples = ns_entry.get("effective") or ns_entry.get("original")
                se = _get_stderr(
                    results.get(group_key, {}),
                    metric_name,
                    metric_suffix,
                    n_samples,
                    metric_scale,
                )
                metrics[metric_name] = (val, se)

    return metrics if metrics else None


def aggregate_partitions(partition_metrics_list):
    """Aggregate metrics across partitions (p0, p1, p2) like prompt variants.

    partition_metrics_list: list of {metric_name: (value, stderr)}
    Returns dict {metric_name: {"max": ..., "mean": ..., ...}}
    """
    # Collect values per metric across partitions
    metric_values = {}
    for pmetrics in partition_metrics_list:
        if pmetrics is None:
            continue
        for metric_name, (val, se) in pmetrics.items():
            if metric_name not in metric_values:
                metric_values[metric_name] = []
            metric_values[metric_name].append((val, se))

    if not metric_values:
        return None

    out = {}
    for metric_name, pairs in metric_values.items():
        values = [v for v, _ in pairs]
        stderrs = [se for _, se in pairs]

        entry = {
            "max": round(max(values), 6),
            "mean": round(statistics.mean(values), 6),
            "median": round(statistics.median(values), 6),
            "min": round(min(values), 6),
        }

        max_idx = values.index(max(values))
        entry["max_prompt_idx"] = max_idx
        if stderrs[max_idx] is not None:
            entry["max_stderr"] = round(stderrs[max_idx], 6)

        min_idx = values.index(min(values))
        if stderrs[min_idx] is not None:
            entry["min_stderr"] = round(stderrs[min_idx], 6)

        if all(se is not None for se in stderrs):
            n = len(stderrs)
            mean_se = math.sqrt(sum(se**2 for se in stderrs)) / n
            entry["mean_stderr"] = round(mean_se, 6)

        med = statistics.median(values)
        closest_idx = min(range(len(values)), key=lambda i: abs(values[i] - med))
        if stderrs[closest_idx] is not None:
            entry["median_stderr"] = round(stderrs[closest_idx], 6)

        entry["n_prompts"] = len(values)
        if len(values) >= 2:
            entry["prompt_sd"] = round(statistics.stdev(values), 6)
            med_val = statistics.median(values)
            entry["prompt_mad"] = round(
                statistics.median([abs(v - med_val) for v in values]), 6
            )
        else:
            entry["prompt_sd"] = 0.0
            entry["prompt_mad"] = 0.0

        out[metric_name] = entry
    return out


def process_checkpoint(ckpt_path, task_configs):
    """Process a single checkpoint directory.

    Returns {benchmark: {"0": {metric: {...}}, ...}}
    """
    scores = {}
    for benchmark, config in task_configs.items():
        bench_path = os.path.join(ckpt_path, benchmark)
        if not os.path.isdir(bench_path):
            continue

        # Collect partitions
        partitions = sorted(
            [
                d
                for d in os.listdir(bench_path)
                if os.path.isdir(os.path.join(bench_path, d))
                and d.startswith("p")
                and d[1:].isdigit()
            ]
        )
        if not partitions:
            continue

        # Each partition is like a prompt variant
        partition_results = []
        for part in partitions:
            part_path = os.path.join(bench_path, part)
            results_file = find_latest_results_json(part_path)
            if results_file is None:
                continue
            metrics = extract_benchmark_scores(results_file, benchmark, config)
            partition_results.append(metrics)

        if not partition_results:
            continue

        agg = aggregate_partitions(partition_results)
        if agg is not None:
            # All partitions are 0-shot currently (no shot subdirectories)
            scores[benchmark] = {"0": agg}

    return scores


def discover_language_tasks(lang_dir):
    """Discover which tasks exist for a language by scanning all models/checkpoints."""
    tasks = set()
    for model_dir in os.listdir(lang_dir):
        model_path = os.path.join(lang_dir, model_dir)
        if not os.path.isdir(model_path) or model_dir.startswith("."):
            continue
        for ckpt in os.listdir(model_path):
            ckpt_path = os.path.join(model_path, ckpt)
            if not os.path.isdir(ckpt_path) or ckpt.startswith("."):
                continue
            for task in os.listdir(ckpt_path):
                task_path = os.path.join(ckpt_path, task)
                if os.path.isdir(task_path) and not task.startswith("."):
                    tasks.add(task)
    return tasks


def main():
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)

    output = {"languages": {}}

    for lang_name in sorted(os.listdir(RESULTS_DIR)):
        lang_dir = RESULTS_DIR / lang_name
        if not lang_dir.is_dir() or lang_name.startswith("."):
            continue

        print(f"\n=== Processing language: {lang_name} ===")

        # Discover tasks and build metrics_setup for this language
        lang_tasks = discover_language_tasks(str(lang_dir))
        task_configs = {}
        discovered_metrics = {}
        for task in sorted(lang_tasks):
            if task in TASK_CONFIG:
                task_configs[task] = TASK_CONFIG[task]
            else:
                print(f"  WARNING: No config for task '{task}', skipping")

        lang_data = {
            "metrics_setup": {},
            "models": {},
        }

        # Process each model
        for model_dir in sorted(os.listdir(lang_dir)):
            model_path = lang_dir / model_dir
            if not model_path.is_dir() or model_dir.startswith("."):
                continue

            model_cfg = MODEL_CONFIG.get(model_dir, {})
            display_name = model_cfg.get("display_name", model_dir)
            color = model_cfg.get("color", "#6366f1")

            print(f"  Model: {model_dir} ({display_name})")

            progress = {}
            for ckpt_name in sorted(os.listdir(model_path)):
                ckpt_path = model_path / ckpt_name
                if not ckpt_path.is_dir() or ckpt_name.startswith("."):
                    continue

                # Parse checkpoint step
                if ckpt_name.isdigit():
                    step = int(ckpt_name)
                elif ckpt_name == "main":
                    step = "main"
                else:
                    continue

                scores = process_checkpoint(str(ckpt_path), task_configs)
                if scores:
                    progress[step] = scores
                    # Track discovered metrics
                    for bench, shot_data in scores.items():
                        for shot, metric_data in shot_data.items():
                            if bench not in discovered_metrics:
                                discovered_metrics[bench] = set()
                            discovered_metrics[bench].update(metric_data.keys())

                print(f"    Checkpoint {ckpt_name}: {len(scores)} tasks")

            lang_data["models"][model_dir] = {
                "display_name": display_name,
                "color": color,
                "progress": progress,
            }

        # Build metrics_setup for this language
        for task, config in task_configs.items():
            disc = discovered_metrics.get(task, set())
            if not disc:
                continue
            main_metric = config["main_metric"]
            max_perf = 100.0 if config.get("metric_scale") == "percent" else 1.0
            base_metrics = sorted(disc - {main_metric})
            available_metrics = (
                ([main_metric] if main_metric in disc else []) + base_metrics
            )

            lang_data["metrics_setup"][task] = {
                "pretty_name": config["pretty_name"],
                "main_metric": main_metric,
                "random_baseline": config["random_baseline"],
                "max_performance": max_perf,
                "category": config.get("category", "uncategorized"),
                "metric_scale": config.get("metric_scale", "unit"),
                "available_metrics": available_metrics,
            }

        # Resolve "main" checkpoint: map it to the max numeric step across
        # all models in this language (it's the final checkpoint)
        max_numeric_step = 0
        for md in lang_data["models"].values():
            for step_key in md["progress"]:
                if isinstance(step_key, int):
                    max_numeric_step = max(max_numeric_step, step_key)
        for md in lang_data["models"].values():
            if "main" in md["progress"] and max_numeric_step > 0:
                md["progress"][max_numeric_step] = md["progress"].pop("main")

        output["languages"][lang_name] = lang_data

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, ensure_ascii=False)

    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\nWritten {OUTPUT_FILE} ({size_kb:.1f} KB)")
    for lang, ld in output["languages"].items():
        models = list(ld["models"].keys())
        tasks = list(ld["metrics_setup"].keys())
        print(f"  {lang}: {len(models)} models, {len(tasks)} tasks")
        for m in models:
            steps = sorted(ld["models"][m]["progress"].keys(), key=lambda x: (isinstance(x, str), x))
            print(f"    {m}: {len(steps)} checkpoints")


if __name__ == "__main__":
    main()
