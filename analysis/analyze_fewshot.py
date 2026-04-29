#!/usr/bin/env python3
"""Compare 0-shot vs 5-shot evaluation across all languages.

For every task in every language, compute the seven HPLT-E quality signals
(monotonicity, SNR, CV, prompt MAD, ranking consistency, prompt-switch rate,
non-randomness) under the 0-shot and 5-shot settings and visualize the per-task
distributions as side-by-side dot plots (one group per shot setting, one dot
per task).

Output: analysis/outputs/fewshot_signals.{png,pdf}
"""

import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "docs" / "data.json"
OUTPUT_DIR = BASE_DIR / "analysis" / "outputs"

SHOTS = ["0", "5"]
PROMPT_AGG = "max"

MIN_TOKENS = 10.0
MAX_TOKENS = 100.0

CRITERIA = [
    ("monotonicity", "Monotonicity\n(Spearman ρ)"),
    ("snr",          "SNR"),
    ("cv",           "CV (%)"),
    ("mad",          "Prompt MAD"),
    ("consistency",  "Consistency\n(Kendall τ)"),
    ("promptSwitch", "Prompt-switch\nrate (%)"),
    ("nonRandom",    "Non-randomness\n(max − baseline)"),
]

SHOT_COLORS = {
    "0": "#2563eb",  # blue
    "5": "#dc2626",  # red
}
SHOT_LABELS = {"0": "0-shot", "5": "5-shot"}


# ── Statistics helpers (mirroring docs/app.js) ───────────────────────────────

def _rank(values):
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def kendall_tau(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    total = n * (n - 1) // 2
    return 0.0 if total == 0 else (concordant - discordant) / total


def linear_interp(x_vals, y_vals, x):
    if not x_vals:
        return None
    if x <= x_vals[0]:
        return y_vals[0]
    if x >= x_vals[-1]:
        return y_vals[-1]
    for i in range(len(x_vals) - 1):
        if x_vals[i] <= x <= x_vals[i + 1]:
            dx = x_vals[i + 1] - x_vals[i]
            if dx == 0:
                return y_vals[i]
            t = (x - x_vals[i]) / dx
            return y_vals[i] + t * (y_vals[i + 1] - y_vals[i])
    return None


# ── Quality-signal computation ───────────────────────────────────────────────

def filter_display_scale(raw, info):
    return raw * 100 if info["metric_scale"] == "unit" else raw


def noise_scale(info):
    return 100 if info["metric_scale"] == "unit" else 1


def numeric_tokens(progress):
    out = []
    for k in progress:
        try:
            out.append(float(k))
        except (TypeError, ValueError):
            continue
    return sorted(out)


def token_key(progress, t):
    for k in progress:
        try:
            if float(k) == t:
                return k
        except (TypeError, ValueError):
            continue
    return None


def per_model_pairs(progress, benchmark, shot, main_metric, info, agg):
    pairs = []
    for t in numeric_tokens(progress):
        if t < MIN_TOKENS or t > MAX_TOKENS:
            continue
        k = token_key(progress, t)
        obj = progress.get(k, {}).get(benchmark, {}).get(shot, {}).get(main_metric)
        if not obj:
            continue
        raw = obj.get(agg)
        if raw is None:
            continue
        pairs.append({
            "tokens": t,
            "score": filter_display_scale(raw, info),
            "raw": raw,
            "obj": obj,
        })
    pairs.sort(key=lambda p: p["tokens"])
    return pairs


def compute_criterion(name, pairs, info, agg):
    nf = noise_scale(info)
    baseline_scaled = filter_display_scale(info.get("random_baseline", 0) or 0, info)

    if name == "nonRandom":
        if not pairs:
            return None
        max_score = max(p["score"] for p in pairs)
        return max(0.0, max_score - baseline_scaled)

    if len(pairs) < 3:
        return None

    if name == "monotonicity":
        return spearman([p["tokens"] for p in pairs], [p["score"] for p in pairs])

    if name == "snr":
        scores = [p["score"] for p in pairs]
        if agg == "median":
            noise_vals = [1.4826 * (p["obj"].get("prompt_mad") or 0) * nf for p in pairs]
        else:
            noise_vals = [(p["obj"].get("prompt_sd") or 0) * nf for p in pairs]
        mean_signal = max(0.0, sum(scores) / len(scores) - baseline_scaled)
        mean_noise = sum(noise_vals) / len(noise_vals)
        if mean_noise > 1e-10:
            return mean_signal / (mean_noise + 1e-8)
        return float("inf") if mean_signal > 0 else 0.0

    if name == "cv":
        scores = [p["score"] for p in pairs]
        mean = sum(scores) / len(scores)
        sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
        if abs(mean) > 1e-10:
            return (sd / abs(mean)) * 100
        return float("inf") if sd > 0 else 0.0

    if name == "mad":
        mads = [p["obj"].get("prompt_mad") for p in pairs if p["obj"].get("prompt_mad") is not None]
        if not mads:
            return None
        return statistics.median(m * nf for m in mads)

    if name == "promptSwitch":
        idxs = [p["obj"].get("max_prompt_idx") for p in pairs]
        if any(i is None for i in idxs) or len(idxs) < 2:
            return None
        switches = sum(1 for i in range(1, len(idxs)) if idxs[i] != idxs[i - 1])
        return (switches / (len(idxs) - 1)) * 100

    raise ValueError(f"Unknown criterion: {name}")


def compute_consistency(models, benchmark, shot, main_metric, info, agg):
    model_dirs = list(models.keys())
    if len(model_dirs) < 2:
        return None

    per_model = {}
    for m in model_dirs:
        prog = models[m]["progress"]
        xs, ys = [], []
        for t in numeric_tokens(prog):
            k = token_key(prog, t)
            obj = prog.get(k, {}).get(benchmark, {}).get(shot, {}).get(main_metric)
            if not obj:
                continue
            raw = obj.get(agg)
            if raw is None:
                continue
            xs.append(t)
            ys.append(filter_display_scale(raw, info))
        per_model[m] = (xs, ys)

    grid = sorted({t for (xs, _) in per_model.values() for t in xs
                   if MIN_TOKENS <= t <= MAX_TOKENS})
    if len(grid) < 2:
        return None

    rankings = []
    for t in grid:
        row, ok = [], True
        for m in model_dirs:
            xs, ys = per_model[m]
            v = linear_interp(xs, ys, t)
            if v is None:
                ok = False
                break
            row.append(v)
        if ok:
            rankings.append(row)
    if len(rankings) < 2:
        return None

    taus = []
    for i in range(len(rankings) - 1):
        tau = kendall_tau(rankings[i], rankings[i + 1])
        taus.append(tau if tau is not None else 0.0)
    return sum(taus) / len(taus) if taus else None


def compute_task_signals(task, lang_data, shot, agg):
    info = lang_data["metrics_setup"][task]
    main_metric = info["main_metric"]
    models = lang_data["models"]

    per_crit = {name: [] for name, _ in CRITERIA if name != "consistency"}
    for m in models:
        pairs = per_model_pairs(
            models[m]["progress"], task, shot, main_metric, info, agg
        )
        for name in per_crit:
            v = compute_criterion(name, pairs, info, agg)
            if v is not None and not math.isinf(v):
                per_crit[name].append(v)

    result = {name: (statistics.median(vs) if vs else None)
              for name, vs in per_crit.items()}
    result["consistency"] = compute_consistency(
        models, task, shot, main_metric, info, agg
    )
    return result


def task_has_shot(task, lang_data, shot):
    """True if any model has a numeric checkpoint that reports the given shot."""
    for m in lang_data["models"].values():
        for k, bench_map in m["progress"].items():
            try:
                float(k)
            except (TypeError, ValueError):
                continue
            if shot in (bench_map.get(task) or {}):
                return True
    return False


# ── Plotting ─────────────────────────────────────────────────────────────────

def jitter(n, width=0.12, seed=0):
    rng = np.random.default_rng(seed)
    if n <= 1:
        return np.zeros(n)
    return rng.uniform(-width, width, size=n)


def plot_signals(signals_by_shot, out_stem, n_tasks):
    n_crit = len(CRITERIA)
    n_cols = math.ceil(n_crit / 2)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 8.4), sharey=False)
    axes_flat = axes.flatten()

    for ax, (name, label) in zip(axes_flat, CRITERIA):
        for i, shot in enumerate(SHOTS):
            vals = [s[name] for s in signals_by_shot[shot] if s[name] is not None]
            x = i + jitter(len(vals), seed=(hash((name, shot)) & 0xFFFF))
            ax.scatter(x, vals, s=40, color=SHOT_COLORS[shot], alpha=0.72,
                       edgecolor="white", linewidth=0.5, zorder=3)
            if vals:
                ax.hlines(np.mean(vals), i - 0.28, i + 0.28,
                          color=SHOT_COLORS[shot], linewidth=2.4, zorder=4)

        ax.set_xticks(range(len(SHOTS)))
        ax.set_xticklabels([SHOT_LABELS[s] for s in SHOTS])
        ax.set_xlim(-0.55, len(SHOTS) - 0.45)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[n_crit:]:
        ax.set_visible(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="white",
               markerfacecolor=SHOT_COLORS[s], markersize=9,
               label=SHOT_LABELS[s])
        for s in SHOTS
    ] + [Line2D([0], [0], color="black", linewidth=2.4, label="mean")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(
        f"0-shot vs. 5-shot comparison across all languages — "
        f"{n_tasks} tasks, {MIN_TOKENS:g}–{MAX_TOKENS:g}B tokens, "
        f"prompt agg={PROMPT_AGG}",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{out_stem}.png"
    pdf = OUTPUT_DIR / f"{out_stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    with open(DATA_FILE) as f:
        data = json.load(f)

    # Only keep tasks that report BOTH shot settings (fair paired comparison).
    task_list = []
    skipped = []
    for lang_name, lang_data in data["languages"].items():
        for task in lang_data["metrics_setup"]:
            have = [s for s in SHOTS if task_has_shot(task, lang_data, s)]
            if len(have) == len(SHOTS):
                task_list.append((lang_name, task))
            else:
                skipped.append((lang_name, task, have))
    task_list.sort()

    print(f"Analyzing {len(task_list)} (language, task) pairs "
          f"with both 0-shot and 5-shot data "
          f"across {len(data['languages'])} languages.")
    if skipped:
        print(f"Skipped {len(skipped)} task(s) missing one of the shot settings:")
        for lang, task, have in skipped:
            print(f"  [{lang}] {task} -> only {have}")

    signals_by_shot = {s: [] for s in SHOTS}
    for lang_name, task in task_list:
        lang_data = data["languages"][lang_name]
        for shot in SHOTS:
            signals_by_shot[shot].append(
                compute_task_signals(task, lang_data, shot, PROMPT_AGG)
            )

    print("\nMeans (across tasks):")
    hdr = f"{'criterion':<14s}" + "".join(f"{SHOT_LABELS[s]:>10s}" for s in SHOTS)
    print(hdr)
    print("-" * len(hdr))
    for name, _ in CRITERIA:
        row = f"{name:<14s}"
        for shot in SHOTS:
            vs = [s[name] for s in signals_by_shot[shot] if s[name] is not None]
            mean = statistics.mean(vs) if vs else float("nan")
            row += f"{mean:>10.3f}"
        print(row)

    png, pdf = plot_signals(signals_by_shot, "fewshot_signals", len(task_list))
    print(f"\nWritten {png}")
    print(f"Written {pdf}")


if __name__ == "__main__":
    main()
