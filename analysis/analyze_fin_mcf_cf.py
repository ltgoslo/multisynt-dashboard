#!/usr/bin/env python3
"""Compare CF vs MCF prompt formulations on Finnish benchmarks.

For each Finnish task whose name contains `_cf_` or `_mcf_`, compute the seven
HPLT-E quality signals used by the dashboard (monotonicity, SNR, CV, prompt MAD,
ranking consistency, prompt-switch rate, non-randomness) under both 0-shot and
5-shot evaluation, pool the per-(task, shot) observations, and visualize the
resulting distributions as side-by-side dot plots.

Output: analysis/outputs/fin_mcf_vs_cf_signals.{png,pdf}
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

LANGUAGE = "Finnish"
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

CF_COLOR = "#2563eb"   # blue
MCF_COLOR = "#dc2626"  # red


# ── Statistics helpers (mirroring docs/app.js) ───────────────────────────────

def _rank(values):
    """Average ranks (1-indexed), ties get mean rank."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-indexed
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
    """Linear interpolation with nearest-edge extrapolation. Returns None if empty."""
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
    """Convert a raw score to percentage-scale, matching HPLT-E convention."""
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
    """Find the original string key corresponding to numeric token value t."""
    for k in progress:
        try:
            if float(k) == t:
                return k
        except (TypeError, ValueError):
            continue
    return None


def per_model_pairs(progress, benchmark, shot, main_metric, info):
    """Collect (tokens, score, obj) per checkpoint within the analysis window."""
    pairs = []
    for t in numeric_tokens(progress):
        if t < MIN_TOKENS or t > MAX_TOKENS:
            continue
        k = token_key(progress, t)
        obj = progress.get(k, {}).get(benchmark, {}).get(shot, {}).get(main_metric)
        if not obj:
            continue
        raw = obj.get(PROMPT_AGG)
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


def compute_criterion(name, pairs, info):
    """Replicate per-model criterion logic from docs/app.js."""
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
        if PROMPT_AGG == "median":
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


def compute_consistency(models, benchmark, shot, main_metric, info):
    """Kendall τ of model rankings between consecutive checkpoint positions."""
    model_dirs = list(models.keys())
    if len(model_dirs) < 2:
        return None

    # Pre-sort tokens per model for interpolation (in display-scale space).
    per_model = {}
    for m in model_dirs:
        prog = models[m]["progress"]
        ts = numeric_tokens(prog)
        ys = []
        xs = []
        for t in ts:
            k = token_key(prog, t)
            obj = prog.get(k, {}).get(benchmark, {}).get(shot, {}).get(main_metric)
            if not obj:
                continue
            raw = obj.get(PROMPT_AGG)
            if raw is None:
                continue
            xs.append(t)
            ys.append(filter_display_scale(raw, info))
        per_model[m] = (xs, ys)

    # Evaluation grid: union of all checkpoint tokens within the window.
    grid = sorted({t for (xs, _) in per_model.values() for t in xs
                   if MIN_TOKENS <= t <= MAX_TOKENS})
    if len(grid) < 2:
        return None

    rankings = []
    for t in grid:
        row = []
        ok = True
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


def compute_task_signals(task, lang_data, shot):
    """Return {criterion: value} for a single task, using the median over models."""
    info = lang_data["metrics_setup"][task]
    main_metric = info["main_metric"]
    models = lang_data["models"]

    per_crit = {name: [] for name, _ in CRITERIA if name != "consistency"}
    for m in models:
        pairs = per_model_pairs(
            models[m]["progress"], task, shot, main_metric, info
        )
        for name in per_crit:
            v = compute_criterion(name, pairs, info)
            if v is not None and not math.isinf(v):
                per_crit[name].append(v)

    result = {name: (statistics.median(vs) if vs else None)
              for name, vs in per_crit.items()}
    result["consistency"] = compute_consistency(
        models, task, shot, main_metric, info
    )
    return result


# ── Plotting ─────────────────────────────────────────────────────────────────

def jitter(n, width=0.12, seed=0):
    rng = np.random.default_rng(seed)
    if n <= 1:
        return np.zeros(n)
    return rng.uniform(-width, width, size=n)


def plot_signals(cf_signals, mcf_signals, out_stem):
    n_crit = len(CRITERIA)
    n_cols = math.ceil(n_crit / 2)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.4 * n_cols, 8.4), sharey=False)
    axes_flat = axes.flatten()

    for ax, (name, label) in zip(axes_flat, CRITERIA):
        cf_vals = [v for v in (s[name] for s in cf_signals) if v is not None]
        mcf_vals = [v for v in (s[name] for s in mcf_signals) if v is not None]

        x_cf = 0.0 + jitter(len(cf_vals), seed=hash(name) & 0xFFFF)
        x_mcf = 1.0 + jitter(len(mcf_vals), seed=(hash(name) + 1) & 0xFFFF)

        ax.scatter(x_cf, cf_vals, s=44, color=CF_COLOR, alpha=0.75,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(x_mcf, mcf_vals, s=44, color=MCF_COLOR, alpha=0.75,
                   edgecolor="white", linewidth=0.5, zorder=3)

        # Mean markers (horizontal bars)
        if cf_vals:
            ax.hlines(np.mean(cf_vals), -0.28, 0.28, color=CF_COLOR,
                      linewidth=2.4, zorder=4)
        if mcf_vals:
            ax.hlines(np.mean(mcf_vals), 0.72, 1.28, color=MCF_COLOR,
                      linewidth=2.4, zorder=4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["CF", "MCF"])
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[n_crit:]:
        ax.set_visible(False)

    # Shared legend / supertitle
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor=CF_COLOR,
               markersize=9, label="CF"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=MCF_COLOR,
               markersize=9, label="MCF"),
        Line2D([0], [0], color="black", linewidth=2.4, label="mean"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.01))
    shots_label = " + ".join(f"{s}-shot" for s in SHOTS)
    fig.suptitle(
        f"Finnish benchmarks — CF vs. MCF quality signals "
        f"({shots_label} pooled, {MIN_TOKENS:g}–{MAX_TOKENS:g}B tokens, "
        f"prompt agg={PROMPT_AGG})",
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
    lang_data = data["languages"][LANGUAGE]

    cf_tasks, mcf_tasks = [], []
    for task in lang_data["metrics_setup"]:
        if "_cf_" in task:
            cf_tasks.append(task)
        elif "_mcf_" in task:
            mcf_tasks.append(task)

    cf_tasks.sort()
    mcf_tasks.sort()

    print(f"Found {len(cf_tasks)} CF tasks and {len(mcf_tasks)} MCF tasks in {LANGUAGE}.")
    print(f"Pooling shot settings: {SHOTS}.")

    # One observation per (task, shot).
    cf_signals = [
        compute_task_signals(t, lang_data, shot)
        for t in cf_tasks for shot in SHOTS
    ]
    mcf_signals = [
        compute_task_signals(t, lang_data, shot)
        for t in mcf_tasks for shot in SHOTS
    ]
    cf_labels = [(t, shot) for t in cf_tasks for shot in SHOTS]
    mcf_labels = [(t, shot) for t in mcf_tasks for shot in SHOTS]

    # Summary table
    header = f"{'task':<48s}{'shot':>6s}" + "".join(f"{n[:10]:>12s}" for n, _ in CRITERIA)
    print("\n" + header)
    print("-" * len(header))
    for (t, shot), sig in zip(cf_labels + mcf_labels, cf_signals + mcf_signals):
        row = f"{t:<48s}{shot:>6s}"
        for name, _ in CRITERIA:
            v = sig[name]
            row += f"{v:>12.3f}" if v is not None else f"{'—':>12s}"
        print(row)

    # Means
    print("\nMeans (across tasks):")
    for name, label in CRITERIA:
        cf_vals = [s[name] for s in cf_signals if s[name] is not None]
        mcf_vals = [s[name] for s in mcf_signals if s[name] is not None]
        mean_cf = statistics.mean(cf_vals) if cf_vals else float("nan")
        mean_mcf = statistics.mean(mcf_vals) if mcf_vals else float("nan")
        print(f"  {name:<14s} CF={mean_cf:8.3f}  MCF={mean_mcf:8.3f}")

    png, pdf = plot_signals(cf_signals, mcf_signals, "fin_mcf_vs_cf_signals")
    print(f"\nWritten {png}")
    print(f"Written {pdf}")


if __name__ == "__main__":
    main()
