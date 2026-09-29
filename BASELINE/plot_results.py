"""Make the README figures from the TensorBoard logs in runs/.

Usage (from BASELINE/, with tensorboard + matplotlib installed):
    python plot_results.py
Writes figures/loss_vs_time.png and figures/speedup_ladder.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

RUNS = "runs"
OUT = "figures"

# Colors: validated categorical slots 1-3 (light surface), text in neutral inks
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CURVES = [  # (run folder, label, color)
    ("a100_fp32", "FP32 (baseline)", "#2a78d6"),
    ("a100_amp", "AMP", "#eb6834"),
    ("a100_amp_opt", "AMP + batch 32 + compile", "#1baf7a"),
]

# Optimization ladder: throughput after warm-up (features/s), from logs/
LADDER = [
    ("FP32 baseline", 60.2),
    ("+ AMP", 196.8),
    ("+ no .item() per step", 206.0),
    ("+ 4 DataLoader workers", 203.2),
    ("+ batch 32, lr 3e-5", 339.3),
    ("+ torch.compile", 393.6),
]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def read_scalar(run, tag):
    """All points of one tag, merged across every event file in the folder."""
    acc = event_accumulator.EventAccumulator(
        os.path.join(RUNS, run), size_guidance={event_accumulator.SCALARS: 0})
    acc.Reload()
    return acc.Scalars(tag)


def ema(values, alpha=0.97):
    out, s = [], values[0]
    for v in values:
        s = alpha * s + (1 - alpha) * v
        out.append(s)
    return out


def plot_loss_vs_time():
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style(ax)
    for run, label, color in CURVES:
        pts = read_scalar(run, "train/loss_step")
        t0 = pts[0].wall_time
        minutes = [(p.wall_time - t0) / 60 for p in pts]
        values = [p.value for p in pts]
        ax.plot(minutes, values, color=color, linewidth=0.6, alpha=0.18)
        smooth = ema(values)
        ax.plot(minutes, smooth, color=color, linewidth=2,
                label=f"{label}: {minutes[-1]:.0f} min")
    ax.set_ylim(0, 3)
    ax.set_xlabel("Wall-clock time since first logged step (minutes)", color=INK2)
    ax.set_ylabel("Training loss (smoothed)", color=INK2)
    ax.set_title("Same loss curve, reached 6× sooner (A100, full SQuAD, 2 epochs)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "loss_vs_time.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_ladder():
    names = [n for n, _ in LADDER]
    tput = [t for _, t in LADDER]
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    y = list(range(len(names)))[::-1]
    ax.barh(y, tput, color="#2a78d6", height=0.6)
    for yi, t in zip(y, tput):
        ax.text(t + 6, yi, f"{t:.0f}/s  ({t / tput[0]:.1f}×)", va="center",
                fontsize=9, color=INK)
    ax.set_yticks(y, names, color=INK)
    ax.set_xlim(0, max(tput) * 1.3)
    ax.set_xlabel("Training throughput after warm-up (features/s)", color=INK2)
    ax.set_title("Speed-up ladder on one A100, one change at a time",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "speedup_ladder.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plot_loss_vs_time()
    plot_ladder()
    print("wrote", os.path.join(OUT, "loss_vs_time.png"), "and",
          os.path.join(OUT, "speedup_ladder.png"))
