"""Make the README figures from the TensorBoard logs.
Writes figures/scaling.png and figures/loss_vs_time.png
The 1-GPU reference run is read from ../BASELINE/runs/a100_amp_opt.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

OUT = "figures"

# Same style as the BASELINE figures
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
IDEAL = "#d9d8d3"  # neutral, for the "perfect scaling" reference bars

CURVES = [  # (TensorBoard folder, label, color)
    ("../BASELINE/runs/a100_amp_opt", "1 GPU (BASELINE)", "#2a78d6"),
    ("runs/ddp-1n", "DDP, 1 node × 2 GPUs", "#eb6834"),
    ("runs/ddp-2n", "DDP, 2 nodes × 2 GPUs", "#1baf7a"),
]

# Throughput after warm-up (features/s), average of the 2 epochs, from logs/
SCALING = [
    ("1 GPU\n(BASELINE)", 1, (385.8 + 383.9) / 2),
    ("2 GPUs\n1 node", 2, (660.4 + 664.2) / 2),
    ("4 GPUs\n2 nodes", 4, (1157.8 + 1154.3) / 2),
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


def read_scalar(folder, tag):
    """All points of one tag, merged across every event file in the folder."""
    acc = event_accumulator.EventAccumulator(
        folder, size_guidance={event_accumulator.SCALARS: 0})
    acc.Reload()
    return acc.Scalars(tag)


def ema(values, alpha=0.9):
    out, s = [], values[0]
    for v in values:
        s = alpha * s + (1 - alpha) * v
        out.append(s)
    return out


def plot_scaling():
    base = SCALING[0][2]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", visible=False)
    x = list(range(len(SCALING)))
    ideal = [base * g for _, g, _ in SCALING]
    real = [t for _, _, t in SCALING]
    ax.bar(x, ideal, width=0.6, color=IDEAL, label="perfect scaling")
    ax.bar(x, real, width=0.44, color="#2a78d6", label="measured")
    for xi, (_, g, t) in zip(x, SCALING):
        ax.text(xi, t + 25, f"{t:.0f}/s\n{t / base:.2f}×, {t / (base * g):.0%}",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, [n for n, _, _ in SCALING], color=INK)
    ax.set_ylim(0, max(ideal) * 1.12)
    ax.set_ylabel("Training throughput (features/s)", color=INK2)
    ax.set_title("DDP scaling: 3.0× with 4 A100s (speed-up, efficiency)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scaling.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_loss_vs_time():
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style(ax)
    for folder, label, color in CURVES:
        pts = read_scalar(folder, "train/loss_step")
        t0 = pts[0].wall_time
        minutes = [(p.wall_time - t0) / 60 for p in pts]
        values = [p.value for p in pts]
        ax.plot(minutes, values, color=color, linewidth=0.6, alpha=0.18)
        ax.plot(minutes, ema(values), color=color, linewidth=2,
                label=f"{label}: {minutes[-1]:.1f} min")
    ax.set_ylim(0, 3)
    ax.set_xlabel("Wall-clock time since first logged step (minutes)", color=INK2)
    ax.set_ylabel("Training loss (smoothed)", color=INK2)
    ax.set_title("Same loss curve, reached sooner with more GPUs (full SQuAD, 2 epochs)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "loss_vs_time.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plot_scaling()
    plot_loss_vs_time()
    print("wrote", os.path.join(OUT, "scaling.png"), "and",
          os.path.join(OUT, "loss_vs_time.png"))
