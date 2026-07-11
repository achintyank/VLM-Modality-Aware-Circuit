"""
Bar chart for the vision-vs-text attention experiment (50 PixelProse images,
Qwen2-VL-2B-Instruct). Uses the aggregate numbers from the run; no model needed.
"""

import matplotlib.pyplot as plt

# --- aggregate results from the 50-image run (caption tokens only) ---
n_vision, n_caption = 17243, 5779
total_vision, total_caption = 2_735_271.2, 948_539.8
avg_vision, avg_caption = 158.63, 164.14

labels = ["Vision", "Caption"]
colors = ["#4C72B0", "#DD8452"]  # blue = vision, orange = caption

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

panels = [
    ("Number of tokens", [n_vision, n_caption], "{:,.0f}"),
    ("Total attention received", [total_vision, total_caption], "{:,.0f}"),
    ("Avg attention per token", [avg_vision, avg_caption], "{:,.1f}"),
]

for ax, (title, values, fmt) in zip(axes, panels):
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # value label on top of each bar
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(values) * 1.15)

fig.suptitle(
    "Attention allocation: vision vs. caption tokens\nQwen2-VL-2B-Instruct, 50 PixelProse images",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("attention_chart.png", dpi=150)
print("saved attention_chart.png")
