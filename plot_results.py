"""Plot aggregate metrics only; no source images are exported."""

from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = Path(__file__).resolve().parent / "results"
metrics = pd.read_csv(results / "reconstruction-metrics.csv")
fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
ax.bar(metrics.model, metrics.test_pixel_mse, color=["#23597a", "#7898aa", "#ad6b34"])
ax.set(
    ylabel="Held-out pixel MSE (lower is better)",
    title="64-dimensional representations; 1,000 test images",
)
for i, value in enumerate(metrics.test_pixel_mse):
    ax.text(i, value + 0.0006, f"{value:.4f}", ha="center")
ax.set_ylim(0, metrics.test_pixel_mse.max() * 1.18)
fig.savefig(results / "reconstruction-comparison.png", dpi=170)
plt.close(fig)
