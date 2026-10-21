"""Rearrange the retained traffic data into empirical and model-comparison figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def ecdf(values):
    values, counts = np.unique(values, return_counts=True)
    return values, np.cumsum(counts) / counts.sum()


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.tick_params(width=0.6, length=3)


def save(figure, name, output):
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    figure.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "data/results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figure")
    args = parser.parse_args()
    source = args.results_dir / "plot_data.npz"
    with np.load(source) as archive:
        data = {key: archive[key] for key in archive.files}
    result = json.loads((args.results_dir / "results.json").read_text(encoding="utf-8"))
    total_duration = result["all"]["duration_s"]
    fit_duration = result["fit"]["duration_s"]
    duration = result["holdout"]["duration_s"]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.65,
        "mathtext.fontset": "stix", "figure.facecolor": "white",
    })
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 1.55), gridspec_kw={"width_ratios": [1.75, 1]})
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.27, top=0.85, wspace=0.31)
    bins = np.linspace(0, total_duration, round(total_duration / 0.1) + 1)
    rates = np.histogram(data["all_t"], bins=bins, weights=data["all_w"])[0] * 8 / 0.1 / 1e6
    axes[0].axvspan(fit_duration, total_duration, color="#e9e9e9", zorder=0)
    axes[0].step(bins[:-1], rates, where="post", color="#171717", linewidth=0.8)
    axes[0].axvline(fit_duration, color="#777777", linestyle="--", linewidth=0.7)
    axes[0].text(0.34, 0.87, "Fitting", transform=axes[0].transAxes, ha="center", fontsize=8.5)
    axes[0].text(0.84, 0.87, "Validation", transform=axes[0].transAxes, ha="center", fontsize=8.5)
    axes[0].set(
        xlim=(0, total_duration), ylim=(0, max(150, float(rates.max()) * 1.05)),
        xticks=np.linspace(0, total_duration, 6),
        xlabel="Time since analysis start (s)", ylabel="Rate (Mbit/s)",
        title="(a) Measured traffic, 100 ms windows",
    )
    rows.extend(("profile_100ms", "measured", float(x), float(y)) for x, y in zip(bins[:-1], rates))
    for key, label, color, line in [
        ("train_w", "Fitting", "#171717", "-"),
        ("observed_w", "Validation", "#777777", "--"),
    ]:
        x, y = ecdf(data[key])
        axes[1].step(x, y, where="post", color=color, linestyle=line, linewidth=1.2, label=label)
        rows.extend(("size_cdf", label.lower(), float(a), float(b)) for a, b in zip(x, y))
    axes[1].set(
        xlim=(0, 1500), ylim=(0, 1.025), xticks=[0, 500, 1000, 1500],
        xlabel="IP packet length (bytes)", ylabel="CDF", title="(b) Measured packet lengths",
    )
    axes[1].legend(loc="upper left", frameon=False, handlelength=1.8, borderaxespad=0.2)
    fit_fraction = 100 * np.mean(data["train_w"] == 1452)
    validation_fraction = 100 * np.mean(data["observed_w"] == 1452)
    axes[1].text(0.04, 0.10, f"1452 bytes:\n{fit_fraction:.2f}% fitting\n{validation_fraction:.2f}% validation",
                 transform=axes[1].transAxes, fontsize=8.5, va="bottom")
    for axis in axes:
        style(axis)
    save(figure, "empirical-profile", output)

    styles = {
        "observed": ("Measured", "#171717", "-"),
        "poisson": ("Poisson", "#777777", "--"),
        "renewal": ("Empirical renewal", "#326b9e", ":"),
        "onoff": ("ON-OFF", "#bc6c25", "-."),
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 1.95))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.24, top=0.76, wspace=0.24)
    bins = np.linspace(0, duration, round(duration / 0.1) + 1)
    maximum = 0.0
    for key, (_, color, line) in styles.items():
        times, workloads = data[key + "_t"], data[key + "_w"]
        gaps = np.rint(np.diff(times) * 1e9) / 1e9
        x, y = ecdf(gaps)
        axes[0].step(x, y, where="post", color=color, linestyle=line, linewidth=1.3)
        rows.extend(("gap_cdf", key, float(a), float(b)) for a, b in zip(x, y))
        rates = np.histogram(times, bins=bins, weights=workloads)[0] * 8 / 0.1 / 1e6
        x, y = ecdf(rates)
        survival = 1 - y
        visible = survival > 0
        axes[1].step(x[visible], survival[visible], where="post", color=color, linestyle=line, linewidth=1.3)
        rows.extend(("rate_ccdf_100ms", key, float(a), float(b)) for a, b in zip(x[visible], survival[visible]))
        maximum = max(maximum, float(rates.max()))
    axes[0].set(
        xscale="log", xlim=(1e-7, 100), ylim=(0, 1.025),
        xlabel="Packet inter-arrival time (s)", ylabel="CDF", title="(a) Inter-arrival distributions",
    )
    axes[1].set(
        yscale="log", ylim=(5e-4, 1.15), xlim=(0, maximum * 1.05),
        xlabel="Rate in 100 ms windows (Mbit/s)", ylabel="P(rate > r)",
        title="(b) Short-window rate tails",
    )
    handles = [Line2D([0], [0], color=color, linestyle=line, linewidth=1.3, label=label)
               for label, color, line in styles.values()]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.02),
                  ncol=4, frameon=False, columnspacing=1.4, handlelength=2.5)
    for axis in axes:
        style(axis)
    save(figure, "model-comparison", output)
    with (output / "curve-data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "series", "x", "y"])
        writer.writerows(rows)
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    provenance = {
        "capture_run": result["run"],
        "source": "data/results/plot_data.npz", "source_sha256": source_hash,
        "empirical_profile": f"All {total_duration:g} measured seconds; last {duration:g} seconds shaded as validation",
        "size_cdfs": f"Fitting {fit_duration:g} seconds and validation {duration:g} seconds; no synthetic workload curve",
        "comparison": "Validation and replicate zero from each model; plotting does not run new simulations",
        "synthetic_seeds": {key: value["representative_seed"] for key, value in result["models"].items()},
        "curve_data": "curve-data.csv, alongside this provenance file",
        "assistance": "ChatGPT AI assistance with Python/Matplotlib plotting code and report revision",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Created two figures from retained arrays in {output}")


if __name__ == "__main__":
    main()
