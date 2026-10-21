"""Fit three source models and compare fixed-seed simulations with held-out traffic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def groups(times, threshold):
    """Return the first and last packet indices of each activity period."""
    split = np.flatnonzero(np.diff(times) > threshold) + 1
    return np.r_[0, split], np.r_[split - 1, len(times) - 1]


def metrics(times, sizes, duration, *, windows=(0.01, 0.1, 1.0), include_gaps=True):
    """Compute workload and count statistics, preserving the original definitions."""
    output = {
        "duration_s": duration,
        "packets": len(times),
        "bytes": int(sizes.sum()),
        "mean_Mbps": float(8 * sizes.sum() / duration / 1e6),
    }
    gaps = np.rint(np.diff(times) * 1e9) / 1e9 if include_gaps else np.empty(0)
    if len(gaps):
        output.update({
            "mean_gap_ms": float(gaps.mean() * 1e3),
            "median_gap_us": float(np.median(gaps) * 1e6),
            "cv_gap": float(gaps.std(ddof=1) / gaps.mean()),
            "lag1_gap_correlation": float(np.corrcoef(gaps[:-1], gaps[1:])[0, 1]),
            "max_gap_s": float(gaps.max()),
            "min_gap_ns": float(gaps.min() * 1e9),
        })
    output["mean_size_bytes"] = float(sizes.mean()) if len(sizes) else None
    output["windows"] = {}
    for window in windows:
        bins = np.linspace(0, duration, round(duration / window) + 1)
        counts = np.histogram(times, bins=bins)[0]
        rates = np.histogram(times, bins=bins, weights=sizes)[0] * 8 / window / 1e6
        output["windows"][str(window)] = {
            "empty_fraction": float(np.mean(counts == 0)),
            "count_fano": float(counts.var(ddof=1) / counts.mean()) if counts.mean() else 0.0,
            "p99_Mbps": float(np.quantile(rates, 0.99)),
            "peak_Mbps": float(rates.max()),
            "mean_Mbps": float(rates.mean()),
        }
    return output


def cdf_distance(reference_sorted, sample):
    """Maximum empirical-CDF difference; the unchanged reference is sorted once."""
    if not len(reference_sorted) or not len(sample):
        return None
    other = np.sort(sample)
    grid = np.union1d(reference_sorted, other)
    reference_cdf = np.searchsorted(reference_sorted, grid, side="right") / len(reference_sorted)
    other_cdf = np.searchsorted(other, grid, side="right") / len(other)
    return float(np.max(np.abs(reference_cdf - other_cdf)))


class Models:
    """Fitting distributions and the three packet generators.

    Workloads and ordinary ON/OFF durations are sampled independently.
    Initial renewal intervals/cycles are length-biased for random-phase observation.
    """

    def __init__(self, train_t, train_w, train_duration, threshold):
        self.gaps = np.rint(np.diff(train_t) * 1e9) / 1e9
        self.sizes, counts = np.unique(train_w, return_counts=True)
        self.size_probabilities = counts / counts.sum()
        self.rate = len(train_t) / train_duration
        self.gap_probabilities = self.gaps / self.gaps.sum()
        self.mean_size = float(np.dot(self.sizes, self.size_probabilities))
        starts, ends = groups(train_t, threshold)
        complete = (train_t[starts] >= threshold) & (train_t[ends] <= train_duration - threshold)
        indices = [j for j in range(len(starts) - 1) if complete[j]]
        if len(indices) < 3:
            raise ValueError("Too few complete activity cycles for this fitted model.")
        self.on = np.array([train_t[ends[j]] - train_t[starts[j]] for j in indices])
        self.off = np.array([train_t[starts[j + 1]] - train_t[ends[j]] for j in indices])
        self.active_gaps = np.concatenate([
            np.rint(np.diff(train_t[starts[j]:ends[j] + 1]) * 1e9) / 1e9
            for j in indices
        ])
        self.active_counts = np.array([ends[j] - starts[j] + 1 for j in indices])
        self.cycle_count = len(indices)
        self.cycle_lengths = self.on[:, None] + self.off[None, :]
        self.initial_cycle_probabilities = self.cycle_lengths.ravel() / self.cycle_lengths.sum()

    def workload(self, rng, count):
        return rng.choice(self.sizes, size=count, p=self.size_probabilities)

    def poisson(self, rng, duration):
        count = rng.poisson(self.rate * duration)
        times = np.sort(rng.uniform(0, duration, count))
        return times, self.workload(rng, count)

    def renewal(self, rng, duration):
        containing_gap = rng.choice(self.gaps, p=self.gap_probabilities)
        first = rng.uniform(0, containing_gap)
        parts = [np.array([first])] if first < duration else []
        current = first
        block_size = max(1024, round(duration / self.gaps.mean() * 1.25))
        while current < duration:
            candidate = current + np.cumsum(rng.choice(self.gaps, size=block_size))
            parts.append(candidate[candidate < duration])
            current = candidate[-1]
        times = np.concatenate(parts) if parts else np.empty(0)
        return times, self.workload(rng, len(times))

    def onoff(self, rng, duration):
        pair = rng.choice(self.cycle_lengths.size, p=self.initial_cycle_probabilities)
        i, j = np.unravel_index(pair, self.cycle_lengths.shape)
        on, off = self.on[i], self.off[j]
        start = -rng.uniform(0, on + off)
        parts = []
        while start < duration:
            if start + on >= 0:
                positions = [np.array([0.0])]
                current = 0.0
                block_size = max(1024, round(on / self.active_gaps.mean() * 1.3))
                while current < on:
                    candidate = current + np.cumsum(rng.choice(self.active_gaps, size=block_size))
                    positions.append(candidate[candidate < on])
                    current = candidate[-1]
                local = np.concatenate(positions) + start
                parts.append(local[(local >= 0) & (local < duration)])
            start += on + off
            on, off = rng.choice(self.on), rng.choice(self.off)
        times = np.concatenate(parts) if parts else np.empty(0)
        return times, self.workload(rng, len(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiment.json")
    parser.add_argument("--packets", type=Path, default=ROOT / "data/processed/packets.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/results")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = config["seed"] if args.seed is None else args.seed
    replicates = config["replicates"] if args.replicates is None else args.replicates
    if replicates < 2:
        parser.error("--replicates must be at least 2 for ensemble standard deviations")
    interval = config["intervals_s"]
    beginning, split, end = interval["analysis_start"], interval["fitting_end"], interval["analysis_end"]
    train_duration, test_duration = split - beginning, end - split
    if not 0 <= beginning < split < end:
        parser.error("The configured fitting and validation intervals must have positive duration")
    threshold = config["onoff_threshold_s"]
    with args.packets.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    t = np.array([float(row["capture_time_s"]) for row in rows])
    w = np.array([int(row["ip_length_bytes"]) for row in rows])
    mask = (t >= beginning) & (t < end)
    all_t, all_w = t[mask] - beginning, w[mask]
    training = all_t < train_duration
    train_t, train_w = all_t[training], all_w[training]
    test_t, test_w = all_t[~training] - train_duration, all_w[~training]
    if len(train_t) < 2 or len(test_t) < 2 or not np.all(np.diff(all_t) > 0):
        raise ValueError("Expected strictly increasing packets and nonempty fitting/validation samples.")
    if not np.all(all_w > 0):
        raise ValueError("Packet workloads must be positive.")
    models = Models(train_t, train_w, train_duration, threshold)
    checks = {}
    for candidate_threshold in [0.1, 0.5, 1.0]:
        starts, ends = groups(train_t, candidate_threshold)
        checks[str(candidate_threshold)] = {
            "groups": len(starts),
            "complete_cycles": int(sum(
                train_t[starts[j]] >= candidate_threshold
                and train_t[ends[j]] <= train_duration - candidate_threshold
                for j in range(len(starts) - 1)
            )),
        }
    actual = metrics(test_t, test_w, test_duration)
    actual["cdf_gap_reference"] = "Successive selected packet times strictly inside the 90-second holdout"
    results = {
        "run": config["run"], "seed": seed, "replicates_per_model": replicates,
        "fit_interval_capture_s": [beginning, split],
        "holdout_interval_capture_s": [split, end],
        "all": metrics(all_t, all_w, end - beginning),
        "fit": metrics(train_t, train_w, train_duration), "holdout": actual,
        "fit_parameters": {
            "poisson_lambda_packets_s": models.rate,
            "renewal_mean_gap_ms": float(models.gaps.mean() * 1e3),
            "batch_distribution": {"1": 1.0},
            "size_support_bytes": models.sizes.tolist(),
            "size_probabilities": models.size_probabilities.tolist(),
            "active_gap_mean_us": float(models.active_gaps.mean() * 1e6),
            "active_gap_samples": len(models.active_gaps),
            "complete_cycles": models.cycle_count,
            "on_durations_s": models.on.tolist(), "off_durations_s": models.off.tolist(),
            "active_packet_counts": models.active_counts.tolist(),
            "mean_on_s": float(models.on.mean()), "mean_off_s": float(models.off.mean()),
            "threshold_s": threshold, "threshold_checks_fit_only": checks,
            "mean_size_bytes": models.mean_size,
            "max_size_mass": float(models.size_probabilities[-1]),
            "poisson_longrun_Mbps": 8 * models.rate * models.mean_size / 1e6,
            "renewal_longrun_Mbps": 8 * models.mean_size / models.gaps.mean() / 1e6,
            "onoff_renewal_rate_approx_Mbps": 8 * models.mean_size / models.active_gaps.mean()
            * models.on.mean() / (models.on.mean() + models.off.mean()) / 1e6,
        },
        "models": {},
        "limitations": [
            "Only nine fully observed fitting cycles; boundary-censored ON periods are excluded from duration fitting.",
            "The chronological validation boundary intersects an ON period, which can affect short-window totals.",
            "Independent empirical draws are stochastic models, not replayed timestamp blocks.",
            "Software receiver timestamps and network packet sizes describe the observed source after transport and path effects.",
            "Shared Wi-Fi conditions and one video/session do not establish general application behavior.",
            "Reported Monte Carlo spreads are predictive simulation variation, not confidence intervals for fitted parameters.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = {
        "observed_t": test_t, "observed_w": test_w, "all_t": all_t, "all_w": all_w,
        "train_t": train_t, "train_w": train_w,
    }
    reference_gaps = np.sort(np.rint(np.diff(test_t) * 1e9))
    reference_sizes = np.sort(test_w)
    for model_index, name in enumerate(["poisson", "renewal", "onoff"]):
        simulations = []
        for replicate in range(replicates):
            rng = np.random.default_rng(seed + 10000 * model_index + replicate)
            tt, ww = getattr(models, name)(rng, test_duration)
            if not (
                len(tt) == len(ww) and np.all(np.diff(tt) > 0)
                and np.all((tt >= 0) & (tt < test_duration)) and np.all(ww > 0)
            ):
                raise ValueError(f"Invalid synthetic trace: {name}, replicate {replicate}")
            score = metrics(tt, ww, test_duration) if replicate == 0 else metrics(
                tt, ww, test_duration, windows=(0.1,), include_gaps=False,
            )
            score["gap_cdf_distance"] = cdf_distance(reference_gaps, np.rint(np.diff(tt) * 1e9))
            score["size_cdf_distance"] = cdf_distance(reference_sizes, ww)
            simulations.append(score)
            if replicate == 0:
                archive[name + "_t"], archive[name + "_w"] = tt, ww
                with (args.output_dir / f"{name}_synthetic_packets.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["time_s", "ip_length_bytes", "batch_size"])
                    writer.writerows((f"{a:.9f}", int(b), 1) for a, b in zip(tt, ww))
        flat = {
            "mean_Mbps": [score["mean_Mbps"] for score in simulations],
            "empty_fraction_100ms": [score["windows"]["0.1"]["empty_fraction"] for score in simulations],
            "count_fano_100ms": [score["windows"]["0.1"]["count_fano"] for score in simulations],
            "p99_Mbps_100ms": [score["windows"]["0.1"]["p99_Mbps"] for score in simulations],
            "gap_cdf_distance": [score["gap_cdf_distance"] for score in simulations],
            "size_cdf_distance": [score["size_cdf_distance"] for score in simulations],
        }
        aggregate = {}
        for key, values in flat.items():
            values = np.asarray(values)
            aggregate[key] = {
                "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                "p025": float(np.quantile(values, 0.025)), "p975": float(np.quantile(values, 0.975)),
            }
        results["models"][name] = {
            "representative_seed": seed + 10000 * model_index,
            "representative": simulations[0], "ensemble": aggregate,
        }
        print(f"{name}: {replicates} traces, mean rate {aggregate['mean_Mbps']['mean']:.6f} Mbit/s", flush=True)
    np.savez_compressed(args.output_dir / "plot_data.npz", **archive)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
