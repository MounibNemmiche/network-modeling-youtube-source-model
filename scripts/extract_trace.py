"""Extract receiver packet times and original IP lengths from the media capture."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def find_tshark(explicit: str | None) -> str:
    if explicit:
        return explicit
    executable = shutil.which("tshark")
    if executable:
        return executable
    candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Wireshark/tshark.exe"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError("Install Wireshark/TShark or supply --tshark PATH.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiment.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--tshark", help="Path to the TShark executable")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    capture = ROOT / config["capture_file"]
    start_ns = int(Decimal(config["capture_start_epoch"]) * 1_000_000_000)
    fields = [
        "frame.number", "frame.time_epoch", "ip.len", "tcp.srcport",
        "tcp.analysis.retransmission", "tcp.analysis.fast_retransmission",
        "udp.srcport",
    ]
    command = [
        find_tshark(args.tshark), "-n", "-r", str(capture),
        "-Y", config["selection_filter"], "-T", "fields", "-E", "header=y",
        "-E", "separator=,", "-E", "quote=d", "-E", "occurrence=f",
    ]
    for field in fields:
        command.extend(["-e", field])
    exported = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "packets.csv"
    previous_ns = None
    count = retransmissions = 0
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "frame_number", "timestamp_ns", "capture_time_s", "ip_length_bytes",
            "transport", "retransmission_flag",
        ])
        for row in csv.DictReader(io.StringIO(exported.stdout)):
            timestamp_ns = int(Decimal(row["frame.time_epoch"]) * 1_000_000_000)
            length = int(row["ip.len"])
            if length <= 0 or (previous_ns is not None and timestamp_ns <= previous_ns):
                raise ValueError("Positive lengths and strictly increasing timestamps are required.")
            relative = Decimal(timestamp_ns - start_ns) / Decimal(1_000_000_000)
            repeated = bool(row["tcp.analysis.retransmission"] or row["tcp.analysis.fast_retransmission"])
            writer.writerow([
                row["frame.number"], timestamp_ns, f"{relative:.9f}", length,
                "TCP" if row["tcp.srcport"] else "UDP", int(repeated),
            ])
            previous_ns = timestamp_ns
            count += 1
            retransmissions += int(repeated)
    if count == 0:
        raise ValueError("The selected capture contains no matching packets.")
    manifest = {
        "run": config["run"],
        "capture_file": config["capture_file"],
        "capture_sha256": hashlib.file_digest(capture.open("rb"), "sha256").hexdigest(),
        "packets": count,
        "retransmission_flags": retransmissions,
        "selection_filter": config["selection_filter"],
        "time_origin_epoch": config["capture_start_epoch"],
        "length_definition": "Original IPv4 total length, including headers, in bytes",
        "frame_number_definition": "Frame number in the filtered media capture",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted {count:,} packets to {destination}")


if __name__ == "__main__":
    main()
