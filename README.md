# YouTube traffic source modeling

Packet data and reproducible analysis for Network Modeling Homework 1.

## Experiment

A six-minute recording of prerecorded [Big Buck Bunny](https://www.youtube.com/watch?v=aqz-KE-bpKQ) playback on YouTube, with 1080p60 selected, normal speed, and shared 5 GHz Wi-Fi. Incoming media-server packets were selected using browser response observations. Audio and video downloads are included; muting varied during recording. Packet lengths are original IP lengths, including headers, and retransmissions are retained.

The first minute is excluded. The next 210 seconds fit the models; the final 90 seconds validate them.

| Interval | Packets | IP bytes | Mean rate (Mbit/s) |
| --- | ---: | ---: | ---: |
| Fitting | 32,768 | 47,399,900 | 1.805710 |
| Validation | 17,165 | 24,862,043 | 2.209959 |

## Files

- `data/raw/youtube-media.pcap`: Wireshark-readable capture of the 66,836 selected packets, including the initial minute. Recorded timestamps and saved packet bytes are preserved.
- `data/processed/`: extracted packet times, lengths, and extraction manifest.
- `config/experiment.json`: experiment settings, fitting/validation intervals, seed, and model configuration.
- `data/playback-controls.csv`, `data/endpoint-attribution.json`, and `data/provenance.json`: playback observations, media-endpoint evidence, and capture hashes.
- `data/results/`: saved numerical results and arrays for reproducing the figures.
- `scripts/`: packet extraction, model fitting/simulation, and plotting.

## Run

Use Python 3.12 and install the dependencies:

```sh
python -m pip install -r requirements.txt
```

Reproduce both figures directly from the saved results:

```sh
python scripts/make_figures.py
```

Outputs are written to `figure/`: the measured traffic profile and packet-length CDFs, followed by the validation/model gap CDFs and 100 ms rate tails. Each figure is saved as PDF and PNG, with numerical curve data and provenance. Synthetic examples use replicate zero for each model.

To rerun fitting and all 1,000 simulations per model before plotting:

```sh
python scripts/analyze_trace.py
python scripts/make_figures.py
```

To regenerate the processed data from the capture, install Wireshark/TShark and run:

```sh
python scripts/extract_trace.py
```

TShark is found on `PATH` or in its standard Windows installation directory; `--tshark` accepts a custom executable path.

## Models and scope

Each received packet is one event. All models independently sample fitting packet lengths, assuming independence from timing. The timing models are Poisson, empirical renewal, and ON-OFF. ON-OFF durations use nine complete fitting cycles. Simulations use seed 20261020, model offsets of 0, 10000, and 20000, and random initial phases. Reported standard deviations describe variation across simulations with the fitted distributions held fixed.

This is one receiver-observed session over shared Wi-Fi. Playback checks are sampled observations. TShark flags 21 retransmissions in the incoming-only capture, compared with 10 in the original recording; these state-dependent flags are unused in fitting and are not loss counts. See `data/provenance.json` for capture details.
