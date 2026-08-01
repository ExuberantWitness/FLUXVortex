"""Compare a claim-runtime verification sweep with its frozen baseline."""
import argparse
import json
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--tolerance", type=float, default=0.4)
    args = parser.parse_args()

    with open(args.baseline) as stream:
        baseline = json.load(stream)
    with open(args.candidate) as stream:
        candidate = json.load(stream)

    missing = sorted(set(baseline) - set(candidate))
    extra = sorted(set(candidate) - set(baseline))
    deltas = []
    for key in sorted(set(baseline) & set(candidate)):
        if "L" not in baseline[key] or "L" not in candidate[key]:
            continue
        for channel in ("L", "T"):
            delta = candidate[key][channel] - baseline[key][channel]
            if not math.isfinite(delta):
                raise SystemExit(f"FAIL: non-finite delta at {key}/{channel}")
            deltas.append((abs(delta), key, channel, delta))
    deltas.sort(reverse=True)
    worst = deltas[0] if deltas else (float("inf"), "-", "-", float("inf"))
    print(f"coverage={len(candidate)}/{len(baseline)} missing={len(missing)} extra={len(extra)}")
    print(
        f"worst={worst[1]}/{worst[2]} delta={worst[3]:+.6f} N "
        f"abs={worst[0]:.6f} tolerance={args.tolerance:.3f}"
    )
    for row in deltas[:10]:
        print(f"  {row[1]:18s} {row[2]} {row[3]:+.6f} N")
    if missing or extra or worst[0] > args.tolerance:
        raise SystemExit("FAIL: sweep compatibility gate")
    print("PASS: sweep compatibility gate")


if __name__ == "__main__":
    main()
