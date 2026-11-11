"""Train an exoplanet random forest from the command line."""

import argparse
from pathlib import Path

from exoplanet import MAX_ROWS, default_features, feature_columns, model_bytes, read_csv, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to your catalog")
    parser.add_argument("--target", default="pl_bmassj")
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--features", nargs="+", help="Numeric input columns; omit to use defaults")
    parser.add_argument("--output", type=Path, default=Path("exoplanet_model.joblib"))
    parser.add_argument("--rows", type=int, default=MAX_ROWS)
    args = parser.parse_args()
    if not 1 <= args.rows <= MAX_ROWS:
        parser.error(f"--rows must be between 1 and {MAX_ROWS:,}")
    try:
        frame, truncated = read_csv(args.csv, args.rows)
        features = args.features or default_features(feature_columns(frame, args.target))
        artifact, report = train(frame, features, args.target, args.task)
        payload = model_bytes(artifact)
        with args.output.open("xb") as output:
            output.write(payload)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    if truncated:
        print(f"Used the first {len(frame):,} rows.")
    print(f"Target: {args.target} | Features: {', '.join(features)}")
    print(f"Train: {report['train_rows']:,} | Test: {report['test_rows']:,} | "
          f"Unusable targets: {report['unlabeled_rows']:,} | Duplicates: {report['duplicate_rows']:,}")
    for key, value in report["metrics"].items():
        key = "R2" if key == "R²" else key
        print(f"{key}: {value:.4f}" if value is not None else f"{key}: unavailable")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
