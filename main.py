import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ann
import knn
import linear_regression
import logistic_regression
import svm


SUPPORTED_COMBINATIONS = {
    "linear": {"house"},
    "knn": {"titanic", "mnist", "cifar10"},
    "logistic": {"titanic", "mnist", "cifar10"},
    "svm": {"titanic", "mnist", "cifar10"},
    "ann": {"house", "titanic", "cifar10"},
}

HANDLERS = {
    "linear": linear_regression,
    "knn": knn,
    "logistic": logistic_regression,
    "svm": svm,
    "ann": ann,
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for the ML_Project course experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algo", required=True, choices=sorted(SUPPORTED_COMBINATIONS))
    parser.add_argument("--data", required=True, choices=["house", "titanic", "mnist", "cifar10"])
    parser.add_argument("--process", required=True, choices=["train", "test", "predict"])
    parser.add_argument("--model", help="Optional model file path.")
    parser.add_argument("--result", help="Optional result directory path.")
    parser.add_argument("--class-a", dest="class_a", help="Binary class A for HOG + logistic/SVM tasks.")
    parser.add_argument("--class-b", dest="class_b", help="Binary class B for HOG + logistic/SVM tasks.")
    parser.add_argument("--max-train", type=int, help="Optional training sample cap for image datasets.")
    parser.add_argument("--max-test", type=int, help="Optional test sample cap for image datasets.")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for ANN image training.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for ANN image training.")
    parser.add_argument("--k", type=int, help="Optional K value for KNN.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.data not in SUPPORTED_COMBINATIONS[args.algo]:
        supported = ", ".join(sorted(SUPPORTED_COMBINATIONS[args.algo]))
        print(
            f"Combination not implemented yet: algo={args.algo}, data={args.data}. "
            f"Supported datasets for {args.algo}: {supported}."
        )
        return

    handler = HANDLERS[args.algo]

    try:
        handler.run(args.process, args)
    except FileNotFoundError as exc:
        print(f"Missing required file: {exc}")
    except NotImplementedError as exc:
        print(f"Friendly reminder: {exc}")
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(f"Execution failed for {args.algo}/{args.data}/{args.process}: {exc}")
        raise


if __name__ == "__main__":
    main()
