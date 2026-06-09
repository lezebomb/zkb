from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, *cmd], cwd=ROOT, text=True, capture_output=True, timeout=3600)
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded classify train/eval loop.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--target-acc", type=float, default=0.75)
    parser.add_argument("--max-minutes", type=float, default=60)
    parser.add_argument("--output", default="models/classify/classifier.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rounds = max(1, min(args.rounds, 3))
    deadline = time.time() + args.max_minutes * 60
    report = {"rounds": [], "best_accuracy": 0.0, "best_model": None, "stopped_reason": "max_rounds"}
    configs = [("mobilenet_v3_small", 1), ("resnet18", 1), ("mobilenet_v3_small", 2)]
    Path("runs/eval").mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        report["stopped_reason"] = "dry-run"
        Path("runs/eval/classify_auto_train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(report)
        return 0
    for idx in range(rounds):
        if time.time() > deadline:
            report["stopped_reason"] = "max_minutes"
            break
        model_name, epochs = configs[idx]
        candidate = Path(f"runs/classify/round{idx + 1}/classifier.pt")
        code, log = run(["scripts/train_classify.py", "--data", args.data, "--model", model_name, "--epochs", str(epochs), "--batch", "4", "--device", args.device, "--output", str(candidate), "--allow-long-run"])
        metrics_path = Path(f"runs/eval/classify_round{idx + 1}_metrics.json")
        if code == 0:
            code_eval, eval_log = run(["scripts/evaluate_classify.py", "--model", str(candidate), "--data", str(Path(args.data) / "val"), "--output", str(metrics_path)])
            log += eval_log
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {"accuracy": 0.0}
        acc = float(metrics.get("accuracy", 0.0))
        report["rounds"].append({"round": idx + 1, "model": model_name, "epochs": epochs, "returncode": code, "accuracy": acc, "metrics_path": str(metrics_path), "log_tail": log[-2000:]})
        if candidate.exists() and acc >= report["best_accuracy"]:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, args.output)
            class_names = candidate.parent / "class_names.json"
            if class_names.exists():
                shutil.copy2(class_names, Path(args.output).parent / "class_names.json")
            report["best_accuracy"] = acc
            report["best_model"] = args.output
        if acc >= args.target_acc:
            report["stopped_reason"] = "target_reached"
            break
    out = Path("runs/eval/classify_auto_train_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

