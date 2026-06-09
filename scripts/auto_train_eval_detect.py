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
    proc = subprocess.run([sys.executable, *cmd], cwd=ROOT, text=True, capture_output=True, timeout=7200)
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded detect train/eval loop.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--target-score", type=float, default=0.70)
    parser.add_argument("--max-minutes", type=float, default=90)
    parser.add_argument("--output", default="models/detect/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rounds = max(1, min(args.rounds, 3))
    deadline = time.time() + args.max_minutes * 60
    report = {"rounds": [], "best_score": 0.0, "best_model": None, "stopped_reason": "max_rounds"}
    configs = [(1, 320, None), (2, 416, None), (3, 416, 0.5)]
    Path("runs/eval").mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        report["stopped_reason"] = "dry-run"
        Path("runs/eval/detect_auto_train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(report)
        return 0
    for idx in range(rounds):
        if time.time() > deadline:
            report["stopped_reason"] = "max_minutes"
            break
        epochs, imgsz, fraction = configs[idx]
        name = f"auto_round{idx + 1}"
        cmd = ["scripts/train_detect_yolo.py", "--model", args.base_model, "--data", args.data, "--epochs", str(epochs), "--imgsz", str(imgsz), "--batch", "2", "--device", args.device, "--project", "runs/detect", "--name", name, "--export", f"runs/detect/{name}_best.pt", "--allow-long-run"]
        if fraction:
            cmd.extend(["--fraction", str(fraction)])
        code, log = run(cmd)
        candidate = Path(f"runs/detect/{name}_best.pt")
        metrics_path = Path(f"runs/eval/detect_round{idx + 1}_metrics.json")
        if code == 0:
            _, eval_log = run(["scripts/evaluate_detect.py", "--model", str(candidate), "--data", args.data, "--output", str(metrics_path), "--imgsz", str(imgsz), "--device", args.device])
            log += eval_log
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {"proxy_center_hit_score": 0.0}
        score = float(metrics.get("proxy_center_hit_score", 0.0))
        report["rounds"].append({"round": idx + 1, "epochs": epochs, "imgsz": imgsz, "returncode": code, "score": score, "metrics_path": str(metrics_path), "log_tail": log[-2000:]})
        if candidate.exists() and score >= report["best_score"]:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, args.output)
            report["best_score"] = score
            report["best_model"] = args.output
        if score >= args.target_score:
            report["stopped_reason"] = "target_reached"
            break
    out = Path("runs/eval/detect_auto_train_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

