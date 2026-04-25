#!/usr/bin/env python3
"""Run downstream correlation experiments for successful Optuna trials."""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import collect_config  # noqa: E402
from ebes.pipeline.base_runner import Runner  # noqa: E402


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrialRun:
    trial_id: int
    trial_dir: Path
    params_path: Path
    ckpt_path: Path | None
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run downstream validation for the first successful Optuna trials "
            "of one dataset/method pair."
        )
    )
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-m", "--method", required=True)
    parser.add_argument("-g", "--gpu", default="cuda:0")
    parser.add_argument(
        "-dv",
        "--downstream-validator",
        default="universal_validator/configs/validator/all.yaml",
    )
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--start-trial", type=int, default=0)
    parser.add_argument("--seed-dir", default="seed_0")
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing correlation trial directories before rerunning.",
    )
    return parser.parse_args()


def numeric_trial_dirs(optuna_dir: Path) -> list[Path]:
    return sorted(
        (p for p in optuna_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )


def find_ckpt(trial_dir: Path, seed_dir: str) -> Path | None:
    ckpt_dir = trial_dir / seed_dir / "ckpt"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def select_trials(
    optuna_dir: Path,
    seed_dir: str,
    start_trial: int,
    num_trials: int,
) -> list[TrialRun]:
    selected = []
    for trial_dir in numeric_trial_dirs(optuna_dir):
        trial_id = int(trial_dir.name)
        if trial_id < start_trial:
            continue
        params_path = trial_dir / "params.txt"
        if not (trial_dir / "results.csv").exists():
            logger.info("skip trial %s: no optuna results.csv", trial_id)
            continue
        if not params_path.exists():
            logger.info("skip trial %s: no params.txt", trial_id)
            continue
        ckpt_path = find_ckpt(trial_dir, seed_dir)
        selected.append(
            TrialRun(
                trial_id=trial_id,
                trial_dir=trial_dir,
                params_path=params_path,
                ckpt_path=ckpt_path,
                mode="inference" if ckpt_path else "test",
            )
        )
        if len(selected) == num_trials:
            break
    return selected


def load_trial_params(params_path: Path) -> dict[str, Any]:
    with params_path.open() as f:
        params = yaml.safe_load(f) or {}
    if not isinstance(params, dict):
        raise ValueError(f"Expected mapping in {params_path}")
    return params


def set_downstream_device(config, device: str) -> None:
    downstream_config = config.get("universal_validator")
    if not downstream_config:
        return
    for model_config in downstream_config.get("models", {}).values():
        shared_params = model_config.get("shared_params", {})
        if "device" in shared_params:
            shared_params["device"] = device


def build_config(args: argparse.Namespace, trial: TrialRun):
    config = collect_config(
        dataset=args.dataset,
        method=args.method,
        experiment="inference" if trial.ckpt_path else "test",
        specify=None,
        gpu=args.gpu,
        downstream_validator=args.downstream_validator,
    )
    trial_params = OmegaConf.create(load_trial_params(trial.params_path))
    config = OmegaConf.merge(config, trial_params)

    config["run_name"] = f"{args.method}/correlation_exp/trial_{trial.trial_id}"
    config["runner"]["params"]["n_runs"] = args.n_runs
    set_downstream_device(config, args.gpu)

    if trial.ckpt_path:
        config["trainer"]["total_iters"] = 0
        config["trainer"]["ckpt_resume"] = str(trial.ckpt_path)
    return config


def write_summary_row(summary_path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "trial",
        "status",
        "mode",
        "ckpt",
        "run_dir",
        "results_csv",
        "params",
        "message",
    ]
    exists = summary_path.exists()
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def run_trial(args: argparse.Namespace, trial: TrialRun, summary_path: Path) -> bool:
    run_dir = ROOT / "log" / args.dataset / args.method / "correlation_exp" / (
        f"trial_{trial.trial_id}"
    )
    results_csv = run_dir / "results.csv"

    if results_csv.exists() and not args.force:
        write_summary_row(
            summary_path,
            {
                "trial": trial.trial_id,
                "status": "skipped",
                "mode": trial.mode,
                "ckpt": trial.ckpt_path or "",
                "run_dir": run_dir,
                "results_csv": results_csv,
                "params": trial.params_path,
                "message": "results.csv already exists",
            },
        )
        print(f"[SKIP] trial={trial.trial_id} results already exist: {results_csv}")
        return True

    if run_dir.exists():
        if not args.force:
            write_summary_row(
                summary_path,
                {
                    "trial": trial.trial_id,
                    "status": "skipped",
                    "mode": trial.mode,
                    "ckpt": trial.ckpt_path or "",
                    "run_dir": run_dir,
                    "results_csv": "",
                    "params": trial.params_path,
                    "message": "run directory exists without results.csv; use --force",
                },
            )
            print(
                f"[SKIP] trial={trial.trial_id} has incomplete existing run dir: "
                f"{run_dir}"
            )
            return True
        shutil.rmtree(run_dir)

    if args.dry_run:
        write_summary_row(
            summary_path,
            {
                "trial": trial.trial_id,
                "status": "selected",
                "mode": trial.mode,
                "ckpt": trial.ckpt_path or "",
                "run_dir": run_dir,
                "results_csv": results_csv,
                "params": trial.params_path,
                "message": "dry run",
            },
        )
        print(
            f"[DRY] trial={trial.trial_id} mode={trial.mode} "
            f"ckpt={trial.ckpt_path or '-'}"
        )
        return True

    print(f"[RUN] trial={trial.trial_id} mode={trial.mode}")
    if trial.ckpt_path is None:
        print(f"[RUN] trial={trial.trial_id} ckpt missing; training from scratch")
    else:
        print(f"[RUN] trial={trial.trial_id} ckpt={trial.ckpt_path}")

    try:
        config = build_config(args, trial)
        runner = Runner.get_runner(config["runner"]["name"])
        runner.run(config)
    except Exception as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        error_path = run_dir / "ERROR.txt"
        error_path.write_text(traceback.format_exc())
        write_summary_row(
            summary_path,
            {
                "trial": trial.trial_id,
                "status": "failed",
                "mode": trial.mode,
                "ckpt": trial.ckpt_path or "",
                "run_dir": run_dir,
                "results_csv": results_csv if results_csv.exists() else "",
                "params": trial.params_path,
                "message": repr(exc),
            },
        )
        print(f"[FAIL] trial={trial.trial_id}: {exc}")
        return False

    status = "inference" if trial.ckpt_path else "test"
    write_summary_row(
        summary_path,
        {
            "trial": trial.trial_id,
            "status": status,
            "mode": trial.mode,
            "ckpt": trial.ckpt_path or "",
            "run_dir": run_dir,
            "results_csv": results_csv if results_csv.exists() else "",
            "params": trial.params_path,
            "message": "ok" if results_csv.exists() else "results.csv missing",
        },
    )
    if not results_csv.exists():
        print(f"[WARN] trial={trial.trial_id} finished but results.csv is missing")
    else:
        print(f"[OK] trial={trial.trial_id} results={results_csv}")
    return results_csv.exists()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    optuna_dir = ROOT / "log" / args.dataset / args.method / "optuna"
    if not optuna_dir.exists():
        print(f"[ERROR] Optuna directory not found: {optuna_dir}")
        return 1

    selected = select_trials(
        optuna_dir=optuna_dir,
        seed_dir=args.seed_dir,
        start_trial=args.start_trial,
        num_trials=args.num_trials,
    )
    if len(selected) < args.num_trials:
        print(
            f"[ERROR] Found only {len(selected)} successful trials, "
            f"requested {args.num_trials}"
        )
        return 1

    output_dir = ROOT / "log" / args.dataset / args.method / "correlation_exp"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    if args.force and summary_path.exists():
        summary_path.unlink()

    ok = True
    for trial in selected:
        ok = run_trial(args, trial, summary_path) and ok
    print(f"[DONE] summary={summary_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
