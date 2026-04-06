from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import run_try


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="仅更新并推送 Kaggle thin kernel，复用已有 code/data datasets。"
    )
    parser.add_argument("--username", default=None, help="Kaggle 用户名；默认自动解析。")
    parser.add_argument(
        "--code-dataset-slug",
        default="windywind-try-008-current-code",
        help="已有代码 dataset slug。",
    )
    parser.add_argument(
        "--data-dataset-slug",
        default="windywind-try-008-current-data",
        help="已有数据 dataset slug。",
    )
    parser.add_argument(
        "--kernel-slug",
        default="windywind-try-008-current-thin-kernel",
        help="要更新的 kernel slug。",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="等待 kernel 进入终态的最长秒数。",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=20,
        help="轮询 kernel 状态的间隔秒数。",
    )
    return parser.parse_args()


def main() -> None:
    run_try.ensure_command_exists("kaggle")
    args = parse_args()
    username = args.username or run_try.detect_kaggle_username()

    run_dir = run_try.OUTPUT_ROOT / "kernel_only_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    log_dir = run_dir / "command_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    code_dataset_id = f"{username}/{args.code_dataset_slug}"
    data_dataset_id = f"{username}/{args.data_dataset_slug}"
    kernel_id = f"{username}/{args.kernel_slug}"

    kernel_dir = run_dir / "kernel"
    run_try.prepare_kernel_dir(
        kernel_dir=kernel_dir,
        code_dataset_id=code_dataset_id,
        data_dataset_id=data_dataset_id,
        kernel_id=kernel_id,
    )

    run_try.run_command(
        ["kaggle", "kernels", "push", "-p", str(kernel_dir), "-t", "3600"],
        log_dir=log_dir,
        check=True,
    )

    status = run_try.wait_for_kernel(
        kernel_id=kernel_id,
        log_dir=log_dir,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )

    remote_outputs_dir = run_dir / "remote_outputs"
    output_download = run_try.run_command(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(remote_outputs_dir), "-o"],
        log_dir=log_dir,
        check=False,
    )

    summary: dict[str, object] = {
        "try_name": run_try.TRY_NAME,
        "mode": "kernel_only",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "username": username,
        "code_dataset_id": code_dataset_id,
        "data_dataset_id": data_dataset_id,
        "kernel_id": kernel_id,
        "run_dir": str(run_dir),
        "kernel_status": status,
        "output_download_returncode": output_download.returncode,
        "remote_outputs_dir": str(remote_outputs_dir) if remote_outputs_dir.exists() else None,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    run_try.write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
