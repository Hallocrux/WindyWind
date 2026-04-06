from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_publish_common import (
    build_data_payload,
    detect_kaggle_username,
    ensure_dataset_created,
    make_run_dir,
    run_publish_flow,
    write_json,
)

TRY_NAME = "015_patchtst_loco"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="发布 windywind 共享数据 dataset 到 Kaggle。")
    parser.add_argument("--username", default=None, help="Kaggle 用户名；默认自动解析。")
    parser.add_argument("--data-dataset-slug", default="windywind-phase3-sequence-data")
    parser.add_argument("--skip-remote", action="store_true", help="只生成 payload 和 publish-spec。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    username = args.username or detect_kaggle_username()
    dataset_id = f"{username}/{args.data_dataset_slug}"
    run_dir = make_run_dir(TRY_NAME, "shared_data")
    data_dir = build_data_payload(run_dir)
    publish_spec_path = run_dir / "publish-spec.json"
    write_json(
        publish_spec_path,
        {
            "version": 1,
            "datasets": [
                {
                    "name": "data",
                    "publish_dir": str(data_dir),
                    "dataset_id": dataset_id,
                    "dataset_title": "windywind phase3 shared data payload",
                    "version_notes": "phase3 shared data payload update",
                    "license_name": "CC0-1.0",
                    "dir_mode": "skip",
                    "attach_to_kernel": False,
                }
            ]
        },
    )
    if not args.skip_remote:
        bootstrap_log = run_dir / "data_dataset_creation.log"
        ensure_dataset_created(dataset_id, data_dir, bootstrap_log)
        run_publish_flow(
            publish_spec_path=publish_spec_path,
            accelerator="P100",
            runtime_seconds=14400,
            log_path=run_dir / "publish.log",
        )
    print(f"共享数据 dataset 已准备：{dataset_id}")
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
