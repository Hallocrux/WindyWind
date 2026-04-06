from __future__ import annotations

import argparse

from kaggle_publish_common import (
    add_publish_args,
    build_code_payload,
    detect_kaggle_username,
    ensure_dataset_created,
    make_run_dir,
    prepare_kernel_dir,
    run_publish_flow,
    write_dataset_metadata,
    write_json,
)

TRY_NAME = "015_patchtst_loco"
ENTRY_REL_PATH = "src/try/015_patchtst_loco/run_patchtst_loco.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="发布 PatchTST windywind 实验到 Kaggle。")
    add_publish_args(
        parser,
        default_code_slug="windywind-patchtst-code",
        default_kernel_slug="windywind-patchtst-runner",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    username = args.username or detect_kaggle_username()
    code_dataset_id = f"{username}/{args.code_dataset_slug}"
    kernel_id = f"{username}/{args.kernel_slug}"
    data_dataset_id = args.data_dataset_id or f"{username}/{args.data_dataset_slug}"
    run_dir = make_run_dir(TRY_NAME, "patchtst")

    include_paths = [
        "pyproject.toml",
        "src/__init__.py",
        "src/current",
        "src/try/009_phase1_feature_groups",
        "src/try/012_phase3_end_to_end_shortlist",
        "src/try/013_phase3_cnn_tcn_smoke",
        "src/try/015_patchtst_loco",
    ]
    code_dir = build_code_payload(run_dir, include_paths)
    write_dataset_metadata(
        code_dir,
        code_dataset_id,
        "windywind PatchTST code payload",
    )
    kernel_dir = prepare_kernel_dir(
        run_dir,
        code_dataset_id=code_dataset_id,
        data_dataset_id=data_dataset_id,
        kernel_id=kernel_id,
        kernel_title="windywind PatchTST runner",
        entry_rel_path=ENTRY_REL_PATH,
    )
    publish_spec_path = run_dir / "publish-spec.json"
    write_json(
        publish_spec_path,
        {
            "version": 1,
            "datasets": [
                {
                    "name": "code",
                    "publish_dir": str(code_dir),
                    "dataset_id": code_dataset_id,
                    "dataset_title": "windywind PatchTST code payload",
                    "version_notes": "PatchTST code payload update",
                    "license_name": "CC0-1.0",
                    "dir_mode": "skip",
                    "attach_to_kernel": True,
                }
            ],
            "kernel": {
                "kernel_dir": str(kernel_dir),
                "kernel_id": kernel_id,
                "kernel_title": "windywind PatchTST runner",
                "code_file": "run_from_datasets.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "extra_dataset_sources": [data_dataset_id],
            },
        },
    )
    if not args.skip_remote:
        ensure_dataset_created(code_dataset_id, code_dir, run_dir / "code_dataset_creation.log")
        run_publish_flow(
            publish_spec_path=publish_spec_path,
            accelerator=args.accelerator,
            runtime_seconds=args.runtime_seconds,
            log_path=run_dir / "publish.log",
        )
    print(f"PatchTST Kaggle 发布目录已准备：{run_dir}")
    print(f"code_dataset_id: {code_dataset_id}")
    print(f"data_dataset_id: {data_dataset_id}")
    print(f"kernel_id: {kernel_id}")


if __name__ == "__main__":
    main()
