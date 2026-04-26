from __future__ import annotations

import argparse
import sys
from pathlib import Path

TRY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRY_ROOT.parents[2]
TRY015_ROOT = REPO_ROOT / "src" / "try" / "015_patchtst_loco"
if str(TRY015_ROOT) not in sys.path:
    sys.path.insert(0, str(TRY015_ROOT))

from kaggle_publish_common import (
    detect_kaggle_username,
    ensure_dataset_created,
    make_run_dir,
    run_publish_flow,
    write_dataset_metadata,
    write_json,
)

TRY_NAME = "067_reuse_embedding_domain_lodo_moe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="发布 067 domain-LODO MoE 到 Kaggle。")
    parser.add_argument("--username", default=None)
    parser.add_argument("--code-dataset-slug", default="windywind-domain-lodo-moe-code")
    parser.add_argument("--data-dataset-slug", default="windywind-domain-lodo-moe-data")
    parser.add_argument("--artifact-dataset-slug", default="windywind-domain-lodo-moe-artifacts")
    parser.add_argument("--kernel-slug", default="windywind-domain-lodo-moe-gpu-run")
    parser.add_argument("--accelerator", default="P100")
    parser.add_argument("--runtime-seconds", type=int, default=14400)
    parser.add_argument("--skip-remote", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    username = args.username or detect_kaggle_username()
    code_dataset_id = f"{username}/{args.code_dataset_slug}"
    data_dataset_id = f"{username}/{args.data_dataset_slug}"
    artifact_dataset_id = f"{username}/{args.artifact_dataset_slug}"
    kernel_id = f"{username}/{args.kernel_slug}"

    run_dir = make_run_dir(TRY_NAME, "domain_lodo")
    code_dir = stage_code_dataset(run_dir)
    data_dir = stage_data_dataset(run_dir)
    artifact_dir = stage_artifact_dataset(run_dir)
    kernel_dir = stage_kernel_dir(
        run_dir,
        kernel_id=kernel_id,
        kernel_title="windywind domain LODO MoE GPU run",
        dataset_sources=[code_dataset_id, data_dataset_id, artifact_dataset_id],
    )

    write_dataset_metadata(code_dir, code_dataset_id, "windywind domain LODO MoE code payload")
    write_dataset_metadata(data_dir, data_dataset_id, "windywind domain LODO MoE data payload")
    write_dataset_metadata(artifact_dir, artifact_dataset_id, "windywind domain LODO MoE artifact payload")

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
                    "dataset_title": "windywind domain LODO MoE code payload",
                    "version_notes": "update code payload for domain LODO MoE",
                    "license_name": "CC0-1.0",
                    "dir_mode": "auto",
                    "attach_to_kernel": True,
                },
                {
                    "name": "data",
                    "publish_dir": str(data_dir),
                    "dataset_id": data_dataset_id,
                    "dataset_title": "windywind domain LODO MoE data payload",
                    "version_notes": "update data payload for domain split and LODO MoE",
                    "license_name": "CC0-1.0",
                    "dir_mode": "auto",
                    "attach_to_kernel": True,
                },
                {
                    "name": "artifacts",
                    "publish_dir": str(artifact_dir),
                    "dataset_id": artifact_dataset_id,
                    "dataset_title": "windywind domain LODO MoE artifact payload",
                    "version_notes": "update embedding checkpoints for domain LODO MoE",
                    "license_name": "CC0-1.0",
                    "dir_mode": "auto",
                    "attach_to_kernel": True,
                },
            ],
            "kernel": {
                "kernel_dir": str(kernel_dir),
                "kernel_id": kernel_id,
                "kernel_title": "windywind domain LODO MoE GPU run",
                "code_file": "main.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": False,
                "extra_dataset_sources": [],
            },
        },
    )

    if not args.skip_remote:
        ensure_dataset_created(code_dataset_id, code_dir, run_dir / "code_dataset_create.log")
        ensure_dataset_created(data_dataset_id, data_dir, run_dir / "data_dataset_create.log")
        ensure_dataset_created(artifact_dataset_id, artifact_dir, run_dir / "artifact_dataset_create.log")
        run_publish_flow(
            publish_spec_path=publish_spec_path,
            accelerator=args.accelerator,
            runtime_seconds=args.runtime_seconds,
            log_path=run_dir / "publish.log",
        )

    print(f"067 Kaggle 发布目录已准备：{run_dir}")
    print(f"code_dataset_id: {code_dataset_id}")
    print(f"data_dataset_id: {data_dataset_id}")
    print(f"artifact_dataset_id: {artifact_dataset_id}")
    print(f"kernel_id: {kernel_id}")


def stage_code_dataset(run_dir: Path) -> Path:
    code_dir = run_dir / "code_dataset"
    copy_targets = [
        ("PROJECT.md", "PROJECT.md"),
        ("README.md", "README.md"),
        ("pyproject.toml", "pyproject.toml"),
        ("src/__init__.py", "src/__init__.py"),
        ("src/current", "src/current"),
        ("src/try/009_phase1_feature_groups", "src/try/009_phase1_feature_groups"),
        ("src/try/012_phase3_end_to_end_shortlist", "src/try/012_phase3_end_to_end_shortlist"),
        ("src/try/047_soft_gate_quickcheck", "src/try/047_soft_gate_quickcheck"),
        ("src/try/053_support_window_residual_quickcheck", "src/try/053_support_window_residual_quickcheck"),
        ("src/try/065_sparse_router_residual_moe_v1", "src/try/065_sparse_router_residual_moe_v1"),
        ("src/try/066_reuse_embedding_domain_split", "src/try/066_reuse_embedding_domain_split"),
        ("src/try/067_reuse_embedding_domain_lodo_moe", "src/try/067_reuse_embedding_domain_lodo_moe"),
        ("src/try/015_patchtst_loco/kaggle_publish_common.py", "src/try/015_patchtst_loco/kaggle_publish_common.py"),
    ]
    for source_rel, target_rel in copy_targets:
        copy_path(REPO_ROOT / source_rel, code_dir / target_rel)
    return code_dir


def stage_data_dataset(run_dir: Path) -> Path:
    data_dir = run_dir / "data_dataset"
    copy_targets = [
        ("data/final", "final"),
        ("data/added", "added"),
        ("data/added2", "added2"),
        ("outputs/try/066_reuse_embedding_domain_split/domain_assignment.csv", "try/066_reuse_embedding_domain_split/domain_assignment.csv"),
        ("outputs/try/066_reuse_embedding_domain_split/domain_summary.csv", "try/066_reuse_embedding_domain_split/domain_summary.csv"),
        ("outputs/try/066_reuse_embedding_domain_split/cluster_selection_report.csv", "try/066_reuse_embedding_domain_split/cluster_selection_report.csv"),
    ]
    for source_rel, target_rel in copy_targets:
        copy_path(REPO_ROOT / source_rel, data_dir / target_rel)
    return data_dir


def stage_artifact_dataset(run_dir: Path) -> Path:
    artifact_dir = run_dir / "artifact_dataset"
    copy_targets = [
        ("outputs/try/057_embedding_space_diagnosis/models/checkpoints", "try/057_embedding_space_diagnosis/models/checkpoints"),
    ]
    for source_rel, target_rel in copy_targets:
        copy_path(REPO_ROOT / source_rel, artifact_dir / target_rel)
    return artifact_dir


def stage_kernel_dir(run_dir: Path, *, kernel_id: str, kernel_title: str, dataset_sources: list[str]) -> Path:
    kernel_dir = run_dir / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "main.py").write_text(render_kernel_main(), encoding="utf-8")
    write_json(
        kernel_dir / "kernel-metadata.json",
        {
            "id": kernel_id,
            "title": kernel_title,
            "code_file": "main.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": False,
            "dataset_sources": dataset_sources,
        },
    )
    return kernel_dir


def copy_path(source: Path, target: Path) -> None:
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for child in source.rglob("*"):
            rel = child.relative_to(source)
            dest = target / rel
            if child.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(child.read_bytes())
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


def render_kernel_main() -> str:
    return """from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


WORKDIR = Path("/kaggle/working/windywind")


def print_input_tree() -> None:
    print("Mounted /kaggle/input entries:")
    for path in sorted(Path("/kaggle/input").glob("*")):
        print(f"- {path}")
    datasets_root = Path("/kaggle/input/datasets")
    if datasets_root.exists():
        print("\\nMounted /kaggle/input/datasets entries:")
        for path in sorted(datasets_root.rglob("*")):
            if len(path.relative_to(datasets_root).parts) <= 4:
                print(f"- {path}")


def overlay_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing mounted dataset path: {source}")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def find_root_by_relative_path(relative_path: str) -> Path:
    relative = Path(relative_path)
    candidates: list[Path] = []
    for matched in Path("/kaggle/input").rglob(relative.name):
        try:
            rel_parts = matched.relative_to(Path("/kaggle/input")).parts
        except ValueError:
            continue
        if len(rel_parts) < len(relative.parts):
            continue
        if rel_parts[-len(relative.parts):] != relative.parts:
            continue
        root = matched
        for _ in relative.parts:
            root = root.parent
        candidates.append(root)
    if not candidates:
        raise FileNotFoundError(f"Could not locate mounted payload for relative path: {relative_path}")
    candidates = sorted(set(candidates), key=lambda path: len(path.parts))
    return candidates[0]


def find_root_by_relative_path_candidates(relative_paths: list[str]) -> Path:
    errors: list[str] = []
    for relative_path in relative_paths:
        try:
            return find_root_by_relative_path(relative_path)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    raise FileNotFoundError("\\n".join(errors))


def overlay_named_child(root: Path, child_name: str, destination: Path) -> None:
    source = root / child_name
    if not source.exists():
        raise FileNotFoundError(f"Missing mounted child path: {source}")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def main() -> None:
    print_input_tree()
    WORKDIR.mkdir(parents=True, exist_ok=True)

    code_root = find_root_by_relative_path_candidates([
        "src/try/067_reuse_embedding_domain_lodo_moe/run_reuse_embedding_domain_lodo_moe.py",
    ])
    data_root = find_root_by_relative_path_candidates([
        "try/066_reuse_embedding_domain_split/domain_assignment.csv",
        "066_reuse_embedding_domain_split/domain_assignment.csv",
        "domain_assignment.csv",
    ])
    artifact_root = find_root_by_relative_path_candidates([
        "try/057_embedding_space_diagnosis/models/checkpoints/unified_all_2s.pt",
        "057_embedding_space_diagnosis/models/checkpoints/unified_all_2s.pt",
        "models/checkpoints/unified_all_2s.pt",
        "unified_all_2s.pt",
    ])
    print(f"Resolved code root: {code_root}")
    print(f"Resolved data root: {data_root}")
    print(f"Resolved artifact root: {artifact_root}")

    overlay_tree(code_root, WORKDIR)
    overlay_named_child(data_root, "final", WORKDIR / "data" / "final")
    overlay_named_child(data_root, "added", WORKDIR / "data" / "added")
    overlay_named_child(data_root, "added2", WORKDIR / "data" / "added2")
    if (data_root / "try").exists():
        overlay_named_child(data_root, "try", WORKDIR / "outputs" / "try")
    elif (data_root / "066_reuse_embedding_domain_split").exists():
        overlay_named_child(data_root, "066_reuse_embedding_domain_split", WORKDIR / "outputs" / "try" / "066_reuse_embedding_domain_split")
    else:
        raise FileNotFoundError("Could not locate domain split payload under data dataset root")

    if (artifact_root / "try").exists():
        overlay_named_child(artifact_root, "try", WORKDIR / "outputs" / "try")
    elif (artifact_root / "057_embedding_space_diagnosis").exists():
        overlay_named_child(artifact_root, "057_embedding_space_diagnosis", WORKDIR / "outputs" / "try" / "057_embedding_space_diagnosis")
    elif (artifact_root / "models").exists():
        overlay_tree(artifact_root, WORKDIR / "outputs" / "try" / "057_embedding_space_diagnosis")
    else:
        raise FileNotFoundError("Could not locate embedding checkpoint payload under artifact dataset root")

    os.chdir(WORKDIR)
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        "python",
        "src/try/067_reuse_embedding_domain_lodo_moe/run_reuse_embedding_domain_lodo_moe.py",
        "--mode",
        "full",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    for relative_path in [
        "outputs/try/067_reuse_embedding_domain_lodo_moe/summary.md",
        "outputs/try/067_reuse_embedding_domain_lodo_moe/summary_by_learned_domain.csv",
        "outputs/try/067_reuse_embedding_domain_lodo_moe/summary_by_raw_source.csv",
    ]:
        path = WORKDIR / relative_path
        if path.exists():
            print(f"\\n=== {relative_path} ===")
            print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
"""


if __name__ == "__main__":
    main()
