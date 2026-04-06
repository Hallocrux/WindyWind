from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
import zipfile
from pathlib import Path


CODE_DATASET_SLUG = "{{CODE_DATASET_SLUG}}"
DATA_DATASET_SLUG = "{{DATA_DATASET_SLUG}}"
WORKSPACE_NAME = "windywind"


def resolve_input_root() -> Path:
    return Path(os.environ.get("WW_KAGGLE_INPUT_ROOT", "/kaggle/input")).resolve()


def resolve_work_root() -> Path:
    return Path(os.environ.get("WW_KAGGLE_WORK_ROOT", "/kaggle/working")).resolve()


def status_path(workspace_root: Path) -> Path:
    return workspace_root / "status.json"


def write_status(workspace_root: Path, payload: dict[str, object]) -> None:
    path = status_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def extract_payload(dataset_root: Path, archive_name: str, workspace_root: Path) -> None:
    archive_path = dataset_root / archive_name
    extracted_dir = dataset_root / Path(archive_name).stem
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(workspace_root)
        return
    if extracted_dir.exists() and extracted_dir.is_dir():
        for child in sorted(extracted_dir.iterdir()):
            target = workspace_root / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)
        return
    raise FileNotFoundError(
        f"未找到 payload: archive={archive_path}, extracted_dir={extracted_dir}"
    )


def collect_tree_entries(root: Path, max_depth: int = 4, max_entries: int = 200) -> list[str]:
    if not root.exists():
        return []

    entries: list[str] = []

    def _walk(current: Path, depth: int) -> None:
        if len(entries) >= max_entries or depth > max_depth:
            return
        for child in sorted(current.iterdir(), key=lambda path: (not path.is_dir(), path.name)):
            if len(entries) >= max_entries:
                return
            rel = child.relative_to(root).as_posix()
            entries.append(f"{rel}/" if child.is_dir() else rel)
            if child.is_dir():
                _walk(child, depth + 1)

    _walk(root, 0)
    return entries


def find_archive_matches(input_root: Path, archive_name: str) -> list[str]:
    if not input_root.exists():
        return []
    return sorted(path.as_posix() for path in input_root.rglob(archive_name))


def resolve_dataset_root(input_root: Path, preferred_slug: str, archive_name: str) -> Path:
    preferred = input_root / preferred_slug
    if preferred.exists():
        return preferred

    archive_matches = sorted(input_root.rglob(archive_name))
    slug_matches = sorted(
        path for path in input_root.rglob(preferred_slug) if path.is_dir() and path.name == preferred_slug
    )
    if len(slug_matches) == 1:
        return slug_matches[0]

    extracted_dir_name = Path(archive_name).stem
    extracted_matches = sorted(
        path for path in input_root.rglob(extracted_dir_name) if path.is_dir() and path.name == extracted_dir_name
    )
    candidates = sorted({path.parent for path in archive_matches} | {path.parent for path in extracted_matches})
    slug_candidates = [
        path
        for path in candidates
        if preferred_slug == path.name or preferred_slug in path.parts
    ]
    if len(slug_candidates) == 1:
        return slug_candidates[0]
    if len(candidates) == 1:
        return candidates[0]

    available = sorted(path.name for path in input_root.iterdir()) if input_root.exists() else []
    raise FileNotFoundError(
        "未找到 dataset 挂载目录。"
        f"preferred={preferred_slug}, archive={archive_name}, available={available}, "
        f"archive_matches={[str(path) for path in archive_matches]}, "
        f"slug_matches={[str(path) for path in slug_matches]}, "
        f"extracted_matches={[str(path) for path in extracted_matches]}"
    )


def prepare_workspace() -> tuple[Path, Path, Path]:
    input_root = resolve_input_root()
    work_root = resolve_work_root()
    workspace_root = work_root / WORKSPACE_NAME
    workspace_root.mkdir(parents=True, exist_ok=True)

    code_dataset_root = resolve_dataset_root(input_root, CODE_DATASET_SLUG, "repo_payload.zip")
    data_dataset_root = resolve_dataset_root(input_root, DATA_DATASET_SLUG, "data_payload.zip")

    extract_payload(code_dataset_root, "repo_payload.zip", workspace_root)
    extract_payload(data_dataset_root, "data_payload.zip", workspace_root)
    return workspace_root, code_dataset_root, data_dataset_root


def main() -> None:
    work_root = resolve_work_root()
    workspace_root = work_root / WORKSPACE_NAME
    write_status(
        workspace_root,
        {
            "stage": "boot",
            "input_root": str(resolve_input_root()),
            "work_root": str(work_root),
            "code_dataset_slug": CODE_DATASET_SLUG,
            "data_dataset_slug": DATA_DATASET_SLUG,
            "python_version": sys.version,
            "input_root_entries": (
                sorted(path.name for path in resolve_input_root().iterdir())
                if resolve_input_root().exists()
                else []
            ),
            "input_root_tree": collect_tree_entries(resolve_input_root()),
            "repo_payload_matches": find_archive_matches(resolve_input_root(), "repo_payload.zip"),
            "data_payload_matches": find_archive_matches(resolve_input_root(), "data_payload.zip"),
        },
    )

    try:
        workspace_root, code_dataset_root, data_dataset_root = prepare_workspace()
        write_status(
            workspace_root,
            {
                "stage": "workspace_ready",
                "workspace_root": str(workspace_root),
                "code_dataset_root": str(code_dataset_root),
                "data_dataset_root": str(data_dataset_root),
                "workspace_entries": sorted(path.name for path in workspace_root.iterdir()),
            },
        )

        os.chdir(workspace_root)
        sys.path.insert(0, str(workspace_root))

        from main import main as run_pipeline

        run_pipeline()

        outputs_dir = workspace_root / "outputs"
        result_files = sorted(
            str(path.relative_to(workspace_root))
            for path in outputs_dir.rglob("*")
            if path.is_file()
        ) if outputs_dir.exists() else []
        write_status(
            workspace_root,
            {
                "stage": "completed",
                "workspace_root": str(workspace_root),
                "result_files": result_files,
            },
        )
    except Exception as exc:
        error_payload = {
            "stage": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "input_root": str(resolve_input_root()),
            "input_root_tree": collect_tree_entries(resolve_input_root()),
            "repo_payload_matches": find_archive_matches(resolve_input_root(), "repo_payload.zip"),
            "data_payload_matches": find_archive_matches(resolve_input_root(), "data_payload.zip"),
        }
        write_status(workspace_root, error_payload)
        raise


if __name__ == "__main__":
    main()
