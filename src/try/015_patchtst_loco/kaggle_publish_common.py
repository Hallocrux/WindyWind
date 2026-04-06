from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_RUN_PUBLISH_FLOW = Path(r"C:\Users\formy\.codex\skills\kaggle-ops\scripts\run_publish_flow.py")
OUTPUT_ROOT = REPO_ROOT / "outputs"


def detect_kaggle_username() -> str:
    env_username = os.environ.get("KAGGLE_USERNAME")
    if env_username:
        return env_username.strip()
    result = subprocess.run(
        ["kaggle", "config", "view"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"无法读取 kaggle config view: {result.stderr or result.stdout}")
    match = re.search(r"^\s*-\s*username:\s*(\S+)\s*$", result.stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("未从 kaggle config view 解析 username")
    return match.group(1)


def make_run_dir(try_name: str, suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / "try" / try_name / "kaggle_runs" / f"{timestamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_dataset_metadata(publish_dir: Path, dataset_id: str, title: str) -> None:
    write_json(
        publish_dir / "dataset-metadata.json",
        {
            "title": title,
            "id": dataset_id,
            "licenses": [{"name": "CC0-1.0"}],
        },
    )


def create_zip(archive_path: Path, entries: list[tuple[Path, Path]]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source, target in entries:
            if source.is_dir():
                for child in sorted(source.rglob("*")):
                    if child.is_dir():
                        continue
                    zf.write(child, (target / child.relative_to(source)).as_posix())
            else:
                zf.write(source, target.as_posix())


def build_code_payload(run_dir: Path, include_rel: list[str]) -> Path:
    code_dir = run_dir / "publish_payloads" / "code_dataset"
    code_dir.mkdir(parents=True, exist_ok=True)
    create_zip(
        code_dir / "repo_payload.zip",
        [(REPO_ROOT / rel, Path(rel)) for rel in include_rel],
    )
    write_json(
        code_dir / "payload_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "payload": "repo_payload.zip",
            "entries": include_rel,
        },
    )
    return code_dir


def build_data_payload(run_dir: Path) -> Path:
    data_dir = run_dir / "publish_payloads" / "data_dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    create_zip(
        data_dir / "data_payload.zip",
        [
            (REPO_ROOT / "data" / "final" / "dataset_manifest.csv", Path("data") / "final" / "dataset_manifest.csv"),
            (REPO_ROOT / "data" / "final" / "datasets", Path("data") / "final" / "datasets"),
        ],
    )
    write_json(
        data_dir / "payload_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "payload": "data_payload.zip",
            "entries": ["data/final/dataset_manifest.csv", "data/final/datasets/"],
        },
    )
    return data_dir


def render_kernel_entry(*, code_dataset_slug: str, data_dataset_slug: str, entry_rel_path: str) -> str:
    return f"""from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

CODE_DATASET_SLUG = "{code_dataset_slug}"
DATA_DATASET_SLUG = "{data_dataset_slug}"
ENTRY_REL_PATH = "{entry_rel_path}"
WORKSPACE_NAME = "windywind"
TORCH_INSTALL_COMMAND = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--quiet",
    "--no-input",
    "--upgrade",
    "torch==2.3.1",
    "torchvision==0.18.1",
    "torchaudio==2.3.1",
    "--index-url",
    "https://download.pytorch.org/whl/cu121",
]


def write_status(workspace_root: Path, payload: dict[str, object]) -> None:
    (workspace_root / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\\n",
        encoding="utf-8",
    )


def resolve_input_root() -> Path:
    return Path(os.environ.get("WW_KAGGLE_INPUT_ROOT", "/kaggle/input")).resolve()


def resolve_work_root() -> Path:
    return Path(os.environ.get("WW_KAGGLE_WORK_ROOT", "/kaggle/working")).resolve()


def resolve_dataset_root(input_root: Path, preferred_slug: str, archive_name: str) -> Path:
    direct = input_root / preferred_slug
    if direct.exists():
        return direct
    archive_matches = sorted(input_root.rglob(archive_name))
    slug_matches = sorted(
        path for path in input_root.rglob(preferred_slug) if path.is_dir() and path.name == preferred_slug
    )
    if len(slug_matches) == 1:
        return slug_matches[0]
    if len(archive_matches) == 1:
        return archive_matches[0].parent
    raise FileNotFoundError(
        f"未找到 dataset 根目录: preferred={{preferred_slug}}, archive={{archive_name}}, archive_matches={{[str(p) for p in archive_matches]}}"
    )


def extract_payload(dataset_root: Path, archive_name: str, workspace_root: Path) -> None:
    archive_path = dataset_root / archive_name
    extracted_dir = dataset_root / Path(archive_name).stem
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(workspace_root)
        return
    if extracted_dir.exists():
        for child in sorted(extracted_dir.rglob("*")):
            rel = child.relative_to(extracted_dir)
            target = workspace_root / rel
            if child.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, target)
        return
    raise FileNotFoundError(f"未找到 payload: {{archive_path}} / {{extracted_dir}}")


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )


def inspect_torch(workspace_root: Path, env: dict[str, str]) -> dict[str, object]:
    completed = run_logged(
        [
            sys.executable,
            "-c",
            (
                "import json, torch; "
                "arch = torch.cuda.get_arch_list() if torch.cuda.is_available() else []; "
                "payload = [('version', torch.__version__), ('cuda', getattr(torch.version, 'cuda', None)), ('arch_list', arch)]; "
                "print(json.dumps(payload))"
            ),
        ],
        cwd=workspace_root,
        env=env,
    )
    payload = {{
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }}
    try:
        parsed = dict(json.loads(completed.stdout.strip().splitlines()[-1]))
        payload.update(parsed)
    except Exception:
        pass
    return payload


def ensure_compatible_torch(workspace_root: Path, env: dict[str, str]) -> dict[str, object]:
    before = inspect_torch(workspace_root, env)
    if "sm_60" in (before.get("arch_list") or []):
        return {{"action": "keep", "before": before, "after": before}}

    uninstall = run_logged(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
        cwd=workspace_root,
        env=env,
    )
    install = run_logged(TORCH_INSTALL_COMMAND, cwd=workspace_root, env=env)
    after = inspect_torch(workspace_root, env)
    if "sm_60" not in (after.get("arch_list") or []):
        raise RuntimeError(
            "torch 兼容性修复失败\\n"
            f"before={{before}}\\n"
            f"uninstall_stdout={{uninstall.stdout}}\\n"
            f"uninstall_stderr={{uninstall.stderr}}\\n"
            f"install_stdout={{install.stdout}}\\n"
            f"install_stderr={{install.stderr}}\\n"
            f"after={{after}}"
        )
    return {{
        "action": "reinstall",
        "before": before,
        "after": after,
        "uninstall_stdout": uninstall.stdout,
        "uninstall_stderr": uninstall.stderr,
        "install_stdout": install.stdout,
        "install_stderr": install.stderr,
    }}


def main() -> None:
    workspace_root = resolve_work_root() / WORKSPACE_NAME
    workspace_root.mkdir(parents=True, exist_ok=True)
    env = {{
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }}
    try:
        input_root = resolve_input_root()
        code_root = resolve_dataset_root(input_root, CODE_DATASET_SLUG, "repo_payload.zip")
        data_root = resolve_dataset_root(input_root, DATA_DATASET_SLUG, "data_payload.zip")
        write_status(
            workspace_root,
            {{
                "stage": "boot",
                "input_root": str(input_root),
                "code_root": str(code_root),
                "data_root": str(data_root),
            }},
        )
        extract_payload(code_root, "repo_payload.zip", workspace_root)
        extract_payload(data_root, "data_payload.zip", workspace_root)
        torch_info = ensure_compatible_torch(workspace_root, env)
        write_status(workspace_root, {{"stage": "torch_ready", "torch_info": torch_info}})
        entry = workspace_root / ENTRY_REL_PATH
        completed = run_logged([sys.executable, str(entry), "--mode", "full"], cwd=workspace_root, env=env)
        (workspace_root / "kernel_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (workspace_root / "kernel_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"模型脚本失败，returncode={{completed.returncode}}\\nstdout:\\n{{completed.stdout}}\\nstderr:\\n{{completed.stderr}}"
            )
        write_status(workspace_root, {{"stage": "completed", "entry": ENTRY_REL_PATH}})
    except Exception as exc:
        write_status(
            workspace_root,
            {{
                "stage": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }},
        )
        raise


if __name__ == "__main__":
    main()
"""


def prepare_kernel_dir(
    run_dir: Path,
    *,
    code_dataset_id: str,
    data_dataset_id: str,
    kernel_id: str,
    kernel_title: str,
    entry_rel_path: str,
) -> Path:
    kernel_dir = run_dir / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    entry_code = render_kernel_entry(
        code_dataset_slug=code_dataset_id.split("/", 1)[1],
        data_dataset_slug=data_dataset_id.split("/", 1)[1],
        entry_rel_path=entry_rel_path.replace("\\", "/"),
    )
    (kernel_dir / "run_from_datasets.py").write_text(entry_code, encoding="utf-8")
    write_json(
        kernel_dir / "kernel-metadata.json",
        {
            "id": kernel_id,
            "title": kernel_title,
            "code_file": "run_from_datasets.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [code_dataset_id, data_dataset_id],
        },
    )
    return kernel_dir


def ensure_dataset_created(dataset_id: str, publish_dir: Path, log_path: Path) -> None:
    command = ["kaggle", "datasets", "create", "-p", str(publish_dir), "-r", "skip"]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    log_path.write_text(
        f"COMMAND:\n{' '.join(command)}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode == 0:
        return
    combined = f"{completed.stdout}\n{completed.stderr}".lower()
    if "already exists" in combined or "409" in combined or "cannot create" in combined:
        return
    raise RuntimeError(f"dataset create failed for {dataset_id}")


def run_publish_flow(
    *,
    publish_spec_path: Path,
    accelerator: str,
    runtime_seconds: int,
    log_path: Path,
) -> None:
    command = [
        sys.executable,
        str(SKILL_RUN_PUBLISH_FLOW),
        "--publish-spec",
        str(publish_spec_path),
        "--accelerator",
        accelerator,
        "--runtime-seconds",
        str(runtime_seconds),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    log_path.write_text(
        f"COMMAND:\n{' '.join(command)}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError("publish flow failed")


def add_publish_args(parser: argparse.ArgumentParser, *, default_code_slug: str, default_kernel_slug: str) -> None:
    parser.add_argument("--username", default=None, help="Kaggle username")
    parser.add_argument("--code-dataset-slug", default=default_code_slug)
    parser.add_argument("--kernel-slug", default=default_kernel_slug)
    parser.add_argument("--data-dataset-slug", default="windywind-phase3-sequence-data")
    parser.add_argument("--data-dataset-id", default=None)
    parser.add_argument("--accelerator", default="P100")
    parser.add_argument("--runtime-seconds", type=int, default=14400)
    parser.add_argument("--skip-remote", action="store_true")
