from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path


TRY_NAME = "008_current_kaggle_thin_kernel"
REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="发布 current 表格主线到 Kaggle，并用 thin kernel 组装目录结构。"
    )
    parser.add_argument("--username", default=None, help="Kaggle 用户名；默认从 `kaggle config view` 解析。")
    parser.add_argument(
        "--code-dataset-slug",
        default="windywind-try-008-current-code",
        help="代码 dataset slug。",
    )
    parser.add_argument(
        "--data-dataset-slug",
        default="windywind-try-008-current-data",
        help="数据 dataset slug。",
    )
    parser.add_argument(
        "--kernel-slug",
        default="windywind-try-008-current-thin-kernel",
        help="kernel slug。",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="等待 Kaggle kernel 结束的最长秒数。",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="轮询 Kaggle kernel 状态的间隔秒数。",
    )
    parser.add_argument(
        "--skip-remote",
        action="store_true",
        help="只做本地 smoke test，不执行 Kaggle 发布。",
    )
    return parser.parse_args()


def ensure_command_exists(command_name: str) -> None:
    if shutil.which(command_name) is None:
        raise RuntimeError(f"未在 PATH 中找到命令：{command_name}")


def detect_kaggle_username() -> str:
    env_username = os.environ.get("KAGGLE_USERNAME")
    if env_username:
        return env_username.strip()

    summary_candidates = sorted(OUTPUT_ROOT.glob("runs/*/summary.json"), reverse=True)
    for summary_path in summary_candidates:
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        username = str(payload.get("username", "")).strip()
        if username:
            return username

    result = subprocess.run(
        ["kaggle", "config", "view"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "无法通过 `kaggle config view` 读取当前配置。\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    match = re.search(r"^\s*-\s*username:\s*(\S+)\s*$", result.stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("未从 `kaggle config view` 输出中解析到 username。")
    return match.group(1)


def make_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def create_zip(archive_path: Path, items: list[tuple[Path, Path]]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source_path, arcname in items:
            if source_path.is_dir():
                for child in sorted(source_path.rglob("*")):
                    if child.is_dir():
                        continue
                    target = (arcname / child.relative_to(source_path)).as_posix()
                    zf.write(child, target)
            else:
                zf.write(source_path, arcname.as_posix())


def build_payloads(run_dir: Path) -> dict[str, Path]:
    publish_root = run_dir / "publish_payloads"
    code_dir = publish_root / "code_dataset"
    data_dir = publish_root / "data_dataset"
    kernel_dir = run_dir / "kernel"

    code_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    create_zip(
        code_dir / "repo_payload.zip",
        [
            (REPO_ROOT / "main.py", Path("main.py")),
            (REPO_ROOT / "src" / "__init__.py", Path("src") / "__init__.py"),
            (REPO_ROOT / "src" / "current", Path("src") / "current"),
        ],
    )
    create_zip(
        data_dir / "data_payload.zip",
        [
            (REPO_ROOT / "data" / "final" / "dataset_manifest.csv", Path("data") / "final" / "dataset_manifest.csv"),
            (REPO_ROOT / "data" / "final" / "datasets", Path("data") / "final" / "datasets"),
        ],
    )

    write_json(
        code_dir / "payload_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "payload": "repo_payload.zip",
            "entries": [
                "main.py",
                "src/__init__.py",
                "src/current/",
            ],
        },
    )
    write_json(
        data_dir / "payload_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "payload": "data_payload.zip",
            "entries": [
                "data/final/dataset_manifest.csv",
                "data/final/datasets/",
            ],
        },
    )
    return {
        "publish_root": publish_root,
        "code_dir": code_dir,
        "data_dir": data_dir,
        "kernel_dir": kernel_dir,
    }


def render_kernel_entry(code_dataset_slug: str, data_dataset_slug: str) -> str:
    template_path = TRY_ROOT / "thin_kernel_template.py"
    template = template_path.read_text(encoding="utf-8")
    return (
        template.replace("{{CODE_DATASET_SLUG}}", code_dataset_slug)
        .replace("{{DATA_DATASET_SLUG}}", data_dataset_slug)
    )


def prepare_kernel_dir(kernel_dir: Path, code_dataset_id: str, data_dataset_id: str, kernel_id: str) -> Path:
    entry_path = kernel_dir / "run_current_from_datasets.py"
    write_text(
        entry_path,
        render_kernel_entry(
            code_dataset_slug=code_dataset_id.split("/", maxsplit=1)[1],
            data_dataset_slug=data_dataset_id.split("/", maxsplit=1)[1],
        ),
    )
    write_json(
        kernel_dir / "kernel-metadata.json",
        {
            "id": kernel_id,
            "title": "windywind try 008 current thin kernel",
            "code_file": entry_path.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": False,
            "dataset_sources": [
                code_dataset_id,
                data_dataset_id,
            ],
        },
    )
    return entry_path


def dataset_metadata_payload(dataset_id: str, title: str) -> dict[str, object]:
    return {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }


def build_command_log_name(args: list[str]) -> str:
    head = "_".join(args[:3]).replace("/", "_").replace("\\", "_")
    return f"{head}_{int(time.time() * 1000)}"


def run_command(
    args: list[str],
    log_dir: Path,
    *,
    cwd: Path | None = None,
    check: bool = True,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )
    log_name = build_command_log_name(args)
    write_text(log_dir / f"{log_name}.stdout.txt", completed.stdout)
    write_text(log_dir / f"{log_name}.stderr.txt", completed.stderr)
    write_json(
        log_dir / f"{log_name}.meta.json",
        {
            "command": args,
            "cwd": str(cwd) if cwd else None,
            "returncode": completed.returncode,
        },
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"命令失败：{' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def publish_dataset(
    publish_dir: Path,
    dataset_id: str,
    title: str,
    version_notes: str,
    log_dir: Path,
) -> str:
    write_json(publish_dir / "dataset-metadata.json", dataset_metadata_payload(dataset_id, title))

    create_result = run_command(
        ["kaggle", "datasets", "create", "-p", str(publish_dir), "-r", "skip"],
        log_dir=log_dir,
        check=False,
    )
    if create_result.returncode == 0:
        return "created"

    error_text = f"{create_result.stdout}\n{create_result.stderr}".lower()
    already_exists_markers = [
        "already exists",
        "409",
        "slug",
        "cannot create",
        "already have a dataset",
    ]
    if any(marker in error_text for marker in already_exists_markers):
        version_result = run_command(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(publish_dir),
                "-r",
                "skip",
                "-m",
                version_notes,
            ],
            log_dir=log_dir,
            check=True,
        )
        if version_result.returncode == 0:
            return "versioned"

    raise RuntimeError(
        f"发布 dataset 失败：{dataset_id}\n"
        f"stdout:\n{create_result.stdout}\n"
        f"stderr:\n{create_result.stderr}"
    )


def wait_for_dataset_ready(
    dataset_id: str,
    probe_root: Path,
    log_dir: Path,
    timeout_seconds: int,
    poll_seconds: int,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    probe_dir = probe_root / dataset_id.replace("/", "__")
    probe_dir.mkdir(parents=True, exist_ok=True)
    last_state = "unknown"
    while time.monotonic() <= deadline:
        result = run_command(
            ["kaggle", "datasets", "metadata", dataset_id, "-p", str(probe_dir)],
            log_dir=log_dir,
            check=False,
        )
        if result.returncode == 0:
            return "ready"

        combined = f"{result.stdout}\n{result.stderr}".lower()
        if "404" in combined or "not found" in combined:
            last_state = "not_found"
        elif "403" in combined and "forbidden" in combined:
            last_state = "forbidden"
        else:
            last_state = "pending"
        time.sleep(poll_seconds)
    return last_state


def wait_for_kernel(kernel_id: str, log_dir: Path, timeout_seconds: int, poll_seconds: int) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_status = "unknown"
    while time.monotonic() <= deadline:
        result = run_command(
            ["kaggle", "kernels", "status", kernel_id],
            log_dir=log_dir,
            check=False,
        )
        combined = f"{result.stdout}\n{result.stderr}".lower()
        if "403" in combined and "forbidden" in combined:
            return "status_unavailable"
        if "complete" in combined or "completed" in combined or "ready" in combined:
            return "complete"
        if "running" in combined:
            last_status = "running"
        elif "queued" in combined or "pending" in combined:
            last_status = "queued"
        elif "failed" in combined or "error" in combined:
            return "failed"
        time.sleep(poll_seconds)
    return last_status


def wait_for_kernel_outputs(
    kernel_id: str,
    target_dir: Path,
    log_dir: Path,
    timeout_seconds: int,
    poll_seconds: int,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_state = "unknown"
    while time.monotonic() <= deadline:
        probe = run_command(
            ["kaggle", "kernels", "output", kernel_id, "-p", str(target_dir), "-o"],
            log_dir=log_dir,
            check=False,
        )
        combined = f"{probe.stdout}\n{probe.stderr}".lower()
        if probe.returncode == 0:
            return "complete"
        if "404" in combined or "not found" in combined:
            last_state = "not_found"
        elif "403" in combined and "forbidden" in combined:
            last_state = "permission_denied"
        else:
            last_state = "pending"
        time.sleep(poll_seconds)
    return last_state


def run_local_smoke(
    kernel_entry_path: Path,
    code_dataset_slug: str,
    data_dataset_slug: str,
    code_publish_dir: Path,
    data_publish_dir: Path,
    log_dir: Path,
) -> Path:
    smoke_root = log_dir.parent / "local_smoke"
    input_root = smoke_root / "input"
    work_root = smoke_root / "working"
    code_mount = input_root / code_dataset_slug
    data_mount = input_root / data_dataset_slug
    code_mount.mkdir(parents=True, exist_ok=True)
    data_mount.mkdir(parents=True, exist_ok=True)

    shutil.copy2(code_publish_dir / "repo_payload.zip", code_mount / "repo_payload.zip")
    shutil.copy2(code_publish_dir / "payload_manifest.json", code_mount / "payload_manifest.json")
    shutil.copy2(data_publish_dir / "data_payload.zip", data_mount / "data_payload.zip")
    shutil.copy2(data_publish_dir / "payload_manifest.json", data_mount / "payload_manifest.json")

    local_python_command = [sys.executable, str(kernel_entry_path)]
    if shutil.which("uv") is not None:
        local_python_command = ["uv", "run", "python", str(kernel_entry_path)]

    run_command(
        local_python_command,
        log_dir=log_dir,
        check=True,
        extra_env={
            "WW_KAGGLE_INPUT_ROOT": str(input_root),
            "WW_KAGGLE_WORK_ROOT": str(work_root),
        },
    )
    return work_root / "windywind"


def download_kernel_outputs(kernel_id: str, target_dir: Path, log_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(target_dir), "-o"],
        log_dir=log_dir,
        check=True,
    )


def main() -> None:
    ensure_command_exists("kaggle")
    args = parse_args()
    username = args.username or detect_kaggle_username()
    run_dir = make_run_dir()
    log_dir = run_dir / "command_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    code_dataset_id = f"{username}/{args.code_dataset_slug}"
    data_dataset_id = f"{username}/{args.data_dataset_slug}"
    kernel_id = f"{username}/{args.kernel_slug}"

    paths = build_payloads(run_dir)
    prepare_kernel_dir(
        kernel_dir=paths["kernel_dir"],
        code_dataset_id=code_dataset_id,
        data_dataset_id=data_dataset_id,
        kernel_id=kernel_id,
    )

    local_workspace = run_local_smoke(
        kernel_entry_path=paths["kernel_dir"] / "run_current_from_datasets.py",
        code_dataset_slug=args.code_dataset_slug,
        data_dataset_slug=args.data_dataset_slug,
        code_publish_dir=paths["code_dir"],
        data_publish_dir=paths["data_dir"],
        log_dir=log_dir,
    )

    summary: dict[str, object] = {
        "try_name": TRY_NAME,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "username": username,
        "code_dataset_id": code_dataset_id,
        "data_dataset_id": data_dataset_id,
        "kernel_id": kernel_id,
        "run_dir": str(run_dir),
        "local_workspace": str(local_workspace),
        "local_smoke_outputs_exist": (local_workspace / "outputs").exists(),
        "remote_publish_skipped": bool(args.skip_remote),
    }

    if not args.skip_remote:
        summary["code_dataset_publish"] = publish_dataset(
            publish_dir=paths["code_dir"],
            dataset_id=code_dataset_id,
            title="windywind try 008 current code payload",
            version_notes="try 008 current code payload update",
            log_dir=log_dir,
        )
        summary["data_dataset_publish"] = publish_dataset(
            publish_dir=paths["data_dir"],
            dataset_id=data_dataset_id,
            title="windywind try 008 current data payload",
            version_notes="try 008 current data payload update",
            log_dir=log_dir,
        )
        dataset_probe_root = run_dir / "dataset_ready_probe"
        summary["code_dataset_ready"] = wait_for_dataset_ready(
            dataset_id=code_dataset_id,
            probe_root=dataset_probe_root,
            log_dir=log_dir,
            timeout_seconds=min(args.timeout_seconds, 900),
            poll_seconds=args.poll_seconds,
        )
        summary["data_dataset_ready"] = wait_for_dataset_ready(
            dataset_id=data_dataset_id,
            probe_root=dataset_probe_root,
            log_dir=log_dir,
            timeout_seconds=min(args.timeout_seconds, 900),
            poll_seconds=args.poll_seconds,
        )
        run_command(
            ["kaggle", "kernels", "push", "-p", str(paths["kernel_dir"]), "-t", "3600"],
            log_dir=log_dir,
            check=True,
        )
        remote_outputs_dir = run_dir / "remote_outputs"
        kernel_status = wait_for_kernel(
            kernel_id=kernel_id,
            log_dir=log_dir,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        if kernel_status == "status_unavailable":
            kernel_status = wait_for_kernel_outputs(
                kernel_id=kernel_id,
                target_dir=remote_outputs_dir,
                log_dir=log_dir,
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
            )
        summary["kernel_status"] = kernel_status
        if kernel_status == "complete":
            summary["remote_outputs_dir"] = str(remote_outputs_dir)

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
