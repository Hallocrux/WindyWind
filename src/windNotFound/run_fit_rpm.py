from __future__ import annotations

import argparse

try:
    from fit_rpm import fit_task_rpm
    from task_io import load_task
except ModuleNotFoundError:
    from src.windNotFound.fit_rpm import fit_task_rpm
    from src.windNotFound.task_io import load_task


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="windNotFound 21 点 RPM 拟合入口。")
    parser.add_argument("--task", required=True, help="标注任务 YAML 路径。")
    args = parser.parse_args(argv)

    result = fit_task_rpm(load_task(args.task))
    print(f"RPM 汇总: {result['summary_path']}")
    for selector in result["payload"]["selectors"]:
        print(
            f"selector={selector['selector_index']} "
            f"status={selector['status']} "
            f"rpm={selector['rpm']}"
        )


if __name__ == "__main__":
    main()
