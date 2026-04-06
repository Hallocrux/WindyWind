from __future__ import annotations

import argparse

from annotate_ui import AnnotationSession


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="windNotFound 标注入口。")
    parser.add_argument("--task", required=True, help="标注任务 YAML 路径。")
    parser.add_argument(
        "--start-at", default=None, help="从 task_item_id 或任务序号开始。"
    )
    parser.add_argument(
        "--readonly", action="store_true", help="只读查看，不允许修改。"
    )
    parser.add_argument(
        "--show-done", action="store_true", help="启动时不跳过已完成项。"
    )
    args = parser.parse_args(argv)

    session = AnnotationSession(
        task_path=args.task,
        start_at=args.start_at,
        readonly=args.readonly,
        show_done=args.show_done,
    )
    print(f"标注输出目录: {session.output_dir}")
    print(f"标注文件: {session.jsonl_path}")
    print(f"汇总文件: {session.summary_path}")
    session.run()


if __name__ == "__main__":
    main()
