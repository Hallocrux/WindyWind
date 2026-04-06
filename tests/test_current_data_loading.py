from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.current.data_loading import (
    CleaningConfig,
    DatasetRecord,
    build_dataset_inventory,
    load_clean_signal_frame,
    scan_dataset_records,
)
from src.current.data_quality import QualityConfig, build_data_quality_report
from src.current.features import WindowConfig, build_case_feature_frame


class CurrentDataLoadingTests(unittest.TestCase):
    def _make_rows(self, signal_pairs: list[tuple[object, object]]) -> list[dict[str, object]]:
        base_time = pd.Timestamp("2026-03-30 16:24:25.000")
        rows: list[dict[str, object]] = []
        for index, (sig_a, sig_b) in enumerate(signal_pairs):
            rows.append(
                {
                    "time": (base_time + pd.Timedelta(milliseconds=20 * index)).strftime(
                        "%Y/%m/%d %H:%M:%S.%f"
                    )[:-3],
                    "sig_a": sig_a,
                    "sig_b": sig_b,
                }
            )
        return rows

    def _make_record(self, rows: list[dict[str, object]]) -> tuple[DatasetRecord, list[str]]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        csv_path = Path(temp_dir.name) / "case.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        record = DatasetRecord(
            case_id=999,
            display_name="测试工况",
            file_name=csv_path.name,
            file_path=csv_path,
            wind_speed=3.9,
            rpm=166.0,
            is_labeled=True,
            original_file_name="case.csv",
            label_source="unit-test",
            notes="",
        )
        signal_columns = [column for column in rows[0].keys() if column != "time"]
        return record, signal_columns

    def _write_case_csv(self, directory: Path, case_id: int) -> None:
        csv_path = directory / f"工况{case_id}.csv"
        pd.DataFrame(
            [
                {"time": '="2026/03/30 16:24:25.000"', "sig_a": 1.0, "sig_b": 10.0},
                {"time": '="2026/03/30 16:24:25.020"', "sig_a": 2.0, "sig_b": 11.0},
                {"time": '="2026/03/30 16:24:25.040"', "sig_a": 3.0, "sig_b": 12.0},
            ]
        ).to_csv(csv_path, index=False)

    def _write_manifest(
        self,
        directory: Path,
        rows: list[dict[str, object]],
        *,
        file_name: str = "dataset_manifest.csv",
    ) -> Path:
        manifest_path = directory / file_name
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        return manifest_path

    def test_trim_edge_missing_and_fill_short_middle_gap(self) -> None:
        record, signal_columns = self._make_record(
            [
                {"time": '="2026/03/30 16:24:25.000"', "sig_a": "", "sig_b": 1.0},
                {"time": '="2026/03/30 16:24:25.020"', "sig_a": 10.0, "sig_b": 1.1},
                {"time": '="2026/03/30 16:24:25.040"', "sig_a": "", "sig_b": 1.2},
                {"time": '="2026/03/30 16:24:25.060"', "sig_a": 14.0, "sig_b": 1.3},
                {"time": '="2026/03/30 16:24:25.080"', "sig_a": 16.0, "sig_b": 1.4},
                {"time": '="2026/03/30 16:24:25.100"', "sig_a": 18.0, "sig_b": ""},
            ]
        )

        cleaned = load_clean_signal_frame(record, signal_columns)

        self.assertEqual(len(cleaned), 4)
        self.assertEqual(cleaned["time"].iloc[0], pd.Timestamp("2026-03-30 16:24:25.020"))
        self.assertEqual(cleaned["time"].iloc[-1], pd.Timestamp("2026-03-30 16:24:25.080"))
        self.assertEqual(cleaned["__segment_id"].tolist(), [0, 0, 0, 0])
        self.assertEqual(cleaned["__row_has_missing"].tolist(), [0, 1, 0, 0])
        self.assertEqual(cleaned["__row_missing_count"].tolist(), [0, 1, 0, 0])
        self.assertAlmostEqual(cleaned.loc[1, "sig_a"], 12.0, places=6)
        self.assertFalse(cleaned[signal_columns].isna().any().any())

    def test_drop_long_middle_gap_and_split_segments(self) -> None:
        rows = self._make_rows(
            [
                (1.0, 10.0),
                (2.0, 11.0),
                (3.0, 12.0),
                ("", 13.0),
                ("", 14.0),
                ("", 15.0),
                ("", 16.0),
                ("", 17.0),
                ("", 18.0),
                (9.0, 19.0),
            ]
        )
        record, signal_columns = self._make_record(rows)

        cleaned = load_clean_signal_frame(record, signal_columns)

        self.assertEqual(len(cleaned), 4)
        self.assertEqual(cleaned["__segment_id"].tolist(), [0, 0, 0, 1])
        self.assertEqual(cleaned["__row_has_missing"].tolist(), [0, 0, 0, 0])
        self.assertEqual(cleaned["__row_missing_count"].tolist(), [0, 0, 0, 0])
        self.assertEqual(cleaned["sig_a"].tolist(), [1.0, 2.0, 3.0, 9.0])

    def test_allow_short_gap_and_drop_long_gap_in_same_case(self) -> None:
        rows = self._make_rows(
            [
                (0.0, 10.0),
                (1.0, 11.0),
                (2.0, 12.0),
                ("", 13.0),
                (4.0, 14.0),
                (5.0, 15.0),
                ("", 16.0),
                ("", 17.0),
                ("", 18.0),
                ("", 19.0),
                ("", 20.0),
                ("", 21.0),
                (12.0, 22.0),
                (13.0, 23.0),
            ]
        )
        record, signal_columns = self._make_record(rows)

        cleaned = load_clean_signal_frame(record, signal_columns)

        self.assertEqual(len(cleaned), 8)
        self.assertEqual(cleaned["__segment_id"].tolist(), [0, 0, 0, 0, 0, 0, 1, 1])
        self.assertEqual(cleaned["__row_has_missing"].tolist(), [0, 0, 0, 1, 0, 0, 0, 0])
        self.assertEqual(cleaned["__row_missing_count"].tolist(), [0, 0, 0, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(cleaned.loc[3, "sig_a"], 3.0, places=6)

    def test_raise_when_edge_trim_removes_everything(self) -> None:
        record, signal_columns = self._make_record(
            [
                {"time": "2026/03/30 16:24:25.000", "sig_a": "", "sig_b": ""},
                {"time": "2026/03/30 16:24:25.020", "sig_a": "", "sig_b": ""},
                {"time": "2026/03/30 16:24:25.040", "sig_a": "", "sig_b": ""},
            ]
        )

        with self.assertRaisesRegex(ValueError, "清洗后无有效数据"):
            load_clean_signal_frame(record, signal_columns)

    def test_time_cleanup_sort_and_deduplicate_still_work(self) -> None:
        record, signal_columns = self._make_record(
            [
                {"time": '="2026/03/30 16:24:25.040"', "sig_a": 3.0, "sig_b": 30.0},
                {"time": '="2026/03/30 16:24:25.000"', "sig_a": 1.0, "sig_b": 10.0},
                {"time": '="2026/03/30 16:24:25.020"', "sig_a": 2.0, "sig_b": 20.0},
                {"time": '="2026/03/30 16:24:25.020"', "sig_a": 999.0, "sig_b": 999.0},
            ]
        )

        cleaned = load_clean_signal_frame(record, signal_columns)

        self.assertEqual(len(cleaned), 3)
        self.assertEqual(
            cleaned["time"].tolist(),
            [
                pd.Timestamp("2026-03-30 16:24:25.000"),
                pd.Timestamp("2026-03-30 16:24:25.020"),
                pd.Timestamp("2026-03-30 16:24:25.040"),
            ],
        )
        self.assertEqual(cleaned["sig_a"].tolist(), [1.0, 2.0, 3.0])

    def test_segmented_windowing_never_crosses_long_gap(self) -> None:
        rows = self._make_rows(
            [
                (0.0, 10.0),
                (1.0, 11.0),
                (2.0, 12.0),
                ("", 13.0),
                (4.0, 14.0),
                (5.0, 15.0),
                ("", 16.0),
                ("", 17.0),
                ("", 18.0),
                ("", 19.0),
                ("", 20.0),
                ("", 21.0),
                (12.0, 22.0),
                (13.0, 23.0),
            ]
        )
        record, signal_columns = self._make_record(rows)
        cleaned = load_clean_signal_frame(record, signal_columns)

        feature_df = build_case_feature_frame(
            record,
            cleaned,
            WindowConfig(sampling_rate=50.0, window_size=3, step_size=1),
        )

        self.assertEqual(len(feature_df), 4)
        self.assertEqual(feature_df["window_index"].tolist(), [0, 1, 2, 3])
        self.assertEqual(feature_df["raw_missing_rows"].tolist(), [0, 1, 1, 1])
        self.assertEqual(
            feature_df["start_time"].tolist(),
            [
                pd.Timestamp("2026-03-30 16:24:25.000"),
                pd.Timestamp("2026-03-30 16:24:25.020"),
                pd.Timestamp("2026-03-30 16:24:25.040"),
                pd.Timestamp("2026-03-30 16:24:25.060"),
            ],
        )

    def test_raise_when_all_segments_are_shorter_than_window(self) -> None:
        rows = self._make_rows(
            [
                (1.0, 10.0),
                (2.0, 11.0),
                ("", 12.0),
                ("", 13.0),
                ("", 14.0),
                ("", 15.0),
                ("", 16.0),
                ("", 17.0),
                (8.0, 18.0),
                (9.0, 19.0),
            ]
        )
        record, signal_columns = self._make_record(rows)
        cleaned = load_clean_signal_frame(record, signal_columns)

        with self.assertRaisesRegex(ValueError, "无足够长连续段可切出窗口"):
            build_case_feature_frame(
                record,
                cleaned,
                WindowConfig(sampling_rate=50.0, window_size=3, step_size=1),
            )

    def test_quality_report_uses_actual_segment_windows(self) -> None:
        rows = self._make_rows(
            [
                (0.0, 10.0),
                (1.0, 11.0),
                (2.0, 12.0),
                ("", 13.0),
                (4.0, 14.0),
                (5.0, 15.0),
                ("", 16.0),
                ("", 17.0),
                ("", 18.0),
                ("", 19.0),
                ("", 20.0),
                ("", 21.0),
                (12.0, 22.0),
                (13.0, 23.0),
            ]
        )
        record, _ = self._make_record(rows)

        case_df, _ = build_data_quality_report(
            records=[record],
            window_config=WindowConfig(sampling_rate=50.0, window_size=3, step_size=1),
            quality_config=QualityConfig(heavy_missing_window_ratio=0.05),
            cleaning_config=CleaningConfig(max_middle_interp_gap_rows=5),
        )

        row = case_df.iloc[0]
        self.assertEqual(int(row["internal_short_gap_rows"]), 1)
        self.assertEqual(int(row["internal_long_gap_rows_dropped"]), 6)
        self.assertEqual(int(row["continuous_segment_count"]), 2)
        self.assertEqual(int(row["rows_after_long_gap_drop"]), 8)
        self.assertEqual(int(row["windows_total"]), 4)
        self.assertEqual(int(row["windows_with_missing"]), 3)
        self.assertEqual(int(row["windows_with_heavy_missing"]), 3)
        self.assertAlmostEqual(float(row["worst_window_missing_ratio"]), 1.0 / 6.0, places=6)

    def test_scan_dataset_records_reads_manifest_only(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        data_dir = root / "datasets"
        data_dir.mkdir()
        self._write_case_csv(data_dir, 1)
        self._write_case_csv(data_dir, 2)
        manifest_path = self._write_manifest(
            root,
            [
                {
                    "case_id": 2,
                    "display_name": "工况2",
                    "wind_speed": "",
                    "rpm": "",
                    "original_file_name": "原始工况2.csv",
                    "label_source": "人工核验",
                    "notes": "无标签",
                },
                {
                    "case_id": 1,
                    "display_name": "工况1",
                    "wind_speed": 2.12,
                    "rpm": 82,
                    "original_file_name": "原始工况1.csv",
                    "label_source": "人工核验",
                    "notes": "",
                },
            ],
        )

        records = scan_dataset_records(data_dir=data_dir, manifest_path=manifest_path)

        self.assertEqual([record.case_id for record in records], [1, 2])
        self.assertEqual([record.file_name for record in records], ["工况1.csv", "工况2.csv"])
        self.assertTrue(records[0].is_labeled)
        self.assertFalse(records[1].is_labeled)
        self.assertEqual(records[1].notes, "无标签")

    def test_scan_dataset_records_raises_for_duplicate_case_id(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        data_dir = root / "datasets"
        data_dir.mkdir()
        self._write_case_csv(data_dir, 1)
        manifest_path = self._write_manifest(
            root,
            [
                {
                    "case_id": 1,
                    "display_name": "工况1-A",
                    "wind_speed": 2.12,
                    "rpm": 82,
                    "original_file_name": "a.csv",
                    "label_source": "人工核验",
                    "notes": "",
                },
                {
                    "case_id": 1,
                    "display_name": "工况1-B",
                    "wind_speed": 2.2,
                    "rpm": 85,
                    "original_file_name": "b.csv",
                    "label_source": "人工核验",
                    "notes": "",
                },
            ],
        )

        with self.assertRaisesRegex(ValueError, "重复的 case_id"):
            scan_dataset_records(data_dir=data_dir, manifest_path=manifest_path)

    def test_scan_dataset_records_raises_for_missing_standard_file(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        data_dir = root / "datasets"
        data_dir.mkdir()
        manifest_path = self._write_manifest(
            root,
            [
                {
                    "case_id": 1,
                    "display_name": "工况1",
                    "wind_speed": 2.12,
                    "rpm": 82,
                    "original_file_name": "原始工况1.csv",
                    "label_source": "人工核验",
                    "notes": "",
                }
            ],
        )

        with self.assertRaisesRegex(FileNotFoundError, "标准数据文件不存在"):
            scan_dataset_records(data_dir=data_dir, manifest_path=manifest_path)

    def test_scan_dataset_records_raises_for_untracked_csv(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        data_dir = root / "datasets"
        data_dir.mkdir()
        self._write_case_csv(data_dir, 1)
        pd.DataFrame(
            [{"time": "2026/03/30 16:24:25.000", "sig_a": 1.0, "sig_b": 10.0}]
        ).to_csv(data_dir / "额外文件.csv", index=False)
        manifest_path = self._write_manifest(
            root,
            [
                {
                    "case_id": 1,
                    "display_name": "工况1",
                    "wind_speed": 2.12,
                    "rpm": 82,
                    "original_file_name": "原始工况1.csv",
                    "label_source": "人工核验",
                    "notes": "",
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "未登记的 CSV 文件"):
            scan_dataset_records(data_dir=data_dir, manifest_path=manifest_path)

    def test_build_dataset_inventory_estimates_sampling_rate(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        data_dir = root / "datasets"
        data_dir.mkdir()
        csv_path = data_dir / "工况1.csv"
        pd.DataFrame(
            [
                {
                    "time": '="2026/03/30 16:24:25.000"',
                    "sig_a": 1.0,
                    "sig_b": 10.0,
                    "WSMS00005.AccX": 0.0,
                },
                {
                    "time": '="2026/03/30 16:24:25.020"',
                    "sig_a": 2.0,
                    "sig_b": 11.0,
                    "WSMS00005.AccX": 0.0,
                },
                {
                    "time": '="2026/03/30 16:24:25.040"',
                    "sig_a": 3.0,
                    "sig_b": 12.0,
                    "WSMS00005.AccX": 0.0,
                },
            ]
        ).to_csv(csv_path, index=False)
        manifest_path = self._write_manifest(
            root,
            [
                {
                    "case_id": 1,
                    "display_name": "工况1",
                    "wind_speed": 2.12,
                    "rpm": 82,
                    "original_file_name": "原始工况1.csv",
                    "label_source": "人工核验",
                    "notes": "",
                }
            ],
        )

        records = scan_dataset_records(data_dir=data_dir, manifest_path=manifest_path)
        inventory = build_dataset_inventory(records)

        self.assertEqual(inventory.loc[0, "file_name"], "工况1.csv")
        self.assertEqual(inventory.loc[0, "row_count"], 3)
        self.assertEqual(inventory.loc[0, "column_count"], 4)
        self.assertAlmostEqual(inventory.loc[0, "duration_seconds"], 0.04, places=6)
        self.assertAlmostEqual(inventory.loc[0, "sampling_hz_est"], 50.0, places=6)
        self.assertEqual(inventory.loc[0, "has_invalid_wsms00005"], 1)


if __name__ == "__main__":
    unittest.main()
