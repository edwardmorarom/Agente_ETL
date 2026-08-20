from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core.ingestion import (
    detect_id_columns,
    drop_rows_without_information,
    main,
)


def test_row_with_all_missing_non_id_columns_is_dropped() -> None:
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "x": [1.0, None, 3.0],
            "y": [2.0, None, 4.0],
        }
    )
    cleaned, dropped_rows = drop_rows_without_information(
        df,
        id_columns=["id"],
        threshold_pct=80.0,
    )

    assert list(cleaned["id"]) == ["a", "c"]
    assert dropped_rows == [
        {"row_index": 1, "id_value": {"id": "b"}, "pct_missing": 100.0}
    ]


def test_row_with_seventy_nine_percent_missing_is_not_dropped() -> None:
    row = {"id": "a"}
    row.update({f"x{i}": None if i < 79 else float(i) for i in range(100)})
    df = pd.DataFrame([row])
    cleaned, dropped_rows = drop_rows_without_information(
        df,
        id_columns=["id"],
        threshold_pct=80.0,
    )

    assert len(cleaned) == 1
    assert dropped_rows == []


def test_float_column_with_unique_values_is_not_detected_as_id() -> None:
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "continuous": [1.1, 2.2, 3.3],
        }
    )

    id_columns = detect_id_columns(df)

    assert id_columns == ["id"]


def test_unique_text_column_is_detected_as_id() -> None:
    df = pd.DataFrame(
        {
            "subject_code": ["p001", "p002", "p003"],
            "score": [1.0, 2.0, 3.0],
        }
    )

    assert detect_id_columns(df) == ["subject_code"]


def test_main_writes_rows_dropped_no_information_to_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "clean.csv"
    output_profile = tmp_path / "profile.json"
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "x": [1.0, None, 3.0],
            "y": [2.0, None, 4.0],
        }
    ).to_csv(input_csv, index=False)

    exit_code = main(
        [
            "--input",
            str(input_csv),
            "--output_csv",
            str(output_csv),
            "--output_profile_json",
            str(output_profile),
            "--id_columns",
            "id",
        ]
    )

    profile = json.loads(output_profile.read_text(encoding="utf-8"))
    cleaned = pd.read_csv(output_csv)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert list(cleaned["id"]) == ["a", "c"]
    assert profile["rows_dropped_no_information"] == {
        "threshold_pct": 80.0,
        "id_columns_detected": ["id"],
        "count": 1,
        "dropped_rows": [
            {"row_index": 1, "id_value": {"id": "b"}, "pct_missing": 100.0}
        ],
    }
    assert "Se eliminaron 1 filas sin informacion suficiente" in captured.out
