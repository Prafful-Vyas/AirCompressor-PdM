import pytest

from src.config import DROP_COLS, RAW_FEATURE_COLS
from src.ingestion import DataIngestor


def make_csv_text() -> str:
    header = RAW_FEATURE_COLS + DROP_COLS
    row1 = [1.0] * len(RAW_FEATURE_COLS) + [1, "Stable", 0, 0, 0, 0]
    row2 = [2.0] * len(RAW_FEATURE_COLS) + [2, "Stable", 0, 0, 0, 0]
    lines = [",".join(header)]
    lines.append(",".join(str(v) for v in row1))
    lines.append(",".join(str(v) for v in row2))
    return "\n".join(lines) + "\n"


def test_load_data_missing_file_raises(tmp_path):
    ingestor = DataIngestor(str(tmp_path / "does_not_exist.csv"))
    with pytest.raises(FileNotFoundError):
        ingestor.load_data()


def test_load_data_success(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(make_csv_text())

    ingestor = DataIngestor(str(csv_path))
    df = ingestor.load_data()

    assert len(df) == 2
    assert ingestor.df is not None


def test_load_data_raises_on_missing_columns(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n")

    ingestor = DataIngestor(str(csv_path))
    with pytest.raises(ValueError, match="missing required columns"):
        ingestor.load_data()


def test_load_data_raises_on_non_numeric_column(tmp_path):
    header = RAW_FEATURE_COLS + DROP_COLS
    row = ["not-a-number"] + ["1.0"] * (len(RAW_FEATURE_COLS) - 1)
    row += ["1", "Stable", "0", "0", "0", "0"]
    text = ",".join(header) + "\n" + ",".join(row) + "\n"

    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(text)

    ingestor = DataIngestor(str(csv_path))
    with pytest.raises(ValueError, match="Non-numeric values"):
        ingestor.load_data()
