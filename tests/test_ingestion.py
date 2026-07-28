import pytest

from src.ingestion import DataIngestor


def test_load_data_missing_file_raises(tmp_path):
    ingestor = DataIngestor(str(tmp_path / "does_not_exist.csv"))
    with pytest.raises(FileNotFoundError):
        ingestor.load_data()


def test_load_data_success(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n")

    ingestor = DataIngestor(str(csv_path))
    df = ingestor.load_data()

    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2
    assert ingestor.df is not None
