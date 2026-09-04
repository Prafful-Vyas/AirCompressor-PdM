import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from src import promote
from src.config import registered_model_name
from src.predict import Predictor
from src.settings import get_settings


def register_a_version(target: str, holdout_f1: float) -> str:
    """Registers one candidate model version for `target`, mirroring what
    Trainer.train_target does, without running the full training pipeline."""
    mlflow.set_experiment("test-experiment")
    with mlflow.start_run():
        mlflow.log_param("target", target)
        mlflow.log_metric("holdout_f1", holdout_f1)
        model = LogisticRegression().fit(np.array([[1], [2], [3], [4]]), [0, 1, 0, 1])
        mlflow.sklearn.log_model(
            model, name="model", registered_model_name=registered_model_name(target)
        )
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{registered_model_name(target)}'")
    return max(versions, key=lambda v: int(v.version)).version


def test_list_versions_surfaces_registered_candidate(isolated_mlflow_tracking, capsys):
    version = register_a_version("bearings", holdout_f1=0.9)

    promote.list_versions("bearings")

    out = capsys.readouterr().out
    assert f"version={version}" in out
    assert "holdout_f1=0.9" in out


def test_promote_refuses_below_threshold_without_force(
    isolated_mlflow_tracking, monkeypatch
):
    monkeypatch.setenv("ACPDM_MIN_HOLDOUT_F1_FOR_PROMOTION", "0.5")
    get_settings.cache_clear()
    version = register_a_version("bearings", holdout_f1=0.1)

    with pytest.raises(SystemExit):
        promote.promote("bearings", version)

    get_settings.cache_clear()


def test_promote_succeeds_with_force_below_threshold(
    isolated_mlflow_tracking, monkeypatch
):
    monkeypatch.setenv("ACPDM_MIN_HOLDOUT_F1_FOR_PROMOTION", "0.5")
    get_settings.cache_clear()
    version = register_a_version("bearings", holdout_f1=0.1)

    promote.promote("bearings", version, force=True)

    client = MlflowClient()
    mv = client.get_model_version_by_alias(
        registered_model_name("bearings"), get_settings().model_registry_alias
    )
    assert mv.version == version

    get_settings.cache_clear()


def test_promote_then_predictor_load_picks_up_promoted_version(
    isolated_mlflow_tracking,
):
    for target in ["bearings", "wpump", "radiator", "exvalve"]:
        version = register_a_version(target, holdout_f1=0.9)
        promote.promote(target, version)

    predictor = Predictor(experiment_name="test-experiment")
    predictor.load()

    assert set(predictor.models.keys()) == {
        "bearings",
        "wpump",
        "radiator",
        "exvalve",
    }


def test_current_reports_no_version_before_promotion(isolated_mlflow_tracking, capsys):
    register_a_version("bearings", holdout_f1=0.9)

    promote.show_current("bearings")

    out = capsys.readouterr().out
    assert "No version currently holds" in out
