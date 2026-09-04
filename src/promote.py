"""CLI to promote a trained, registered model version to the alias that
Predictor serves (see src/settings.py's model_registry_alias, default
"production"). Training (src/train.py) only registers a candidate version
-- it never becomes servable until explicitly promoted here. Re-pointing
the alias to an older version is an instant rollback.

Usage:
    python -m src.promote list <target>
    python -m src.promote promote <target> <version> [--alias NAME] [--force]
    python -m src.promote current <target>
"""

import argparse
import logging
import sys

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from .config import TARGET_COLS, registered_model_name
from .settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _current_alias_version(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def list_versions(target: str) -> None:
    client = MlflowClient()
    name = registered_model_name(target)
    alias = get_settings().model_registry_alias
    versions = sorted(
        client.search_model_versions(f"name='{name}'"),
        key=lambda v: int(v.version),
        reverse=True,
    )

    if not versions:
        print(f"No registered versions for '{name}'. Run `python -m src.train` first.")
        return

    current = _current_alias_version(client, name, alias)
    for v in versions:
        run = client.get_run(v.run_id)
        holdout_f1 = run.data.metrics.get("holdout_f1")
        mean_cv_f1 = run.data.metrics.get("mean_cv_f1")
        marker = " <- current" if current and v.version == current.version else ""
        print(
            f"version={v.version} run_id={v.run_id} "
            f"holdout_f1={holdout_f1} mean_cv_f1={mean_cv_f1}{marker}"
        )


def promote(
    target: str, version: str, alias: str | None = None, force: bool = False
) -> None:
    client = MlflowClient()
    name = registered_model_name(target)
    alias = alias or get_settings().model_registry_alias
    threshold = get_settings().min_holdout_f1_for_promotion

    mv = client.get_model_version(name, version)
    run = client.get_run(mv.run_id)
    holdout_f1 = run.data.metrics.get("holdout_f1")

    if holdout_f1 is None:
        logger.warning(
            f"Version {version} of '{name}' has no holdout_f1 metric "
            "recorded; skipping the threshold check."
        )
    elif holdout_f1 < threshold and not force:
        logger.error(
            f"Refusing to promote '{name}' version {version}: "
            f"holdout_f1={holdout_f1:.4f} is below the minimum "
            f"{threshold:.4f}. Pass --force to override."
        )
        sys.exit(1)

    client.set_registered_model_alias(name, alias, version)
    logger.info(f"Promoted '{name}' version {version} to alias '{alias}'.")


def show_current(target: str) -> None:
    client = MlflowClient()
    name = registered_model_name(target)
    alias = get_settings().model_registry_alias
    mv = _current_alias_version(client, name, alias)

    if mv is None:
        print(f"No version currently holds the '{alias}' alias for '{name}'.")
    else:
        print(f"'{name}' alias '{alias}' -> version {mv.version} (run {mv.run_id})")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage MLflow model-registry promotions for AirCompressor-PdM."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list", help="List registered versions for a target."
    )
    list_parser.add_argument("target", choices=TARGET_COLS)

    promote_parser = subparsers.add_parser(
        "promote", help="Promote a version to the serving alias."
    )
    promote_parser.add_argument("target", choices=TARGET_COLS)
    promote_parser.add_argument("version")
    promote_parser.add_argument(
        "--alias", default=None, help="Defaults to settings.model_registry_alias."
    )
    promote_parser.add_argument(
        "--force", action="store_true", help="Skip the holdout_f1 threshold check."
    )

    current_parser = subparsers.add_parser(
        "current", help="Show which version currently serves a target."
    )
    current_parser.add_argument("target", choices=TARGET_COLS)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.command == "list":
        list_versions(args.target)
    elif args.command == "promote":
        promote(args.target, args.version, alias=args.alias, force=args.force)
    elif args.command == "current":
        show_current(args.target)


if __name__ == "__main__":
    main()
