from src.settings import Settings, get_settings


def test_defaults():
    settings = Settings(_env_file=None)

    assert settings.env == "development"
    assert settings.api_key == "dev-local-key"
    assert settings.cors_allow_origins == []
    assert settings.mlflow_tracking_uri is None
    assert settings.max_readings_per_request == 500
    assert settings.model_registry_alias == "production"


def test_env_var_overrides(monkeypatch):
    monkeypatch.setenv("ACPDM_API_KEY", "super-secret")
    monkeypatch.setenv("ACPDM_ENV", "production")
    monkeypatch.setenv("ACPDM_MAX_READINGS_PER_REQUEST", "10")

    settings = Settings(_env_file=None)

    assert settings.api_key == "super-secret"
    assert settings.env == "production"
    assert settings.max_readings_per_request == 10


def test_cors_origins_parsed_from_comma_separated_env(monkeypatch):
    monkeypatch.setenv(
        "ACPDM_CORS_ALLOW_ORIGINS", '["https://a.example", "https://b.example"]'
    )

    settings = Settings(_env_file=None)

    assert settings.cors_allow_origins == ["https://a.example", "https://b.example"]


def test_env_file_is_loaded(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ACPDM_API_KEY=from-dotenv\n")
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.api_key == "from-dotenv"


def test_get_settings_is_cached(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("ACPDM_API_KEY", "cached-value")

    first = get_settings()
    monkeypatch.setenv("ACPDM_API_KEY", "different-value")
    second = get_settings()

    assert first is second
    assert second.api_key == "cached-value"

    get_settings.cache_clear()
