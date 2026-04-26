from pathlib import Path

from src.config import GeminiConfig


def test_gemini_config_reads_values_from_env_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-key\n"
        "GEMINI_MODEL=gemini-1.5-flash\n"
        "GEMINI_CLASS_NAME=MusicRecommenderAgent\n",
        encoding="utf-8",
    )

    config = GeminiConfig.from_env(str(env_file))

    assert config.api_key == "test-key"
    assert config.model == "gemini-1.5-flash"
    assert config.class_name == "MusicRecommenderAgent"
    assert config.is_configured is True


def test_gemini_config_defaults_when_key_missing(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")

    config = GeminiConfig.from_env(str(env_file))

    assert config.api_key == ""
    assert config.model == "gemini-1.5-flash"
    assert config.class_name == "MusicRecommenderAgent"
    assert config.is_configured is False
