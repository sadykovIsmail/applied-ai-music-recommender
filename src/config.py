from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict


@dataclass
class GeminiConfig:
    api_key: str
    model: str
    class_name: str

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "GeminiConfig":
        file_vars = _read_env_file(env_path)

        api_key = os.getenv("GEMINI_API_KEY", file_vars.get("GEMINI_API_KEY", "")).strip()
        model = os.getenv("GEMINI_MODEL", file_vars.get("GEMINI_MODEL", "gemini-1.5-flash")).strip()
        class_name = os.getenv(
            "GEMINI_CLASS_NAME",
            file_vars.get("GEMINI_CLASS_NAME", "MusicRecommenderAgent"),
        ).strip()

        return cls(api_key=api_key, model=model, class_name=class_name)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


def _read_env_file(env_path: str) -> Dict[str, str]:
    path = Path(env_path)
    if not path.exists():
        return {}

    parsed: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip().strip('"').strip("'")

    return parsed
