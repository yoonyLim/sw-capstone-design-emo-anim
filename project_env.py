from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent
_ENV_LOADED = False


def load_env(env_path=None):
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    path = Path(env_path) if env_path else PROJECT_DIR / ".env"
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    _ENV_LOADED = True


def get_str(name, default=""):
    load_env()
    return os.environ.get(name, default)


def get_int(name, default):
    value = get_str(name, str(default))
    return int(value)


def get_path(name, default=None):
    value = get_str(name, "" if default is None else str(default))
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path
