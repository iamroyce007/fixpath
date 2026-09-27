"""Runtime settings read from the environment (.env is loaded if present)."""
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # optional in minimal installs
    pass

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    kit_dir: Path
    goal_trailing_period: bool
    validation_expected_values: bool
    provenance_min_ratio: float
    deeplink_min_score: float
    cache_sim_threshold: float
    llm_timeout_s: float
    score_w_retrieval: float
    score_w_provenance: float
    score_w_fixes: float


def load_settings() -> Settings:
    kit_dir = Path(os.environ.get("KIT_DIR", "kit"))
    if not kit_dir.is_absolute():
        kit_dir = ROOT / kit_dir
    return Settings(
        kit_dir=kit_dir,
        goal_trailing_period=_bool("GOAL_TRAILING_PERIOD", False),
        validation_expected_values=_bool("VALIDATION_EXPECTED_VALUES", True),
        provenance_min_ratio=_float("PROVENANCE_MIN_RATIO", 80.0),
        deeplink_min_score=_float("DEEPLINK_MIN_SCORE", 0.35),
        cache_sim_threshold=_float("CACHE_SIM_THRESHOLD", 0.88),
        llm_timeout_s=_float("LLM_TIMEOUT_S", 6.0),
        score_w_retrieval=_float("SCORE_W_RETRIEVAL", 0.5),
        score_w_provenance=_float("SCORE_W_PROVENANCE", 0.35),
        score_w_fixes=_float("SCORE_W_FIXES", 0.15),
    )


settings = load_settings()
