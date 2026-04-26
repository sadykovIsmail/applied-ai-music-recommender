import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from scripts.test_harness import run_harness
from src.main import run_demo
from src.agent import AIDJAgent
from src.recommender import load_songs
from src.reliability import ReliableRecommendationRunner
from src.retrieval import MusicContextRetriever
from src.specialization import MusicResponseSpecializer


def ensure_showcase_artifacts(output_dir: str = "outputs") -> Dict[str, Any]:
    out_dir = Path(output_dir)
    reliability_path = out_dir / "reliability_demo.json"
    harness_path = out_dir / "test_harness_results.json"

    if not reliability_path.exists():
        run_demo(output_dir=str(out_dir))
    if not harness_path.exists():
        run_harness(output_dir=str(out_dir))

    return load_showcase_payload(output_dir=str(out_dir))


def refresh_showcase_artifacts(output_dir: str = "outputs") -> Dict[str, Any]:
    out_dir = Path(output_dir)
    run_demo(output_dir=str(out_dir))
    run_harness(output_dir=str(out_dir))
    return load_showcase_payload(output_dir=str(out_dir))


def load_showcase_payload(output_dir: str = "outputs") -> Dict[str, Any]:
    out_dir = Path(output_dir)
    reliability = _load_json(out_dir / "reliability_demo.json")
    specialization = _load_json(out_dir / "specialization_demo.json")
    harness = _load_json(out_dir / "test_harness_results.json")
    return {
        "reliability": reliability,
        "specialization": specialization,
        "harness": harness,
    }


def profile_names(payload: Dict[str, Any]) -> List[str]:
    return [profile["profile_name"] for profile in payload["reliability"]["profiles"]]


def get_profile(payload: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
    for profile in payload["reliability"]["profiles"]:
        if profile["profile_name"] == profile_name:
            return profile
    raise KeyError(f"Unknown profile: {profile_name}")


def dashboard_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    reliability_profiles = payload["reliability"]["profiles"]
    harness_summary = payload["harness"]["summary"]
    return {
        "profile_count": len(reliability_profiles),
        "average_confidence": payload["reliability"]["summary"]["average_confidence"],
        "fallback_runs": payload["reliability"]["summary"]["fallback_runs"],
        "harness_passed": harness_summary["passed"],
        "harness_total": harness_summary["scenario_count"],
    }


def confidence_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for profile in payload["reliability"]["profiles"]:
        diagnostics = profile["diagnostics"]
        rows.append(
            {
                "profile": profile["profile_name"],
                "confidence_score": diagnostics["confidence_score"],
                "retrieval_confidence": diagnostics["retrieval_confidence"],
                "rule_compliance_confidence": diagnostics["rule_compliance_confidence"],
                "generation_confidence": diagnostics["generation_confidence"],
                "fallback_used": diagnostics["fallback_used"],
                "status": diagnostics["status"],
            }
        )
    return rows


def harness_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for scenario in payload["harness"]["scenarios"]:
        rows.append(
            {
                "scenario": scenario["name"],
                "status": scenario["status"],
                "confidence_score": scenario["confidence_score"],
                "latency_ms": scenario["latency_ms"],
                "fallback_used": scenario["fallback_used"],
                "warnings": " | ".join(scenario["warnings"]) if scenario["warnings"] else "",
            }
        )
    return rows


def preset_profiles() -> List[Dict[str, Any]]:
    return [
        {
            "name": "High-Energy Pop Fan",
            "prefs": {"genre": "pop", "mood": "happy", "energy": 0.85},
        },
        {
            "name": "Chill Lofi Listener",
            "prefs": {"genre": "lofi", "mood": "chill", "energy": 0.38},
        },
        {
            "name": "Deep Intense Rock",
            "prefs": {"genre": "rock", "mood": "intense", "energy": 0.92},
        },
        {
            "name": "Contradictory Edge Case",
            "prefs": {"genre": "ambient", "mood": "sad", "energy": 0.90},
        },
    ]


@lru_cache(maxsize=1)
def get_runtime() -> Dict[str, Any]:
    songs = load_songs("data/songs.csv")
    retriever = MusicContextRetriever.from_multi_source(
        songs=songs,
        text_sources=[
            ("data/artist_context.md", "artist_notes", 0.9),
            ("data/genre_notes.md", "genre_notes", 0.8),
        ],
    )
    specializer = MusicResponseSpecializer()
    runner = ReliableRecommendationRunner(
        songs=songs,
        retriever=retriever,
        specializer=specializer,
    )
    agent = AIDJAgent(songs=songs, retriever=retriever)
    return {
        "songs": songs,
        "retriever": retriever,
        "specializer": specializer,
        "runner": runner,
        "agent": agent,
    }


def run_live_profile(
    profile_name: str,
    prefs: Dict[str, Any],
    mode: str = "vinyl_historian",
    use_few_shot: bool = True,
) -> Any:
    runtime = get_runtime()
    return runtime["runner"].run(
        profile_name=profile_name,
        user_prefs=prefs,
        mode=mode,
        use_few_shot=use_few_shot,
        k=5,
        context_k=2,
    )


def run_live_agent(user_intent: Dict[str, Any], playlist_size: int = 5) -> Any:
    runtime = get_runtime()
    return runtime["agent"].run(user_intent=user_intent, playlist_size=playlist_size)


def playlist_rows(result: Any) -> List[Dict[str, Any]]:
    rows = []
    for idx, song in enumerate(result.playlist, start=1):
        rows.append(
            {
                "slot": idx,
                "title": song["title"],
                "artist": song["artist"],
                "genre": song["genre"],
                "mood": song["mood"],
                "energy": song["energy"],
            }
        )
    return rows


def trace_rows(result: Any) -> List[Dict[str, Any]]:
    rows = []
    for step in result.trace:
        rows.append(
            {
                "step": step.step,
                "tool": step.tool,
                "input_summary": step.input_summary,
                "key_result": step.key_result,
                "decision_rationale": step.decision_rationale,
            }
        )
    return rows


def infer_profile_from_text(request_text: str) -> Dict[str, Any]:
    runtime = get_runtime()
    songs = runtime["songs"]
    text = request_text.strip().lower()

    genre_options = sorted(
        {
            str(song.get("genre", "")).strip().lower()
            for song in songs
            if str(song.get("genre", "")).strip()
        },
        key=len,
        reverse=True,
    )
    mood_options = sorted(
        {
            str(song.get("mood", "")).strip().lower()
            for song in songs
            if str(song.get("mood", "")).strip()
        },
        key=len,
        reverse=True,
    )

    genre = _match_option(text, genre_options) or _genre_synonym(text)
    mood = _match_option(text, mood_options) or _mood_synonym(text)
    energy = _infer_energy(text, genre=genre, mood=mood)

    return {
        "genre": genre or "",
        "mood": mood or "",
        "energy": energy,
        "normalized_request": text,
    }


def build_chat_markdown(session: Any) -> str:
    lines = [
        f"### {session.profile_name}",
        f"Confidence: `{session.diagnostics.confidence_score:.3f}`",
        "",
    ]

    if session.recommendations:
        lines.append("Top picks:")
        for idx, (song, score, _) in enumerate(session.recommendations[:3], start=1):
            lines.append(
                f"- `{idx}.` **{song['title']}** by {song['artist']} "
                f"({song['genre']}, {song['mood']}, energy {float(song['energy']):.2f}) "
                f"`score={score:.2f}`"
            )
    else:
        lines.append("No ranked picks were available.")

    lines.extend(
        [
            "",
            "Specialized response:",
            "",
            "```text",
            session.final_text,
            "```",
        ]
    )

    return "\n".join(lines)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _match_option(text: str, options: List[str]) -> str:
    for option in options:
        if option and option in text:
            return option
    return ""


def _genre_synonym(text: str) -> str:
    mapping = {
        "study": "lofi",
        "focus": "lofi",
        "workout": "pop",
        "club": "edm",
        "electronic": "edm",
        "metal": "metal",
        "indie": "indie pop",
        "calm": "ambient",
    }
    for token, genre in mapping.items():
        if token in text:
            return genre
    return ""


def _mood_synonym(text: str) -> str:
    mapping = {
        "study": "focused",
        "focus": "focused",
        "sad": "chill",
        "relax": "relaxed",
        "calm": "peaceful",
        "party": "happy",
        "workout": "intense",
        "energetic": "intense",
        "late night": "moody",
    }
    for token, mood in mapping.items():
        if token in text:
            return mood
    return ""


def _infer_energy(text: str, genre: str, mood: str) -> float:
    explicit = re.search(r"(\d(?:\.\d+)?)", text)
    if explicit:
        raw = float(explicit.group(1))
        if raw > 1.0:
            raw = raw / 10.0 if raw <= 10 else 1.0
        return max(0.0, min(1.0, raw))

    if any(token in text for token in ["high energy", "energetic", "workout", "intense", "party", "pump"]):
        return 0.88
    if any(token in text for token in ["calm", "peaceful", "ambient", "sleep", "soft"]):
        return 0.25
    if any(token in text for token in ["chill", "lofi", "study", "relax"]):
        return 0.38
    if any(token in text for token in ["mid", "medium", "balanced"]):
        return 0.55

    genre_defaults = {
        "lofi": 0.38,
        "ambient": 0.25,
        "rock": 0.88,
        "pop": 0.80,
        "edm": 0.92,
        "jazz": 0.42,
    }
    if genre in genre_defaults:
        return genre_defaults[genre]

    mood_defaults = {
        "intense": 0.90,
        "happy": 0.78,
        "focused": 0.42,
        "chill": 0.36,
        "relaxed": 0.34,
        "peaceful": 0.24,
        "moody": 0.62,
    }
    if mood in mood_defaults:
        return mood_defaults[mood]

    return 0.50
