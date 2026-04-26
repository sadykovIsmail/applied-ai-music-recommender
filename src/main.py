"""Command line runner for the Music Recommender Simulation.

Run from the project root:
    python -m src.main
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agent import AIDJAgent
from src.config import GeminiConfig
from src.recommender import load_songs
from src.reliability import ReliableRecommendationRunner
from src.retrieval import MusicContextRetriever
from src.specialization import MusicResponseSpecializer


Profile = Tuple[str, Dict[str, Any]]


def print_recommendations(profile_name: str, recommendations) -> None:
    print(f"\n{'=' * 52}")
    print(f"  Profile: {profile_name}")
    print(f"{'=' * 52}")

    if not recommendations:
        print("  No ranked recommendations were produced.")
        return

    for idx, (song, score, explanation) in enumerate(recommendations, 1):
        print(f"\n  {idx}. {song['title']} by {song['artist']}")
        print(f"     Genre: {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']:.2f}")
        print(f"     Score: {score:.2f}")
        print(f"     Why:   {explanation}")


def print_specialized_preview(text: str, mode: str) -> None:
    print(f"\n{'-' * 52}")
    print(f"  Output Preview ({mode})")
    print(f"{'-' * 52}")
    print(text)


def print_reliability_dashboard(profile_name: str, diagnostics) -> None:
    print(f"\n{'-' * 52}")
    print(f"  Reliability Dashboard: {profile_name}")
    print(f"{'-' * 52}")
    print(f"  Status: {diagnostics.status}")
    print(f"  Confidence: {diagnostics.confidence_score:.3f}")
    print(f"  Retrieval: {diagnostics.retrieval_confidence:.3f}")
    print(f"  Rule Compliance: {diagnostics.rule_compliance_confidence:.3f}")
    print(f"  Generation: {diagnostics.generation_confidence:.3f}")
    print(f"  Fallback Used: {'yes' if diagnostics.fallback_used else 'no'}")
    if diagnostics.warnings:
        print("  Warnings:")
        for warning in diagnostics.warnings:
            print(f"   - {warning}")


def print_agent_playlist(result) -> None:
    print(f"\n{'-' * 52}")
    print("  AI DJ Agent Playlist")
    print(f"{'-' * 52}")

    for idx, song in enumerate(result.playlist, 1):
        print(f"  {idx}. {song['title']} by {song['artist']} ({song['genre']}, energy={song['energy']:.2f})")

    if result.warnings:
        print("  Warnings:")
        for warning in result.warnings:
            print(f"   - {warning}")

    print("  Trace:")
    for step in result.trace:
        print(f"   - step={step.step} tool={step.tool} result={step.key_result}")


def save_specialization_artifact(sessions, output_dir: str) -> None:
    out = Path(output_dir) / "specialization_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profiles": [
            {
                "profile_name": session.profile_name,
                "standard": session.standard_text,
                "specialized": session.specialized_text,
                "final_text": session.final_text,
            }
            for session in sessions
        ]
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_agent_trace(result, output_dir: str) -> None:
    out = Path(output_dir) / "agent_trace_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "revised": result.revised,
        "warnings": result.warnings,
        "max_energy_jump_seen": result.max_energy_jump_seen,
        "trace": [
            {
                "step": step.step,
                "tool": step.tool,
                "input_summary": step.input_summary,
                "key_result": step.key_result,
                "decision_rationale": step.decision_rationale,
            }
            for step in result.trace
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_reliability_artifact(
    sessions,
    agent_result,
    gemini: GeminiConfig,
    mode: str,
    output_dir: str,
) -> Dict[str, Any]:
    out = Path(output_dir) / "reliability_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "gemini_configured": gemini.is_configured,
        "mode": mode,
        "profiles": [session.to_dict() for session in sessions],
        "agent": {
            "playlist_size": len(agent_result.playlist),
            "revised": agent_result.revised,
            "warnings": agent_result.warnings,
            "max_energy_jump_seen": agent_result.max_energy_jump_seen,
        },
        "summary": {
            "profile_count": len(sessions),
            "average_confidence": round(
                sum(session.diagnostics.confidence_score for session in sessions) / max(len(sessions), 1),
                3,
            ),
            "fallback_runs": sum(1 for session in sessions if session.diagnostics.fallback_used),
        },
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload["summary"]


def build_profiles() -> List[Profile]:
    return [
        (
            "High-Energy Pop Fan",
            {"genre": "pop", "mood": "happy", "energy": 0.85},
        ),
        (
            "Chill Lofi Listener",
            {"genre": "lofi", "mood": "chill", "energy": 0.38},
        ),
        (
            "Deep Intense Rock",
            {"genre": "rock", "mood": "intense", "energy": 0.92},
        ),
        (
            "Edge Case: High-Energy but Sad (conflicting preferences)",
            {"genre": "ambient", "mood": "sad", "energy": 0.90},
        ),
    ]


def run_demo(output_dir: str = "outputs") -> Dict[str, Any]:
    gemini = GeminiConfig.from_env()
    specialization_mode = os.getenv("SPECIALIZATION_MODE", "vinyl_historian").strip().lower()
    mode = specialization_mode if specialization_mode in {"standard", "vinyl_historian"} else "vinyl_historian"
    use_few_shot = os.getenv("SPECIALIZATION_FEW_SHOT", "true").strip().lower() == "true"

    songs = load_songs(os.path.join("data", "songs.csv"))
    retriever = MusicContextRetriever.from_multi_source(
        songs=songs,
        text_sources=[
            ("data/artist_context.md", "artist_notes", 0.9),
            ("data/genre_notes.md", "genre_notes", 0.8),
        ],
    )
    specializer = MusicResponseSpecializer()
    runner = ReliableRecommendationRunner(songs=songs, retriever=retriever, specializer=specializer)
    agent = AIDJAgent(songs=songs, retriever=retriever)

    sessions = []
    for profile_name, prefs in build_profiles():
        session = runner.run(
            profile_name=profile_name,
            user_prefs=prefs,
            mode=mode,
            use_few_shot=use_few_shot,
            k=5,
            context_k=2,
        )
        sessions.append(session)

    agent_result = agent.run(user_intent={"genre": "lofi", "mood": "chill", "energy": 0.38}, playlist_size=5)

    save_specialization_artifact(sessions, output_dir=output_dir)
    save_agent_trace(agent_result, output_dir=output_dir)
    summary = save_reliability_artifact(
        sessions=sessions,
        agent_result=agent_result,
        gemini=gemini,
        mode=mode,
        output_dir=output_dir,
    )

    return {
        "profile_count": len(sessions),
        "average_confidence": summary["average_confidence"],
        "fallback_runs": summary["fallback_runs"],
        "output_dir": str(Path(output_dir)),
    }


def main() -> None:
    gemini = GeminiConfig.from_env()
    if gemini.is_configured:
        print(f"Gemini config loaded for class '{gemini.class_name}' using model '{gemini.model}'.")
    else:
        print("Gemini API key not set yet. Running with local retrieval mode.")

    output_dir = "outputs"
    specialization_mode = os.getenv("SPECIALIZATION_MODE", "vinyl_historian").strip().lower()
    mode = specialization_mode if specialization_mode in {"standard", "vinyl_historian"} else "vinyl_historian"
    use_few_shot = os.getenv("SPECIALIZATION_FEW_SHOT", "true").strip().lower() == "true"

    songs = load_songs(os.path.join("data", "songs.csv"))
    retriever = MusicContextRetriever.from_multi_source(
        songs=songs,
        text_sources=[
            ("data/artist_context.md", "artist_notes", 0.9),
            ("data/genre_notes.md", "genre_notes", 0.8),
        ],
    )
    specializer = MusicResponseSpecializer()
    runner = ReliableRecommendationRunner(songs=songs, retriever=retriever, specializer=specializer)
    agent = AIDJAgent(songs=songs, retriever=retriever)

    print(f"Loaded {len(songs)} songs.")
    sessions = []
    for profile_name, prefs in build_profiles():
        session = runner.run(
            profile_name=profile_name,
            user_prefs=prefs,
            mode=mode,
            use_few_shot=use_few_shot,
            k=5,
            context_k=2,
        )
        sessions.append(session)
        print_recommendations(profile_name, session.recommendations)
        print_specialized_preview(session.final_text, mode=mode)
        print_reliability_dashboard(profile_name, session.diagnostics)

    agent_result = agent.run(user_intent={"genre": "lofi", "mood": "chill", "energy": 0.38}, playlist_size=5)
    print_agent_playlist(agent_result)

    save_specialization_artifact(sessions, output_dir=output_dir)
    save_agent_trace(agent_result, output_dir=output_dir)
    summary = save_reliability_artifact(
        sessions=sessions,
        agent_result=agent_result,
        gemini=gemini,
        mode=mode,
        output_dir=output_dir,
    )
    print(f"\nSaved artifacts to {output_dir} with average confidence {summary['average_confidence']:.3f}.")


if __name__ == "__main__":
    main()
