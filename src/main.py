Command line runner for the Music Recommender Simulation.

Run from the project root:
    python -m src.main

import os
import json
from pathlib import Path
from src.agent import AIDJAgent
from src.config import GeminiConfig
from src.recommender import load_songs, recommend_songs_with_rag
from src.retrieval import MusicContextRetriever
from src.specialization import MusicResponseSpecializer


def print_recommendations(profile_name: str, recommendations) -> None:
    """Print a formatted block of recommendations for a given profile."""
    print(f"\n{'=' * 52}")
    print(f"  Profile: {profile_name}")
    print(f"{'=' * 52}")
    for i, (song, score, explanation) in enumerate(recommendations, 1):
        print(f"\n  {i}. {song['title']}  by {song['artist']}")
        print(f"     Genre: {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']:.2f}")
        print(f"     Score: {score:.2f}")
        print(f"     Why:   {explanation}")


def print_specialized_preview(text: str, mode: str) -> None:
    print(f"\n{'-' * 52}")
    print(f"  Specialized Output ({mode})")
    print(f"{'-' * 52}")
    print(text)


def save_specialization_artifact(mode: str, standard_text: str, specialized_text: str) -> None:
    out = Path("outputs/specialization_demo.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "standard": standard_text,
        "specialized": specialized_text,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_agent_playlist(result) -> None:
    print(f"\n{'-' * 52}")
    print("  AI DJ Agent Playlist")
    print(f"{'-' * 52}")

    for i, song in enumerate(result.playlist, 1):
        print(f"  {i}. {song['title']} by {song['artist']} ({song['genre']}, energy={song['energy']:.2f})")

    if result.warnings:
        print("  Warnings:")
        for warning in result.warnings:
            print(f"   - {warning}")

    print("  Trace:")
    for step in result.trace:
        print(f"   - step={step.step} tool={step.tool} result={step.key_result}")


def save_agent_trace(result, output_path: str = "outputs/agent_trace_demo.json") -> None:
    out = Path(output_path)
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


def main() -> None:
    gemini = GeminiConfig.from_env()
    if gemini.is_configured:
        print(f"Gemini config loaded for class '{gemini.class_name}' using model '{gemini.model}'.")
    else:
        print("Gemini API key not set yet. Running with local retrieval mode.")

    specialization_mode = os.getenv("SPECIALIZATION_MODE", "vinyl_historian").strip().lower()
    use_few_shot = os.getenv("SPECIALIZATION_FEW_SHOT", "true").strip().lower() == "true"

    csv_path = os.path.join("data", "songs.csv")
    songs = load_songs(csv_path)
    retriever = MusicContextRetriever.from_multi_source(
        songs=songs,
        text_sources=[
            ("data/artist_context.md", "artist_notes", 0.9),
            ("data/genre_notes.md", "genre_notes", 0.8),
        ],
    )
    agent = AIDJAgent(songs=songs, retriever=retriever)
    specializer = MusicResponseSpecializer()
    print(f"Loaded {len(songs)} songs.")

    profiles = [
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

    for name, prefs in profiles:
        recs = recommend_songs_with_rag(prefs, songs, retriever, k=5, context_k=2)
        print_recommendations(name, recs)

        standard_text = specializer.render_response(name, recs, mode="standard")
        mode = specialization_mode if specialization_mode in {"standard", "vinyl_historian"} else "vinyl_historian"
        specialized_text = specializer.render_response(
            name,
            recs,
            mode=mode,
            use_few_shot=use_few_shot,
        )
        print_specialized_preview(specialized_text, mode=mode)
        save_specialization_artifact(mode=mode, standard_text=standard_text, specialized_text=specialized_text)

    # Agentic workflow demo run for observable planning/execution/evaluation trace.
    agent_result = agent.run(user_intent={"genre": "lofi", "mood": "chill", "energy": 0.38}, playlist_size=5)
    print_agent_playlist(agent_result)
    save_agent_trace(agent_result)

    print(f"\n{'=' * 52}\n")


if __name__ == "__main__":
    main()
