import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.recommender import load_songs
from src.retrieval import MusicContextRetriever, build_query_from_user_prefs


def compare_single_vs_multi_source(
    songs_csv_path: str,
    profiles: Sequence[Dict],
    text_sources: Sequence[Tuple[str, str, float]],
    k: int = 3,
    context_k: int = 4,
) -> Dict:
    songs = load_songs(songs_csv_path)

    single = MusicContextRetriever.from_song_catalog(songs)
    multi = MusicContextRetriever.from_multi_source(songs=songs, text_sources=text_sources)

    single_uniques = []
    multi_uniques = []

    for profile in profiles:
        query = build_query_from_user_prefs(profile)

        single_contexts = single.retrieve(query, top_k=context_k)
        multi_contexts = multi.retrieve(query, top_k=context_k)

        single_uniques.append(_count_unique_sources(single_contexts))
        multi_uniques.append(_count_unique_sources(multi_contexts))

    single_avg = _avg(single_uniques)
    multi_avg = _avg(multi_uniques)

    return {
        "profiles_evaluated": len(profiles),
        "single_source": {"avg_unique_sources": single_avg},
        "multi_source": {"avg_unique_sources": multi_avg},
        "delta": {"avg_unique_sources": multi_avg - single_avg},
        "params": {"k": k, "context_k": context_k},
    }


def run_default_evaluation() -> Dict:
    profiles = [
        {"genre": "pop", "mood": "happy", "energy": 0.85},
        {"genre": "lofi", "mood": "chill", "energy": 0.38},
        {"genre": "rock", "mood": "intense", "energy": 0.92},
    ]
    text_sources = [
        ("data/artist_context.md", "artist_notes", 0.9),
        ("data/genre_notes.md", "genre_notes", 0.8),
    ]

    report = compare_single_vs_multi_source(
        songs_csv_path="data/songs.csv",
        profiles=profiles,
        text_sources=text_sources,
        k=3,
        context_k=4,
    )

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "rag_enhancement_eval.json"
    md_path = out_dir / "rag_enhancement_summary.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    return report


def _avg(values: List[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _count_unique_sources(contexts) -> int:
    return len({ctx.source for ctx in contexts})


def _to_markdown(report: Dict) -> str:
    return (
        "# RAG Enhancement Evaluation\n\n"
        f"Profiles evaluated: {report['profiles_evaluated']}\n\n"
        "| Mode | Avg Unique Sources |\n"
        "|---|---:|\n"
        f"| Single-source | {report['single_source']['avg_unique_sources']:.2f} |\n"
        f"| Multi-source | {report['multi_source']['avg_unique_sources']:.2f} |\n"
        f"| Delta | {report['delta']['avg_unique_sources']:.2f} |\n"
    )


if __name__ == "__main__":
    result = run_default_evaluation()
    print(json.dumps(result, indent=2))
