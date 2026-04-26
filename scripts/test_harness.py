import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

from src.recommender import load_songs
from src.reliability import ReliableRecommendationRunner
from src.retrieval import MusicContextRetriever
from src.specialization import MusicResponseSpecializer


class FailingRetriever:
    def retrieve(self, query: str, top_k: int = 3):
        raise RuntimeError("simulated retrieval outage")


def run_harness(output_dir: str = "outputs") -> Dict[str, Any]:
    songs = load_songs("data/songs.csv")
    standard_retriever = MusicContextRetriever.from_multi_source(
        songs=songs,
        text_sources=[
            ("data/artist_context.md", "artist_notes", 0.9),
            ("data/genre_notes.md", "genre_notes", 0.8),
        ],
    )
    specializer = MusicResponseSpecializer()

    scenarios = [
        {
            "name": "happy_path_lofi",
            "runner": ReliableRecommendationRunner(songs, standard_retriever, specializer),
            "profile_name": "Chill Lofi Listener",
            "prefs": {"genre": "lofi", "mood": "chill", "energy": 0.38},
            "mode": "vinyl_historian",
            "expect_fallback": False,
        },
        {
            "name": "contradictory_preferences",
            "runner": ReliableRecommendationRunner(songs, standard_retriever, specializer),
            "profile_name": "Conflicting Ambient Sprint",
            "prefs": {"genre": "ambient", "mood": "sad", "energy": 0.95},
            "mode": "vinyl_historian",
            "expect_fallback": False,
        },
        {
            "name": "invalid_energy_input",
            "runner": ReliableRecommendationRunner(songs, standard_retriever, specializer),
            "profile_name": "Invalid Energy",
            "prefs": {"genre": "pop", "mood": "happy", "energy": "not-a-number"},
            "mode": "standard",
            "expect_fallback": False,
        },
        {
            "name": "retrieval_failure",
            "runner": ReliableRecommendationRunner(songs, FailingRetriever(), specializer),
            "profile_name": "Retrieval Failure",
            "prefs": {"genre": "rock", "mood": "intense", "energy": 0.91},
            "mode": "standard",
            "expect_fallback": False,
        },
        {
            "name": "empty_catalog",
            "runner": ReliableRecommendationRunner([], MusicContextRetriever([]), specializer),
            "profile_name": "Empty Catalog",
            "prefs": {"genre": "pop", "mood": "happy", "energy": 0.82},
            "mode": "vinyl_historian",
            "expect_fallback": True,
        },
    ]

    scenario_results: List[Dict[str, Any]] = []
    for scenario in scenarios:
        started = perf_counter()
        session = scenario["runner"].run(
            profile_name=scenario["profile_name"],
            user_prefs=scenario["prefs"],
            mode=scenario["mode"],
            use_few_shot=True,
        )
        latency_ms = round((perf_counter() - started) * 1000, 2)

        quality_checks = {
            "has_output": bool(session.final_text.strip()),
            "confidence_reported": session.diagnostics.confidence_score >= 0.0,
            "fallback_expectation_met": session.diagnostics.fallback_used == scenario["expect_fallback"],
            "status_is_graceful": session.diagnostics.status in {"ok", "degraded"},
        }
        passed = all(quality_checks.values())

        scenario_results.append(
            {
                "name": scenario["name"],
                "status": "pass" if passed else "fail",
                "latency_ms": latency_ms,
                "confidence_score": session.diagnostics.confidence_score,
                "warnings": session.diagnostics.warnings,
                "fallback_used": session.diagnostics.fallback_used,
                "quality_checks": quality_checks,
            }
        )

    passed = sum(1 for result in scenario_results if result["status"] == "pass")
    report = {
        "summary": {
            "scenario_count": len(scenario_results),
            "passed": passed,
            "failed": len(scenario_results) - passed,
            "average_confidence": round(
                sum(result["confidence_score"] for result in scenario_results) / len(scenario_results),
                3,
            ),
        },
        "scenarios": scenario_results,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_harness_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "test_harness_summary.md").write_text(_to_markdown(report), encoding="utf-8")
    return report


def _to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Reliability Test Harness Summary",
        "",
        f"- Scenarios: {report['summary']['scenario_count']}",
        f"- Passed: {report['summary']['passed']}",
        f"- Failed: {report['summary']['failed']}",
        f"- Average confidence: {report['summary']['average_confidence']:.3f}",
        "",
        "| Scenario | Status | Confidence | Latency (ms) | Fallback |",
        "|---|---|---:|---:|---|",
    ]

    for scenario in report["scenarios"]:
        lines.append(
            "| {name} | {status} | {confidence:.3f} | {latency:.2f} | {fallback} |".format(
                name=scenario["name"],
                status=scenario["status"],
                confidence=scenario["confidence_score"],
                latency=scenario["latency_ms"],
                fallback="yes" if scenario["fallback_used"] else "no",
            )
        )

    return "\n".join(lines)


if __name__ == "__main__":
    print(json.dumps(run_harness(), indent=2))
