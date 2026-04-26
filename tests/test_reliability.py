import json

from src.recommender import load_songs
from src.retrieval import MusicContextRetriever
from src.specialization import MusicResponseSpecializer


class BrokenRetriever:
    def retrieve(self, query: str, top_k: int = 3):
        raise RuntimeError("index offline")


class BrokenSpecializer:
    def render_response(self, *args, **kwargs):
        raise RuntimeError("model gateway unavailable")


def test_reliable_runner_degrades_gracefully_on_retrieval_failure():
    from src.reliability import ReliableRecommendationRunner

    songs = load_songs("data/songs.csv")
    runner = ReliableRecommendationRunner(
        songs=songs,
        retriever=BrokenRetriever(),
        specializer=MusicResponseSpecializer(),
    )

    result = runner.run(
        profile_name="Pop Reliability Check",
        user_prefs={"genre": "pop", "mood": "happy", "energy": 0.85},
        mode="standard",
    )

    assert len(result.recommendations) == 5
    assert result.diagnostics.status == "degraded"
    assert result.diagnostics.retrieval_confidence == 0.0
    assert any("retrieval" in warning.lower() for warning in result.diagnostics.warnings)


def test_reliable_runner_uses_low_confidence_fallback_for_empty_catalog():
    from src.reliability import ReliableRecommendationRunner

    runner = ReliableRecommendationRunner(
        songs=[],
        retriever=MusicContextRetriever([]),
        specializer=MusicResponseSpecializer(),
    )

    result = runner.run(
        profile_name="Empty Catalog",
        user_prefs={"genre": "pop", "mood": "happy", "energy": 0.85},
        mode="vinyl_historian",
        use_few_shot=True,
    )

    assert result.recommendations == []
    assert result.diagnostics.fallback_used is True
    assert result.diagnostics.confidence_score < 0.5
    assert "Low-confidence fallback" in result.final_text


def test_reliable_runner_captures_generation_failures():
    from src.reliability import ReliableRecommendationRunner

    songs = load_songs("data/songs.csv")
    retriever = MusicContextRetriever.from_song_catalog(songs)
    runner = ReliableRecommendationRunner(
        songs=songs,
        retriever=retriever,
        specializer=BrokenSpecializer(),
    )

    result = runner.run(
        profile_name="Generation Failure",
        user_prefs={"genre": "lofi", "mood": "chill", "energy": "bad-number"},
        mode="vinyl_historian",
    )

    assert result.diagnostics.generation_confidence == 0.0
    assert any("invalid energy" in warning.lower() for warning in result.diagnostics.warnings)
    assert any("generation" in warning.lower() for warning in result.diagnostics.warnings)
    assert "Low-confidence fallback" in result.final_text


def test_run_demo_writes_reliability_artifact(tmp_path):
    from src.main import run_demo

    summary = run_demo(output_dir=str(tmp_path))

    artifact_path = tmp_path / "reliability_demo.json"
    assert artifact_path.exists()

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert len(artifact["profiles"]) == 4
    assert all("confidence_score" in entry["diagnostics"] for entry in artifact["profiles"])
    assert summary["profile_count"] == 4


def test_test_harness_writes_results_and_summary(tmp_path):
    from scripts.test_harness import run_harness

    report = run_harness(output_dir=str(tmp_path))

    results_path = tmp_path / "test_harness_results.json"
    summary_path = tmp_path / "test_harness_summary.md"

    assert results_path.exists()
    assert summary_path.exists()
    assert report["summary"]["scenario_count"] >= 5
    assert report["summary"]["passed"] >= 3
