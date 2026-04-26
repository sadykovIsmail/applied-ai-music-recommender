from src.recommender import load_songs, recommend_songs
from src.specialization.specializer import MusicResponseSpecializer
from scripts.compare_specialization import evaluate_specialization_delta


def _sample_recommendations():
    songs = load_songs("data/songs.csv")
    ranked = recommend_songs({"genre": "lofi", "mood": "chill", "energy": 0.38}, songs, k=3)
    return ranked


def test_specialized_output_differs_from_standard_output():
    ranked = _sample_recommendations()
    specializer = MusicResponseSpecializer()

    standard_text = specializer.render_response("Chill Lofi Listener", ranked, mode="standard")
    specialized_text = specializer.render_response("Chill Lofi Listener", ranked, mode="vinyl_historian")

    assert standard_text != specialized_text
    assert "VINYL HISTORIAN" in specialized_text


def test_few_shot_constraints_improve_schema_compliance():
    report = evaluate_specialization_delta(
        profiles=[
            ("Chill Lofi Listener", {"genre": "lofi", "mood": "chill", "energy": 0.38}),
            ("High-Energy Pop Fan", {"genre": "pop", "mood": "happy", "energy": 0.85}),
        ],
        k=3,
    )

    assert report["baseline"]["schema_compliance_rate"] < report["specialized_few_shot"]["schema_compliance_rate"]
    assert report["delta"]["schema_compliance_rate"] > 0.0


def test_persona_drift_detector_flags_generic_text():
    specializer = MusicResponseSpecializer()

    generic_text = "Here are some recommendations based on your profile."
    assert specializer.persona_consistency_score(generic_text, mode="vinyl_historian") < 0.5


def test_specializer_handles_empty_recommendations_gracefully():
    specializer = MusicResponseSpecializer()

    out = specializer.render_response("Empty Case", [], mode="vinyl_historian", use_few_shot=True)
    assert "No tracks available" in out
