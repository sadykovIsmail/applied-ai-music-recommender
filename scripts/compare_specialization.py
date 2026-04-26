import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.recommender import load_songs, recommend_songs
from src.specialization import MusicResponseSpecializer


Profile = Tuple[str, Dict]


def evaluate_specialization_delta(profiles: Sequence[Profile], k: int = 3) -> Dict:
    songs = load_songs("data/songs.csv")
    specializer = MusicResponseSpecializer()

    baseline_scores: List[float] = []
    specialized_scores: List[float] = []

    for profile_name, prefs in profiles:
        ranked = recommend_songs(prefs, songs, k=k)

        baseline_text = specializer.render_response(profile_name, ranked, mode="standard")
        specialized_text = specializer.render_response(
            profile_name,
            ranked,
            mode="vinyl_historian",
            use_few_shot=True,
        )

        baseline_scores.append(1.0 if specializer.schema_compliant(baseline_text, mode="vinyl_historian") else 0.0)
        specialized_scores.append(1.0 if specializer.schema_compliant(specialized_text, mode="vinyl_historian") else 0.0)

    baseline_rate = _avg(baseline_scores)
    specialized_rate = _avg(specialized_scores)

    return {
        "baseline": {"schema_compliance_rate": baseline_rate},
        "specialized_few_shot": {"schema_compliance_rate": specialized_rate},
        "delta": {"schema_compliance_rate": specialized_rate - baseline_rate},
        "profiles_evaluated": len(profiles),
    }


def run_default_specialization_eval() -> Dict:
    profiles: List[Profile] = [
        ("Chill Lofi Listener", {"genre": "lofi", "mood": "chill", "energy": 0.38}),
        ("High-Energy Pop Fan", {"genre": "pop", "mood": "happy", "energy": 0.85}),
        ("Deep Intense Rock", {"genre": "rock", "mood": "intense", "energy": 0.92}),
    ]

    report = evaluate_specialization_delta(profiles=profiles, k=3)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "specialization_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "specialization_summary.md").write_text(_to_markdown(report), encoding="utf-8")

    return report


def _to_markdown(report: Dict) -> str:
    return (
        "# Specialization Evaluation\n\n"
        f"Profiles evaluated: {report['profiles_evaluated']}\n\n"
        "| Mode | Schema Compliance |\n"
        "|---|---:|\n"
        f"| Baseline | {report['baseline']['schema_compliance_rate']:.2f} |\n"
        f"| Specialized (few-shot) | {report['specialized_few_shot']['schema_compliance_rate']:.2f} |\n"
        f"| Delta | {report['delta']['schema_compliance_rate']:.2f} |\n"
    )


def _avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


if __name__ == "__main__":
    print(json.dumps(run_default_specialization_eval(), indent=2))
