from dataclasses import dataclass
from typing import List, Sequence, Tuple


RecommendationRow = Tuple[dict, float, str]


@dataclass
class PromptProfile:
    mode: str
    persona_header: str
    required_markers: List[str]


class MusicResponseSpecializer:
    """Render recommendation outputs in either standard or specialized persona modes."""

    def __init__(self):
        self._profiles = {
            "standard": PromptProfile(
                mode="standard",
                persona_header="STANDARD RECOMMENDER",
                required_markers=[],
            ),
            "vinyl_historian": PromptProfile(
                mode="vinyl_historian",
                persona_header="VINYL HISTORIAN",
                required_markers=["Persona:", "Curation Rules:", "Picks:", "Constraint Check:"],
            ),
        }

    def render_response(
        self,
        profile_name: str,
        recommendations: Sequence[RecommendationRow],
        mode: str = "standard",
        use_few_shot: bool = False,
    ) -> str:
        if mode not in self._profiles:
            raise ValueError(f"Unsupported specialization mode: {mode}")

        if mode == "standard":
            return self._render_standard(profile_name, recommendations)

        return self._render_vinyl_historian(profile_name, recommendations, use_few_shot=use_few_shot)

    def persona_consistency_score(self, text: str, mode: str = "vinyl_historian") -> float:
        if mode not in self._profiles:
            raise ValueError(f"Unsupported specialization mode: {mode}")

        required = self._profiles[mode].required_markers
        if not required:
            return 1.0

        hits = sum(1 for marker in required if marker in text)
        return hits / len(required)

    def schema_compliant(self, text: str, mode: str = "vinyl_historian") -> bool:
        return self.persona_consistency_score(text, mode=mode) >= 1.0

    def _render_standard(self, profile_name: str, recommendations: Sequence[RecommendationRow]) -> str:
        lines = [f"Profile: {profile_name}"]
        if not recommendations:
            lines.append("No tracks available for this request.")
            return "\n".join(lines)

        for idx, (song, score, explanation) in enumerate(recommendations, 1):
            lines.append(f"{idx}. {song['title']} by {song['artist']} ({song['genre']}) score={score:.2f}")
            lines.append(f"   why: {explanation}")

        return "\n".join(lines)

    def _render_vinyl_historian(
        self,
        profile_name: str,
        recommendations: Sequence[RecommendationRow],
        use_few_shot: bool,
    ) -> str:
        # Constrained schema with markers enables measurable compliance checks.
        lines = [
            "VINYL HISTORIAN // CURATION NOTE",
            f"Persona: Analog-first archivist for {profile_name}",
            "Curation Rules: preserve mood arc; prioritize era texture; avoid abrupt energy leaps",
            "Picks:",
        ]

        if not recommendations:
            lines.append("- No tracks available")
            lines.append("Constraint Check: PASS (handled empty input)")
            return "\n".join(lines)

        exemplar = ""
        if use_few_shot:
            exemplar = (
                "Few-shot pattern:\n"
                "- Example pick format: [Era Guess] Title - rationale - Constraint Check: PASS\n"
            )
            lines.append(exemplar.rstrip())

        for idx, (song, score, explanation) in enumerate(recommendations, 1):
            era_guess = self._guess_era(float(song.get("tempo_bpm", 100)))
            constraint = "PASS" if float(song.get("energy", 0.5)) <= 1.0 else "WARN"
            lines.append(
                f"- [{era_guess}] {idx}. {song['title']} by {song['artist']} | "
                f"Rationale: {explanation} | Constraint Check: {constraint}"
            )

        lines.append("Constraint Check: PASS")
        return "\n".join(lines)

    def _guess_era(self, tempo_bpm: float) -> str:
        if tempo_bpm < 85:
            return "Late-Night Tape Era"
        if tempo_bpm < 120:
            return "Mid-Tempo Vinyl Era"
        return "Festival Pressing Era"
