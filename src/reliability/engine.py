from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

from src.recommender import recommend_songs
from src.retrieval import build_query_from_user_prefs, format_context_citations


RecommendationRow = tuple[dict, float, str]


@dataclass
class ReliabilityLog:
    level: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReliabilityDiagnostics:
    status: str
    confidence_score: float
    retrieval_confidence: float
    rule_compliance_confidence: float
    generation_confidence: float
    warnings: List[str]
    fallback_used: bool
    logs: List[ReliabilityLog]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["logs"] = [asdict(log) for log in self.logs]
        return payload


@dataclass
class RecommendationSession:
    profile_name: str
    user_prefs: Dict[str, Any]
    recommendations: List[RecommendationRow]
    contexts: List[Any]
    standard_text: str
    specialized_text: str
    final_text: str
    diagnostics: ReliabilityDiagnostics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "user_prefs": self.user_prefs,
            "recommendations": [
                {
                    "song": song,
                    "score": score,
                    "explanation": explanation,
                }
                for song, score, explanation in self.recommendations
            ],
            "contexts": [
                {
                    "chunk_id": getattr(ctx, "chunk_id", ""),
                    "source": getattr(ctx, "source", ""),
                    "score": getattr(ctx, "score", 0.0),
                    "source_type": getattr(ctx, "source_type", ""),
                    "trust_level": getattr(ctx, "trust_level", 0.0),
                }
                for ctx in self.contexts
            ],
            "standard_text": self.standard_text,
            "specialized_text": self.specialized_text,
            "final_text": self.final_text,
            "diagnostics": self.diagnostics.to_dict(),
        }


class ReliableRecommendationRunner:
    def __init__(
        self,
        songs: Sequence[Dict[str, Any]],
        retriever: Any,
        specializer: Any,
        low_confidence_threshold: float = 0.5,
    ):
        self.songs = list(songs)
        self.retriever = retriever
        self.specializer = specializer
        self.low_confidence_threshold = low_confidence_threshold

    def run(
        self,
        profile_name: str,
        user_prefs: Dict[str, Any],
        mode: str = "standard",
        use_few_shot: bool = False,
        k: int = 5,
        context_k: int = 2,
    ) -> RecommendationSession:
        warnings: List[str] = []
        logs: List[ReliabilityLog] = []
        prefs = self._normalize_user_prefs(user_prefs, warnings, logs)

        contexts = self._safe_retrieve(prefs, warnings, logs, top_k=context_k)
        recommendations = self._build_recommendations(prefs, contexts, warnings, logs, k=k)
        standard_text, specialized_text, generation_confidence = self._render_outputs(
            profile_name=profile_name,
            recommendations=recommendations,
            mode=mode,
            use_few_shot=use_few_shot,
            warnings=warnings,
            logs=logs,
        )

        retrieval_confidence = self._retrieval_confidence(contexts)
        rule_confidence = self._rule_compliance_confidence(prefs, recommendations)
        confidence_score = round(
            (retrieval_confidence + rule_confidence + generation_confidence) / 3.0,
            3,
        )

        fallback_used = (
            not recommendations
            or generation_confidence == 0.0
            or confidence_score < self.low_confidence_threshold
        )
        final_text = specialized_text
        if fallback_used:
            final_text = self._build_low_confidence_fallback(profile_name, recommendations, warnings)
            logs.append(
                ReliabilityLog(
                    level="WARN",
                    message="Applied low-confidence fallback output.",
                    context={"profile_name": profile_name},
                )
            )

        status = "ok" if not warnings and not fallback_used else "degraded"
        diagnostics = ReliabilityDiagnostics(
            status=status,
            confidence_score=confidence_score,
            retrieval_confidence=retrieval_confidence,
            rule_compliance_confidence=rule_confidence,
            generation_confidence=generation_confidence,
            warnings=warnings,
            fallback_used=fallback_used,
            logs=logs,
        )

        return RecommendationSession(
            profile_name=profile_name,
            user_prefs=prefs,
            recommendations=recommendations,
            contexts=contexts,
            standard_text=standard_text,
            specialized_text=specialized_text,
            final_text=final_text,
            diagnostics=diagnostics,
        )

    def _normalize_user_prefs(
        self,
        user_prefs: Dict[str, Any],
        warnings: List[str],
        logs: List[ReliabilityLog],
    ) -> Dict[str, Any]:
        normalized = {
            "genre": str(user_prefs.get("genre", "")).strip().lower(),
            "mood": str(user_prefs.get("mood", "")).strip().lower(),
        }

        try:
            energy = float(user_prefs.get("energy", 0.5))
        except (TypeError, ValueError):
            energy = 0.5
            warnings.append("Invalid energy value received; defaulted to 0.50.")
            logs.append(
                ReliabilityLog(
                    level="WARN",
                    message="Invalid energy input was normalized.",
                    context={"energy": user_prefs.get("energy")},
                )
            )

        normalized["energy"] = max(0.0, min(1.0, energy))
        return normalized

    def _safe_retrieve(
        self,
        prefs: Dict[str, Any],
        warnings: List[str],
        logs: List[ReliabilityLog],
        top_k: int,
    ) -> List[Any]:
        if self.retriever is None:
            return []

        query = build_query_from_user_prefs(prefs)
        if not query:
            warnings.append("No meaningful retrieval query could be formed.")
            logs.append(
                ReliabilityLog(
                    level="WARN",
                    message="Skipped retrieval because the query was empty.",
                )
            )
            return []

        try:
            contexts = list(self.retriever.retrieve(query, top_k=top_k))
            logs.append(
                ReliabilityLog(
                    level="INFO",
                    message="Retrieved context for recommendation run.",
                    context={"query": query, "contexts": len(contexts)},
                )
            )
            return contexts
        except Exception as exc:  # pragma: no cover - exercised by tests with a fake retriever
            warnings.append(f"Retrieval failure encountered; continuing with local ranking only. ({exc})")
            logs.append(
                ReliabilityLog(
                    level="ERROR",
                    message="Retrieval failed during recommendation run.",
                    context={"error": str(exc), "query": query},
                )
            )
            return []

    def _build_recommendations(
        self,
        prefs: Dict[str, Any],
        contexts: List[Any],
        warnings: List[str],
        logs: List[ReliabilityLog],
        k: int,
    ) -> List[RecommendationRow]:
        if not self.songs:
            warnings.append("Song catalog is empty; no ranked recommendations are available.")
            logs.append(
                ReliabilityLog(
                    level="ERROR",
                    message="Recommendation run received an empty song catalog.",
                )
            )
            return []

        ranked = recommend_songs(prefs, self.songs, k=k)
        if not ranked:
            warnings.append("Recommendation engine returned no matches.")
            logs.append(
                ReliabilityLog(
                    level="ERROR",
                    message="Recommendation engine returned no ranked results.",
                )
            )
            return []

        if not contexts:
            logs.append(
                ReliabilityLog(
                    level="WARN",
                    message="No retrieval context was available; explanations remain local-only.",
                )
            )
            return ranked

        citation_line = format_context_citations(contexts)
        grounded: List[RecommendationRow] = []
        for song, score, explanation in ranked:
            grounded.append((song, score, f"{explanation}; {citation_line}"))
        return grounded

    def _render_outputs(
        self,
        profile_name: str,
        recommendations: List[RecommendationRow],
        mode: str,
        use_few_shot: bool,
        warnings: List[str],
        logs: List[ReliabilityLog],
    ) -> tuple[str, str, float]:
        try:
            standard_text = self.specializer.render_response(
                profile_name,
                recommendations,
                mode="standard",
            )
            specialized_mode = mode if mode in {"standard", "vinyl_historian"} else "vinyl_historian"
            specialized_text = self.specializer.render_response(
                profile_name,
                recommendations,
                mode=specialized_mode,
                use_few_shot=use_few_shot,
            )
            confidence = self._generation_confidence(specialized_text, specialized_mode)
            logs.append(
                ReliabilityLog(
                    level="INFO",
                    message="Rendered recommendation outputs.",
                    context={"mode": specialized_mode, "generation_confidence": confidence},
                )
            )
            return standard_text, specialized_text, confidence
        except Exception as exc:  # pragma: no cover - exercised by tests with a fake specializer
            warnings.append(f"Generation failure encountered; using fallback text only. ({exc})")
            logs.append(
                ReliabilityLog(
                    level="ERROR",
                    message="Output rendering failed during recommendation run.",
                    context={"error": str(exc)},
                )
            )
            return "", "", 0.0

    def _retrieval_confidence(self, contexts: Sequence[Any]) -> float:
        if not contexts:
            return 0.0
        avg = sum(float(getattr(ctx, "score", 0.0)) for ctx in contexts) / len(contexts)
        return round(max(0.0, min(1.0, avg)), 3)

    def _rule_compliance_confidence(
        self,
        prefs: Dict[str, Any],
        recommendations: Sequence[RecommendationRow],
    ) -> float:
        if not recommendations:
            return 0.0

        checks = []
        for song, _, _ in recommendations[:3]:
            genre_score = 1.0 if prefs["genre"] and song.get("genre", "").lower() == prefs["genre"] else 0.5
            mood_score = 1.0 if prefs["mood"] and song.get("mood", "").lower() == prefs["mood"] else 0.5
            energy_score = 1.0 - abs(float(song.get("energy", 0.5)) - float(prefs["energy"]))
            checks.append((genre_score + mood_score + energy_score) / 3.0)

        avg = sum(checks) / len(checks)
        return round(max(0.0, min(1.0, avg)), 3)

    def _generation_confidence(self, rendered_text: str, mode: str) -> float:
        if not rendered_text.strip():
            return 0.0

        if mode == "standard":
            return 1.0

        persona_score = 1.0
        scorer = getattr(self.specializer, "persona_consistency_score", None)
        if callable(scorer):
            persona_score = float(scorer(rendered_text, mode=mode))
        return round(max(0.0, min(1.0, persona_score)), 3)

    def _build_low_confidence_fallback(
        self,
        profile_name: str,
        recommendations: Sequence[RecommendationRow],
        warnings: Sequence[str],
    ) -> str:
        lines = [
            f"Low-confidence fallback for {profile_name}",
            "The system detected unreliable inputs or weak supporting evidence.",
        ]

        if recommendations:
            song, score, explanation = recommendations[0]
            lines.append(
                f"Best available pick: {song['title']} by {song['artist']} "
                f"(score={score:.2f})"
            )
            lines.append(f"Why it still surfaced: {explanation}")
        else:
            lines.append("No tracks available for this request.")

        if warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in warnings)

        return "\n".join(lines)
