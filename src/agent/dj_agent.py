from dataclasses import dataclass
from typing import Dict, List, Sequence

from src.recommender import recommend_songs
from src.retrieval import MusicContextRetriever, build_query_from_user_prefs


@dataclass
class AgentTraceStep:
    step: int
    tool: str
    input_summary: str
    key_result: str
    decision_rationale: str


@dataclass
class AgentResult:
    playlist: List[Dict]
    trace: List[AgentTraceStep]
    revised: bool
    warnings: List[str]
    max_energy_jump_seen: float


class AIDJAgent:
    """Planner-executor-evaluator loop for playlist generation."""

    def __init__(
        self,
        songs: Sequence[Dict],
        retriever: MusicContextRetriever,
        max_iterations: int = 2,
        max_energy_jump: float = 0.35,
    ):
        self.songs = list(songs)
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.max_energy_jump = max_energy_jump

    def run(self, user_intent: Dict, playlist_size: int = 5) -> AgentResult:
        trace: List[AgentTraceStep] = []
        warnings: List[str] = []
        revised = False

        plan = self._plan(user_intent=user_intent, playlist_size=playlist_size)
        trace.append(
            AgentTraceStep(
                step=1,
                tool="planner",
                input_summary=str(user_intent),
                key_result=f"playlist_size={plan['playlist_size']} genre={plan['genre']}",
                decision_rationale="Turn user intent into executable recommendation constraints.",
            )
        )

        candidates = self._fetch_candidate_tracks(plan)
        trace.append(
            AgentTraceStep(
                step=2,
                tool="fetch_candidate_tracks",
                input_summary=f"genre={plan['genre']} mood={plan['mood']} energy={plan['energy']}",
                key_result=f"candidates={len(candidates)}",
                decision_rationale="Retrieve ranked candidates before flow validation.",
            )
        )

        if not candidates:
            warnings.append("No candidates found for the requested constraints.")
            return AgentResult(
                playlist=[],
                trace=trace,
                revised=False,
                warnings=warnings,
                max_energy_jump_seen=0.0,
            )

        playlist = candidates[: playlist_size]

        # Execute iterative evaluation-revision loop.
        max_jump_seen = 0.0
        for iteration in range(1, self.max_iterations + 1):
            max_jump_seen = self._max_energy_jump(playlist)
            is_smooth = max_jump_seen <= self.max_energy_jump

            trace.append(
                AgentTraceStep(
                    step=2 + iteration,
                    tool="energy_flow_checker",
                    input_summary=f"iteration={iteration} threshold={self.max_energy_jump:.2f}",
                    key_result=f"max_jump={max_jump_seen:.2f}",
                    decision_rationale="Check energy continuity between adjacent tracks.",
                )
            )

            if is_smooth:
                break

            revised = True
            playlist = self._revise_playlist(playlist, candidates, playlist_size)
        else:
            warnings.append("Reached max iterations while trying to smooth energy transitions.")

        genre_ok = self._check_genre_balance(playlist)
        trace.append(
            AgentTraceStep(
                step=3 + self.max_iterations,
                tool="genre_balance_checker",
                input_summary=f"tracks={len(playlist)}",
                key_result=f"balanced={genre_ok}",
                decision_rationale="Avoid over-concentration when alternatives exist.",
            )
        )

        if not genre_ok:
            revised = True
            playlist = self._enforce_genre_balance(playlist, candidates)

        return AgentResult(
            playlist=playlist,
            trace=trace,
            revised=revised,
            warnings=warnings,
            max_energy_jump_seen=self._max_energy_jump(playlist),
        )

    def _plan(self, user_intent: Dict, playlist_size: int) -> Dict:
        genre = str(user_intent.get("genre", "")).strip().lower()
        mood = str(user_intent.get("mood", "")).strip().lower()
        energy = float(user_intent.get("energy", 0.5))
        energy = max(0.0, min(1.0, energy))

        return {
            "genre": genre,
            "mood": mood,
            "energy": energy,
            "playlist_size": max(1, int(playlist_size)),
        }

    def _fetch_candidate_tracks(self, plan: Dict) -> List[Dict]:
        # First pass: strict genre + mood intent.
        prefs = {
            "genre": plan["genre"],
            "mood": plan["mood"],
            "energy": plan["energy"],
        }

        ranked = recommend_songs(prefs, self.songs, k=max(plan["playlist_size"] * 3, 10))
        candidates = [song for song, _, _ in ranked]

        if not candidates:
            return []

        # Fallback for contradictory/rare genre intent: relax genre constraint.
        if plan["genre"] and all(song.get("genre", "").lower() != plan["genre"] for song in candidates[: plan["playlist_size"]]):
            relaxed = {"genre": "", "mood": plan["mood"], "energy": plan["energy"]}
            relaxed_ranked = recommend_songs(relaxed, self.songs, k=max(plan["playlist_size"] * 3, 10))
            candidates = [song for song, _, _ in relaxed_ranked]

        # Retrieval call remains explicit and traceable for agentic behavior.
        query = build_query_from_user_prefs(prefs)
        _ = self.retriever.retrieve(query, top_k=3)

        return self._dedupe_by_id(candidates)

    def _revise_playlist(self, playlist: List[Dict], candidates: List[Dict], playlist_size: int) -> List[Dict]:
        # Smoothing heuristic: sort by energy to reduce adjacent jumps.
        smoothed = sorted(playlist, key=lambda s: float(s.get("energy", 0.0)))
        if len(smoothed) >= playlist_size:
            return smoothed[:playlist_size]

        used_ids = {int(song["id"]) for song in smoothed}
        for song in candidates:
            sid = int(song["id"])
            if sid in used_ids:
                continue
            smoothed.append(song)
            used_ids.add(sid)
            if len(smoothed) >= playlist_size:
                break

        return smoothed

    def _check_genre_balance(self, playlist: List[Dict]) -> bool:
        genres = {song.get("genre", "").lower() for song in playlist if song.get("genre")}
        return len(genres) >= 2 or len(playlist) <= 1

    def _enforce_genre_balance(self, playlist: List[Dict], candidates: List[Dict]) -> List[Dict]:
        if not playlist:
            return playlist

        current_genres = {song.get("genre", "").lower() for song in playlist if song.get("genre")}
        if len(current_genres) >= 2:
            return playlist

        for song in candidates:
            candidate_genre = song.get("genre", "").lower()
            if candidate_genre and candidate_genre not in current_genres:
                return playlist[:-1] + [song]

        return playlist

    def _max_energy_jump(self, playlist: List[Dict]) -> float:
        if len(playlist) <= 1:
            return 0.0

        max_jump = 0.0
        for idx in range(1, len(playlist)):
            prev_energy = float(playlist[idx - 1].get("energy", 0.0))
            curr_energy = float(playlist[idx].get("energy", 0.0))
            jump = abs(curr_energy - prev_energy)
            if jump > max_jump:
                max_jump = jump
        return max_jump

    def _dedupe_by_id(self, songs: List[Dict]) -> List[Dict]:
        unique: List[Dict] = []
        seen_ids = set()
        for song in songs:
            sid = int(song["id"])
            if sid in seen_ids:
                continue
            unique.append(song)
            seen_ids.add(sid)
        return unique
