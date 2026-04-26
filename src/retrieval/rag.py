from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Sequence, Tuple


@dataclass
class RetrievedContext:
    """One retrievable context chunk with provenance metadata."""

    chunk_id: str
    text: str
    source: str
    score: float
    source_type: str = "catalog"
    trust_level: float = 1.0


class MusicContextRetriever:
    """Simple lexical retriever for song and artist context."""

    def __init__(self, chunks: Sequence[RetrievedContext]):
        self._chunks = list(chunks)

    @classmethod
    def from_song_catalog(cls, songs: List[Dict]) -> "MusicContextRetriever":
        chunks: List[RetrievedContext] = []
        artist_map: Dict[str, Dict[str, set]] = {}

        for song in songs:
            song_id = song["id"]
            title = song["title"]
            artist = song["artist"]
            genre = song["genre"]
            mood = song["mood"]
            energy = float(song["energy"])
            tempo = float(song["tempo_bpm"])

            chunks.append(
                RetrievedContext(
                    chunk_id=f"song:{song_id}",
                    text=(
                        f"{title} by {artist} is a {genre} song with a {mood} mood, "
                        f"energy {energy:.2f}, and tempo {tempo:.0f} BPM."
                    ),
                    source="songs.csv",
                    score=0.0,
                    source_type="catalog",
                    trust_level=1.0,
                )
            )

            if artist not in artist_map:
                artist_map[artist] = {"genres": set(), "moods": set()}
            artist_map[artist]["genres"].add(genre)
            artist_map[artist]["moods"].add(mood)

        for artist, meta in artist_map.items():
            genres = ", ".join(sorted(meta["genres"]))
            moods = ", ".join(sorted(meta["moods"]))
            chunks.append(
                RetrievedContext(
                    chunk_id=f"artist:{artist.lower().replace(' ', '_')}",
                    text=f"{artist} commonly appears with genres {genres} and moods {moods}.",
                    source="songs.csv",
                    score=0.0,
                    source_type="catalog",
                    trust_level=1.0,
                )
            )

        return cls(chunks)

    @classmethod
    def from_multi_source(
        cls,
        songs: List[Dict],
        text_sources: Sequence[Tuple[str, str, float]],
    ) -> "MusicContextRetriever":
        base = cls.from_song_catalog(songs)
        chunks = list(base._chunks)

        for source_path, source_type, trust_level in text_sources:
            path = Path(source_path)
            if not path.exists():
                continue

            raw = path.read_text(encoding="utf-8")
            paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
            for idx, paragraph in enumerate(paragraphs, start=1):
                chunks.append(
                    RetrievedContext(
                        chunk_id=f"{source_type}:{path.stem}:{idx}",
                        text=paragraph,
                        source=str(path).replace("\\", "/"),
                        score=0.0,
                        source_type=source_type,
                        trust_level=float(trust_level),
                    )
                )

        return cls(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedContext]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for chunk in self._chunks:
            chunk_tokens = _tokenize(chunk.text)
            overlap = len(query_tokens.intersection(chunk_tokens))
            if overlap == 0:
                continue

            # Normalize by query size so tighter query matches rank higher.
            lexical = overlap / max(len(query_tokens), 1)
            score = lexical * _trust_weight(chunk.trust_level)
            scored.append(
                RetrievedContext(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    score=score,
                    source_type=chunk.source_type,
                    trust_level=chunk.trust_level,
                )
            )

        scored.sort(key=lambda c: c.score, reverse=True)

        # Prefer source diversity when possible so multi-source setups
        # can surface supporting evidence from custom data.
        diversified: List[RetrievedContext] = []
        seen_sources = set()
        for ctx in scored:
            if ctx.source in seen_sources:
                continue
            diversified.append(ctx)
            seen_sources.add(ctx.source)
            if len(diversified) >= top_k:
                return diversified

        for ctx in scored:
            if len(diversified) >= top_k:
                break
            if ctx in diversified:
                continue
            diversified.append(ctx)

        return diversified


def build_query_from_user_prefs(user_prefs: Dict) -> str:
    parts = [
        str(user_prefs.get("genre", "")),
        str(user_prefs.get("mood", "")),
        f"energy {user_prefs.get('energy', '')}",
    ]
    return " ".join(part for part in parts if part).strip()


def format_context_citations(contexts: Sequence[RetrievedContext]) -> str:
    if not contexts:
        return "No external context retrieved."

    refs = [f"[{ctx.chunk_id} via {ctx.source}, score={ctx.score:.2f}]" for ctx in contexts]
    return "Sources: " + ", ".join(refs)


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _trust_weight(trust_level: float) -> float:
    bounded = max(0.1, min(1.0, trust_level))
    return 0.75 + (0.25 * bounded)
