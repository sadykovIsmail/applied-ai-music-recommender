from dataclasses import dataclass
import re
from typing import Dict, List, Sequence


@dataclass
class RetrievedContext:
    """One retrievable context chunk with provenance metadata."""

    chunk_id: str
    text: str
    source: str
    score: float


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
            score = overlap / max(len(query_tokens), 1)
            scored.append(
                RetrievedContext(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    score=score,
                )
            )

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]


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
