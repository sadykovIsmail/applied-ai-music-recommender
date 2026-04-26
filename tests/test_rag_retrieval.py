from src.recommender import load_songs, recommend_songs_with_rag
from src.retrieval import MusicContextRetriever, build_query_from_user_prefs


def test_retriever_returns_context_for_pop_happy_query():
    songs = load_songs("data/songs.csv")
    retriever = MusicContextRetriever.from_song_catalog(songs)

    query = build_query_from_user_prefs({"genre": "pop", "mood": "happy", "energy": 0.85})
    contexts = retriever.retrieve(query, top_k=3)

    assert len(contexts) > 0
    joined_text = " ".join(c.text.lower() for c in contexts)
    assert "pop" in joined_text or "happy" in joined_text


def test_recommend_songs_with_rag_includes_citations():
    songs = load_songs("data/songs.csv")
    retriever = MusicContextRetriever.from_song_catalog(songs)

    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.38}
    ranked = recommend_songs_with_rag(prefs, songs, retriever, k=3, context_k=2)

    assert len(ranked) == 3
    for _, _, explanation in ranked:
        assert "Sources:" in explanation
