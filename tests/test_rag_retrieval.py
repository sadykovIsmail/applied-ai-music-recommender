from src.recommender import load_songs, recommend_songs_with_rag
from src.retrieval import MusicContextRetriever, build_query_from_user_prefs
from scripts.evaluate_rag_enhancement import compare_single_vs_multi_source


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


def test_multisource_retriever_returns_diverse_sources():
    songs = load_songs("data/songs.csv")
    retriever = MusicContextRetriever.from_multi_source(
        songs=songs,
        text_sources=[
            ("data/artist_context.md", "artist_notes", 0.9),
            ("data/genre_notes.md", "genre_notes", 0.8),
        ],
    )

    query = build_query_from_user_prefs({"genre": "lofi", "mood": "chill", "energy": 0.40})
    contexts = retriever.retrieve(query, top_k=5)

    assert len(contexts) >= 2
    sources = {ctx.source for ctx in contexts}
    assert "songs.csv" in sources
    assert any(src.endswith("artist_context.md") or src.endswith("genre_notes.md") for src in sources)


def test_multisource_eval_reports_nonzero_citation_diversity_gain():
    report = compare_single_vs_multi_source(
        songs_csv_path="data/songs.csv",
        profiles=[{"genre": "pop", "mood": "happy", "energy": 0.85}],
        text_sources=[
            ("data/artist_context.md", "artist_notes", 0.9),
            ("data/genre_notes.md", "genre_notes", 0.8),
        ],
        k=3,
        context_k=4,
    )

    assert report["single_source"]["avg_unique_sources"] >= 1.0
    assert report["multi_source"]["avg_unique_sources"] > report["single_source"]["avg_unique_sources"]
