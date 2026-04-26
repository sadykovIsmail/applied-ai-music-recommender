from src.agent.dj_agent import AIDJAgent
from src.recommender import load_songs
from src.retrieval import MusicContextRetriever


def _build_agent_from_catalog(max_iterations: int = 2, max_energy_jump: float = 0.35) -> AIDJAgent:
    songs = load_songs("data/songs.csv")
    retriever = MusicContextRetriever.from_multi_source(
        songs=songs,
        text_sources=[
            ("data/artist_context.md", "artist_notes", 0.9),
            ("data/genre_notes.md", "genre_notes", 0.8),
        ],
    )
    return AIDJAgent(
        songs=songs,
        retriever=retriever,
        max_iterations=max_iterations,
        max_energy_jump=max_energy_jump,
    )


def test_agent_generates_playlist_and_trace():
    agent = _build_agent_from_catalog()

    result = agent.run(
        user_intent={"genre": "lofi", "mood": "chill", "energy": 0.38},
        playlist_size=5,
    )

    assert len(result.playlist) == 5
    assert len(result.trace) >= 3
    tools_used = {step.tool for step in result.trace}
    assert "planner" in tools_used
    assert "fetch_candidate_tracks" in tools_used
    assert "energy_flow_checker" in tools_used


def test_agent_handles_empty_catalog_gracefully():
    agent = AIDJAgent(songs=[], retriever=MusicContextRetriever([]))

    result = agent.run(user_intent={"genre": "pop", "mood": "happy", "energy": 0.8}, playlist_size=4)

    assert result.playlist == []
    assert any("no candidates" in w.lower() for w in result.warnings)


def test_agent_revises_playlist_when_energy_flow_is_rough():
    agent = _build_agent_from_catalog(max_iterations=2, max_energy_jump=0.01)

    result = agent.run(
        user_intent={"genre": "rock", "mood": "intense", "energy": 0.90},
        playlist_size=5,
    )

    assert result.revised is True


def test_agent_stops_after_max_iterations_if_constraints_unmet():
    agent = _build_agent_from_catalog(max_iterations=1, max_energy_jump=0.01)

    result = agent.run(
        user_intent={"genre": "rock", "mood": "intense", "energy": 0.95},
        playlist_size=5,
    )

    assert any("max iterations" in w.lower() for w in result.warnings)


def test_agent_falls_back_when_genre_not_found():
    agent = _build_agent_from_catalog()

    result = agent.run(
        user_intent={"genre": "nonexistent-genre", "mood": "chill", "energy": 0.40},
        playlist_size=4,
    )

    assert len(result.playlist) == 4
