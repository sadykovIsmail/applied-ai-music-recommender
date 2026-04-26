from src.showcase import (
    dashboard_metrics,
    ensure_showcase_artifacts,
    get_profile,
    harness_rows,
    playlist_rows,
    profile_names,
    refresh_showcase_artifacts,
    run_live_agent,
    run_live_profile,
    trace_rows,
)


def test_refresh_showcase_artifacts_writes_expected_payloads(tmp_path):
    payload = refresh_showcase_artifacts(output_dir=str(tmp_path))

    assert (tmp_path / "reliability_demo.json").exists()
    assert (tmp_path / "specialization_demo.json").exists()
    assert (tmp_path / "test_harness_results.json").exists()
    assert payload["harness"]["summary"]["scenario_count"] >= 5


def test_ensure_showcase_artifacts_loads_existing_payloads(tmp_path):
    refresh_showcase_artifacts(output_dir=str(tmp_path))

    payload = ensure_showcase_artifacts(output_dir=str(tmp_path))

    assert len(profile_names(payload)) == 4
    assert dashboard_metrics(payload)["profile_count"] == 4


def test_get_profile_and_harness_rows_expose_dashboard_data():
    payload = ensure_showcase_artifacts()

    name = profile_names(payload)[0]
    profile = get_profile(payload, name)
    rows = harness_rows(payload)

    assert profile["profile_name"] == name
    assert "diagnostics" in profile
    assert any(row["scenario"] == "happy_path_lofi" for row in rows)


def test_run_live_profile_returns_grounded_recommendations():
    session = run_live_profile(
        profile_name="Live Lofi",
        prefs={"genre": "lofi", "mood": "chill", "energy": 0.38},
        mode="vinyl_historian",
        use_few_shot=True,
    )

    assert len(session.recommendations) == 5
    assert "VINYL HISTORIAN" in session.specialized_text
    assert session.diagnostics.confidence_score > 0.0


def test_run_live_agent_exposes_playlist_and_trace_rows():
    result = run_live_agent(
        user_intent={"genre": "rock", "mood": "intense", "energy": 0.92},
        playlist_size=5,
    )

    playlist = playlist_rows(result)
    trace = trace_rows(result)

    assert len(playlist) == 5
    assert len(trace) >= 3
    assert any(step["tool"] == "planner" for step in trace)
