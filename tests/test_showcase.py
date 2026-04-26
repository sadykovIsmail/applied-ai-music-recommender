from src.showcase import (
    dashboard_metrics,
    ensure_showcase_artifacts,
    get_profile,
    harness_rows,
    profile_names,
    refresh_showcase_artifacts,
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
