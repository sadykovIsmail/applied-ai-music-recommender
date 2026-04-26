import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.test_harness import run_harness
from src.main import run_demo


def ensure_showcase_artifacts(output_dir: str = "outputs") -> Dict[str, Any]:
    out_dir = Path(output_dir)
    reliability_path = out_dir / "reliability_demo.json"
    harness_path = out_dir / "test_harness_results.json"

    if not reliability_path.exists():
        run_demo(output_dir=str(out_dir))
    if not harness_path.exists():
        run_harness(output_dir=str(out_dir))

    return load_showcase_payload(output_dir=str(out_dir))


def refresh_showcase_artifacts(output_dir: str = "outputs") -> Dict[str, Any]:
    out_dir = Path(output_dir)
    run_demo(output_dir=str(out_dir))
    run_harness(output_dir=str(out_dir))
    return load_showcase_payload(output_dir=str(out_dir))


def load_showcase_payload(output_dir: str = "outputs") -> Dict[str, Any]:
    out_dir = Path(output_dir)
    reliability = _load_json(out_dir / "reliability_demo.json")
    specialization = _load_json(out_dir / "specialization_demo.json")
    harness = _load_json(out_dir / "test_harness_results.json")
    return {
        "reliability": reliability,
        "specialization": specialization,
        "harness": harness,
    }


def profile_names(payload: Dict[str, Any]) -> List[str]:
    return [profile["profile_name"] for profile in payload["reliability"]["profiles"]]


def get_profile(payload: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
    for profile in payload["reliability"]["profiles"]:
        if profile["profile_name"] == profile_name:
            return profile
    raise KeyError(f"Unknown profile: {profile_name}")


def dashboard_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    reliability_profiles = payload["reliability"]["profiles"]
    harness_summary = payload["harness"]["summary"]
    return {
        "profile_count": len(reliability_profiles),
        "average_confidence": payload["reliability"]["summary"]["average_confidence"],
        "fallback_runs": payload["reliability"]["summary"]["fallback_runs"],
        "harness_passed": harness_summary["passed"],
        "harness_total": harness_summary["scenario_count"],
    }


def confidence_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for profile in payload["reliability"]["profiles"]:
        diagnostics = profile["diagnostics"]
        rows.append(
            {
                "profile": profile["profile_name"],
                "confidence_score": diagnostics["confidence_score"],
                "retrieval_confidence": diagnostics["retrieval_confidence"],
                "rule_compliance_confidence": diagnostics["rule_compliance_confidence"],
                "generation_confidence": diagnostics["generation_confidence"],
                "fallback_used": diagnostics["fallback_used"],
                "status": diagnostics["status"],
            }
        )
    return rows


def harness_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for scenario in payload["harness"]["scenarios"]:
        rows.append(
            {
                "scenario": scenario["name"],
                "status": scenario["status"],
                "confidence_score": scenario["confidence_score"],
                "latency_ms": scenario["latency_ms"],
                "fallback_used": scenario["fallback_used"],
                "warnings": " | ".join(scenario["warnings"]) if scenario["warnings"] else "",
            }
        )
    return rows


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
