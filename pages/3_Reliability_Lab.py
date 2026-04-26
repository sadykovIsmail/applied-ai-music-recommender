import pandas as pd
import streamlit as st

from src.showcase import (
    confidence_rows,
    dashboard_metrics,
    ensure_showcase_artifacts,
    harness_rows,
    profile_names,
    refresh_showcase_artifacts,
)


st.set_page_config(page_title="Reliability Lab", page_icon="🧪", layout="wide")


def main() -> None:
    st.title("Reliability Lab")
    st.caption("Review confidence, failure-path behavior, and automated harness outcomes.")

    with st.sidebar:
        if st.button("Refresh Saved Artifacts", use_container_width=True):
            payload = refresh_showcase_artifacts()
            st.success("Artifacts regenerated.")
        else:
            payload = ensure_showcase_artifacts()
        selected_profile = st.selectbox("Focus Profile", profile_names(payload))

    metrics = dashboard_metrics(payload)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Profiles", metrics["profile_count"])
    m2.metric("Avg Confidence", f"{metrics['average_confidence']:.3f}")
    m3.metric("Fallback Runs", metrics["fallback_runs"])
    m4.metric("Harness Score", f"{metrics['harness_passed']}/{metrics['harness_total']}")

    profile = next(
        entry for entry in payload["reliability"]["profiles"] if entry["profile_name"] == selected_profile
    )
    diagnostics = profile["diagnostics"]

    chart_df = pd.DataFrame(
        {
            "score": [
                diagnostics["retrieval_confidence"],
                diagnostics["rule_compliance_confidence"],
                diagnostics["generation_confidence"],
                diagnostics["confidence_score"],
            ]
        },
        index=["retrieval", "rule_compliance", "generation", "overall"],
    )

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        st.subheader("Focused Profile Diagnostics")
        st.bar_chart(chart_df)
        if diagnostics["warnings"]:
            st.warning("\n".join(diagnostics["warnings"]))
        else:
            st.success("No warnings for this saved profile.")
        with st.expander("Structured Logs"):
            st.json(diagnostics["logs"])

    with right:
        st.subheader("Harness Scenarios")
        st.dataframe(pd.DataFrame(harness_rows(payload)), use_container_width=True, hide_index=True)

    st.subheader("Confidence Table")
    st.dataframe(pd.DataFrame(confidence_rows(payload)), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
