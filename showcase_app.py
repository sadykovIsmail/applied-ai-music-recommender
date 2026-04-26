from pathlib import Path

import pandas as pd
import streamlit as st

from src.showcase import (
    confidence_rows,
    dashboard_metrics,
    ensure_showcase_artifacts,
    get_profile,
    harness_rows,
    profile_names,
    refresh_showcase_artifacts,
)


st.set_page_config(
    page_title="VibeFinder Showcase",
    page_icon="assets/system-architecture.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&display=swap');
        :root {
            --bg: #f4efe6;
            --card: rgba(255,255,255,0.86);
            --ink: #1f2a2c;
            --muted: #5c6867;
            --accent: #0f766e;
            --accent-2: #c97c2f;
            --line: rgba(31,42,44,0.10);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(201,124,47,0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(15,118,110,0.16), transparent 26%),
                linear-gradient(180deg, #f7f1e7 0%, #efe6d8 100%);
            color: var(--ink);
        }
        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
        }
        .hero {
            padding: 2.2rem 2rem 1.8rem 2rem;
            border-radius: 28px;
            background:
                linear-gradient(130deg, rgba(255,255,255,0.88), rgba(255,248,238,0.78)),
                linear-gradient(120deg, rgba(15,118,110,0.10), rgba(201,124,47,0.08));
            border: 1px solid var(--line);
            box-shadow: 0 24px 60px rgba(58, 44, 18, 0.10);
        }
        .eyebrow {
            font-family: 'IBM Plex Mono', monospace;
            letter-spacing: .08em;
            color: var(--accent);
            font-size: 0.82rem;
            text-transform: uppercase;
        }
        .hero h1 {
            font-family: 'Instrument Serif', serif;
            font-size: 4rem;
            line-height: 0.92;
            margin: 0.4rem 0 0.8rem 0;
            color: #172124;
        }
        .hero p {
            color: var(--muted);
            max-width: 58rem;
            font-size: 1.04rem;
        }
        .chip-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        .chip {
            padding: 0.5rem 0.8rem;
            border-radius: 999px;
            background: rgba(15,118,110,0.08);
            border: 1px solid rgba(15,118,110,0.14);
            color: #114b47;
            font-size: 0.9rem;
        }
        .section-card {
            padding: 1.1rem 1.15rem;
            border-radius: 24px;
            background: var(--card);
            border: 1px solid var(--line);
            box-shadow: 0 18px 38px rgba(58, 44, 18, 0.07);
        }
        .section-title {
            font-family: 'Instrument Serif', serif;
            font-size: 2rem;
            margin-bottom: 0.4rem;
        }
        .muted {
            color: var(--muted);
        }
        div[data-testid="stMetric"] {
            padding: 1rem;
            background: rgba(255,255,255,0.72);
            border: 1px solid var(--line);
            border-radius: 20px;
        }
        .code-note {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: rgba(23,33,36,0.92);
            color: #f8efe4;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Applied AI System · Demo Surface</div>
            <h1>VibeFinder Showcase</h1>
            <p>
                A presentation-ready dashboard for the music recommender final project,
                combining grounded retrieval, agentic playlist planning, specialization,
                and reliability evaluation in one place.
            </p>
            <div class="chip-row">
                <div class="chip">Multi-source RAG</div>
                <div class="chip">Agent Trace</div>
                <div class="chip">Vinyl Historian Persona</div>
                <div class="chip">Confidence Diagnostics</div>
                <div class="chip">Automated Harness</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    hero()

    output_dir = "outputs"
    with st.sidebar:
        st.markdown("### Control Room")
        st.caption("Refresh the saved demo outputs before a presentation if you want fresh artifacts.")
        if st.button("Refresh Demo + Harness", use_container_width=True):
            with st.spinner("Regenerating artifacts..."):
                payload = refresh_showcase_artifacts(output_dir=output_dir)
            st.success("Artifacts refreshed.")
        else:
            payload = ensure_showcase_artifacts(output_dir=output_dir)

        selected_profile = st.selectbox("Profile", profile_names(payload))
        st.markdown("### Run Commands")
        st.code("python -m src.main", language="bash")
        st.code("python -m scripts.test_harness", language="bash")
        st.code("python -m pytest -q", language="bash")

    metrics = dashboard_metrics(payload)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Profiles", metrics["profile_count"])
    m2.metric("Avg Confidence", f"{metrics['average_confidence']:.3f}")
    m3.metric("Fallback Runs", metrics["fallback_runs"])
    m4.metric("Harness Score", f"{metrics['harness_passed']}/{metrics['harness_total']}")

    profile = get_profile(payload, selected_profile)
    profile_df = pd.DataFrame(profile["recommendations"])
    confidence_df = pd.DataFrame(confidence_rows(payload))
    harness_df = pd.DataFrame(harness_rows(payload))

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.markdown('<div class="section-title">Profile Explorer</div>', unsafe_allow_html=True)
        st.caption("Inspect how one persona travels through ranking, specialization, and reliability scoring.")
        st.dataframe(
            profile_df.assign(
                title=profile_df["song"].map(lambda row: row["title"]),
                artist=profile_df["song"].map(lambda row: row["artist"]),
                genre=profile_df["song"].map(lambda row: row["genre"]),
                energy=profile_df["song"].map(lambda row: row["energy"]),
            )[["title", "artist", "genre", "energy", "score"]],
            use_container_width=True,
            hide_index=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Standard Output")
            st.code(profile["standard_text"], language="text")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Specialized Output")
            st.code(profile["specialized_text"], language="text")
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Reliability Lens</div>', unsafe_allow_html=True)
        diagnostics = profile["diagnostics"]
        st.markdown(
            f"""
            <div class="section-card">
                <p class="muted">Status</p>
                <h3>{diagnostics['status'].upper()}</h3>
                <p class="muted">Warnings: {len(diagnostics['warnings'])} · Fallback used: {'yes' if diagnostics['fallback_used'] else 'no'}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.bar_chart(
            pd.DataFrame(
                {
                    "score": [
                        diagnostics["retrieval_confidence"],
                        diagnostics["rule_compliance_confidence"],
                        diagnostics["generation_confidence"],
                        diagnostics["confidence_score"],
                    ]
                },
                index=[
                    "retrieval",
                    "rule_compliance",
                    "generation",
                    "overall",
                ],
            )
        )
        if diagnostics["warnings"]:
            st.warning("\n".join(diagnostics["warnings"]))
        else:
            st.success("No warnings were raised for this profile.")

        with st.expander("Retrieved Context"):
            st.json(profile["contexts"])
        with st.expander("Structured Logs"):
            st.json(diagnostics["logs"])

    st.markdown('<div class="section-title">System Scoreboard</div>', unsafe_allow_html=True)
    c_left, c_right = st.columns([1.1, 0.9], gap="large")
    with c_left:
        st.caption("Confidence breakdown across the four demo profiles.")
        st.dataframe(confidence_df, use_container_width=True, hide_index=True)
    with c_right:
        st.caption("Automated harness outcomes across healthy and failure-path scenarios.")
        st.dataframe(harness_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Presentation Notes</div>', unsafe_allow_html=True)
    notes_left, notes_right = st.columns([0.8, 1.2], gap="large")
    with notes_left:
        st.image(str(Path("assets/system-architecture.png")), use_container_width=True)
    with notes_right:
        st.markdown(
            """
            <div class="section-card">
                <p class="muted">
                    Suggested live demo flow: open the dashboard, select the edge-case profile,
                    point out the cited retrieval context, compare the standard and specialized
                    outputs, then close by showing the harness table with pass/fail coverage and confidence.
                </p>
                <div class="code-note">streamlit run showcase_app.py</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
