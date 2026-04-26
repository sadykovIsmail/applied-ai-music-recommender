import pandas as pd
import streamlit as st

from src.showcase import preset_profiles, run_live_profile


st.set_page_config(page_title="Live Recommender", page_icon="🎛️", layout="wide")


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&display=swap');
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(201,124,47,0.18), transparent 30%),
                linear-gradient(180deg, #f7f1e7 0%, #efe6d8 100%);
        }
        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
        }
        .page-title {
            font-family: 'Instrument Serif', serif;
            font-size: 3rem;
            margin-bottom: 0.2rem;
        }
        .muted {
            color: #596765;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_page_style()
    st.markdown('<div class="page-title">Live Recommendation Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="muted">Run the actual system live, compare standard and specialized output, and inspect the retrieved evidence.</p>',
        unsafe_allow_html=True,
    )

    presets = preset_profiles()
    preset_map = {preset["name"]: preset["prefs"] for preset in presets}

    with st.sidebar:
        preset_name = st.selectbox("Preset Profile", list(preset_map.keys()))
        default = preset_map[preset_name]
        profile_name = st.text_input("Display Name", value=preset_name)
        genre = st.text_input("Genre", value=default["genre"])
        mood = st.text_input("Mood", value=default["mood"])
        energy = st.slider("Energy", 0.0, 1.0, float(default["energy"]), 0.01)
        mode = st.selectbox("Output Mode", ["vinyl_historian", "standard"])
        use_few_shot = st.toggle("Use Few-Shot Constraints", value=True)

    session = run_live_profile(
        profile_name=profile_name,
        prefs={"genre": genre, "mood": mood, "energy": energy},
        mode=mode,
        use_few_shot=use_few_shot,
    )

    diagnostics = session.diagnostics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Confidence", f"{diagnostics.confidence_score:.3f}")
    m2.metric("Retrieval", f"{diagnostics.retrieval_confidence:.3f}")
    m3.metric("Rule Compliance", f"{diagnostics.rule_compliance_confidence:.3f}")
    m4.metric("Generation", f"{diagnostics.generation_confidence:.3f}")

    rec_rows = [
        {
            "title": song["title"],
            "artist": song["artist"],
            "genre": song["genre"],
            "mood": song["mood"],
            "energy": song["energy"],
            "score": score,
        }
        for song, score, _ in session.recommendations
    ]
    st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

    t1, t2, t3 = st.tabs(["Standard Output", "Specialized Output", "Final Output"])
    with t1:
        st.code(session.standard_text or "No standard output generated.", language="text")
    with t2:
        st.code(session.specialized_text or "No specialized output generated.", language="text")
    with t3:
        st.code(session.final_text, language="text")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Retrieved Context")
        st.json(session.to_dict()["contexts"])
    with right:
        st.subheader("Warnings and Logs")
        if diagnostics.warnings:
            st.warning("\n".join(diagnostics.warnings))
        else:
            st.success("No warnings for this run.")
        st.json(diagnostics.logs)


if __name__ == "__main__":
    main()
