import pandas as pd
import streamlit as st

from src.showcase import playlist_rows, preset_profiles, run_live_agent, trace_rows


st.set_page_config(page_title="AI DJ Agent", page_icon="🎚️", layout="wide")


def main() -> None:
    st.title("AI DJ Agent")
    st.caption("Generate a playlist through a planner-executor-evaluator loop and inspect the intermediate trace.")

    presets = preset_profiles()
    preset_map = {preset["name"]: preset["prefs"] for preset in presets}

    with st.sidebar:
        preset_name = st.selectbox("Intent Seed", list(preset_map.keys()), index=1)
        default = preset_map[preset_name]
        genre = st.text_input("Genre", value=default["genre"])
        mood = st.text_input("Mood", value=default["mood"])
        energy = st.slider("Energy", 0.0, 1.0, float(default["energy"]), 0.01)
        playlist_size = st.slider("Playlist Size", 3, 8, 5, 1)

    result = run_live_agent(
        user_intent={"genre": genre, "mood": mood, "energy": energy},
        playlist_size=playlist_size,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Tracks", len(result.playlist))
    m2.metric("Revised", "Yes" if result.revised else "No")
    m3.metric("Max Energy Jump", f"{result.max_energy_jump_seen:.2f}")

    st.subheader("Playlist")
    st.dataframe(pd.DataFrame(playlist_rows(result)), use_container_width=True, hide_index=True)

    if result.warnings:
        st.warning("\n".join(result.warnings))
    else:
        st.success("The agent completed without warnings.")

    st.subheader("Reasoning Trace")
    st.dataframe(pd.DataFrame(trace_rows(result)), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
