from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Project Overview", page_icon="📚", layout="wide")


def main() -> None:
    st.title("Project Overview")
    st.caption("What this system is, what makes it AI-enabled, and how to narrate it in the demo.")

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown(
            """
            ### What The Project Is
            `VibeFinder` is not a chatbot-first product. It is an applied AI music recommendation system.

            The system combines:
            - multi-source retrieval before explanation generation
            - an AI DJ agent that plans and evaluates playlists
            - a specialized `vinyl_historian` response mode
            - a reliability layer with confidence scoring and fallback behavior
            - an automated harness for repeatable evaluation

            ### Why It Counts As RAG
            The app retrieves supporting context from the song catalog plus custom artist and genre notes before composing explanations. That retrieved evidence is cited directly in the recommendation output.

            ### Why It Counts As Agentic
            The AI DJ module does more than rank tracks once. It plans a playlist, checks transitions, revises when needed, and exposes its step-by-step trace.
            """
        )
    with right:
        st.image(str(Path("assets/system-architecture.png")), use_container_width=True)

    st.markdown(
        """
        ### Suggested Video Flow
        1. Start on the `Live Recommender` page and run a normal profile.
        2. Compare standard and specialized output.
        3. Switch to `AI DJ Agent` and show the trace table.
        4. Open `Reliability Lab` and show the harness pass/fail grid plus confidence.
        5. End here on the architecture overview if you want a clean wrap-up.
        """
    )


if __name__ == "__main__":
    main()
