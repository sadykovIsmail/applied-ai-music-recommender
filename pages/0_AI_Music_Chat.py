import streamlit as st

from src.showcase import build_chat_markdown, infer_profile_from_text, run_live_profile


st.set_page_config(page_title="AI Music Chat", page_icon="💬", layout="wide")


def initialize_chat() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Tell me the kind of music you want. For example: "
                    "`I want chill lofi for late-night studying` or "
                    "`Give me intense rock for the gym`."
                ),
            }
        ]


def render_chat_history() -> None:
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            extras = message.get("extras")
            if not extras:
                continue
            with st.expander("Parsed Profile"):
                st.json(extras["parsed_profile"])
            with st.expander("Retrieved Context"):
                st.json(extras["contexts"])
            with st.expander("Reliability Logs"):
                st.json(extras["logs"])


def main() -> None:
    initialize_chat()
    st.title("AI Music Chat")
    st.caption(
        "A chat-style front door for the recommender. Type a request in natural language and the app will infer a profile, run retrieval-backed recommendations, and return a specialized response."
    )

    with st.sidebar:
        st.markdown("### Prompt Ideas")
        st.markdown("- `I want chill lofi for studying`")
        st.markdown("- `Give me intense rock for the gym`")
        st.markdown("- `Need peaceful ambient music for sleep`")
        st.markdown("- `Find happy pop for a party`")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.pop("chat_messages", None)
            st.rerun()

    render_chat_history()

    prompt = st.chat_input("Describe the music vibe you want...")
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    parsed = infer_profile_from_text(prompt)
    profile_name = "Chat Listener"
    session = run_live_profile(
        profile_name=profile_name,
        prefs={
            "genre": parsed["genre"],
            "mood": parsed["mood"],
            "energy": parsed["energy"],
        },
        mode="vinyl_historian",
        use_few_shot=True,
    )
    content = build_chat_markdown(session)
    extras = {
        "parsed_profile": {
            "genre": parsed["genre"],
            "mood": parsed["mood"],
            "energy": parsed["energy"],
        },
        "contexts": session.to_dict()["contexts"],
        "logs": session.diagnostics.to_dict()["logs"],
    }
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": content,
            "extras": extras,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(content)
        with st.expander("Parsed Profile"):
            st.json(extras["parsed_profile"])
        with st.expander("Retrieved Context"):
            st.json(extras["contexts"])
        with st.expander("Reliability Logs"):
            st.json(extras["logs"])


if __name__ == "__main__":
    main()
