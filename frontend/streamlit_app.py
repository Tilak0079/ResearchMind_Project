"""
Streamlit frontend
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000/api/v1/query"

st.set_page_config(page_title="CS Research Assistant", page_icon="📄")
st.title("📄 ResearchMind-AI")

# st.session_state persists across Streamlit's re-runs.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Redraw the full chat history on every re-run (since Streamlit re-runs
# top to bottom each time, we need to re-display everything each time).
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question about CS research papers...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... (this may take 30-90 seconds)"):
            try:
                response = requests.post(
    API_URL,
    json={"query": user_query, "session_id": st.session_state.session_id},
    timeout=600,
)
                response.raise_for_status()
                data = response.json()

                if data["success"]:
                    answer = data["answer"]

                    # Show routing info as a small caption - useful for
                    # demoing HOW the system decided to answer, not just what it said.
                    st.caption(
                        f"Route: {data['route_taken']} | Confidence: {data['confidence_score']:.2f}"
                    )
                    st.markdown(answer)

                    if data.get("sources"):
                        with st.expander("Sources"):
                            for source in data["sources"]:
                                trust_note = " ⚠️ fetched from web (verified)" if source.get("trust_flag") else ""
                                link_text = f" — [View paper]({source['link']})" if source.get("link") else ""
                                st.write(f"- {source['section']} (Page {source.get('page', 'N/A')}){trust_note}{link_text}")
                else:
                    answer = f"⚠️ {data['error_reason']}"
                    st.warning(answer)

            except requests.RequestException as exc:
                answer = f"⚠️ Could not reach the backend server: {exc}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})