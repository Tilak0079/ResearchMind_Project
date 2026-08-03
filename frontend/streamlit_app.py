"""
Streamlit frontend
"""

import uuid
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
        st.markdown(message.get("content", ""))
        
        if message["role"] == "assistant" and "data" in message:
            data = message["data"]
            
            # Display artifacts
            artifacts = data.get("artifacts", [])
            for art in artifacts:
                art_type = art.get("type")
                if art_type in ["figure", "figure_caption"] and art.get("url"):
                    st.image(art["url"], caption=art.get("title", "Figure"))
                elif art_type == "table" and art.get("content"):
                    st.markdown(art["content"])
                elif art_type == "equation" and art.get("content"):
                    st.latex(art["content"])
            
            # Display Summary
            summary = data.get("summary", [])
            if summary:
                st.markdown("**Summary:**")
                for item in summary:
                    st.markdown(f"- {item}")
                    
            # Display Limitations
            limitations = data.get("limitations", [])
            if limitations:
                st.markdown("**Limitations:**")
                for item in limitations:
                    st.markdown(f"- {item}")
                    
            # Detailed Info (Expanders)
            col1, col2, col3 = st.columns(3)
            with col1:
                if data.get("citations"):
                    with st.expander("Citations"):
                        for c in data["citations"]:
                            trust = " ⚠️ unverified" if c.get("trust_tier") == "unverified" else ""
                            link = f" [🔗]({c['url']})" if c.get("url") else ""
                            st.markdown(f"- **{c.get('title', 'Unknown')}** (Page {c.get('page', '?')}){trust}{link}")
            with col2:
                if data.get("evidence"):
                    with st.expander("Evidence"):
                        for e in data["evidence"]:
                            st.caption(f"**{e.get('paper')}** (Score: {e.get('similarity_score', 0):.2f})")
                            st.write(e.get("chunk_text", ""))
                            st.divider()
            with col3:
                conf = data.get("confidence")
                if conf:
                    with st.expander(f"Confidence: {conf.get('overall_confidence', 0):.2f}"):
                        st.markdown(f"*{conf.get('explanation', '')}*")
                        if conf.get("factors"):
                            for f in conf["factors"]:
                                st.markdown(f"- {f}")
                                
            # Follow-up questions
            follow_ups = data.get("follow_up_questions", [])
            if follow_ups:
                st.markdown("**Suggested Follow-ups:**")
                for q in follow_ups:
                    st.button(q, key=f"{message.get('id', '')}_{q}") # Button for follow-ups

user_query = st.chat_input("Ask a question about CS research papers...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    # Streamlit re-run will render the user message from the loop above, but we also render it directly here for the first pass
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

                if data.get("success"):
                    try:
                        answer = data.get("answer", "")
                        st.markdown(answer)
                        
                        # Display artifacts
                        artifacts = data.get("artifacts", [])
                        for art in artifacts:
                            art_type = art.get("type")
                            if art_type in ["figure", "figure_caption"] and art.get("url"):
                                st.image(art["url"], caption=art.get("title", "Figure"))
                            elif art_type == "table" and art.get("content"):
                                st.markdown(art["content"])
                            elif art_type == "equation" and art.get("content"):
                                st.latex(art["content"])
                        
                        # Display Summary
                        summary = data.get("summary", [])
                        if summary:
                            st.markdown("**Summary:**")
                            for item in summary:
                                st.markdown(f"- {item}")
                                
                        # Display Limitations
                        limitations = data.get("limitations", [])
                        if limitations:
                            st.markdown("**Limitations:**")
                            for item in limitations:
                                st.markdown(f"- {item}")
                                
                        # Detailed Info (Expanders)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if data.get("citations"):
                                with st.expander("Citations"):
                                    for c in data["citations"]:
                                        trust = " ⚠️ unverified" if c.get("trust_tier") == "unverified" else ""
                                        link = f" [🔗]({c['url']})" if c.get("url") else ""
                                        st.markdown(f"- **{c.get('title', 'Unknown')}** (Page {c.get('page', '?')}){trust}{link}")
                        with col2:
                            if data.get("evidence"):
                                with st.expander("Evidence"):
                                    for e in data["evidence"]:
                                        st.caption(f"**{e.get('paper')}** (Score: {e.get('similarity_score', 0):.2f})")
                                        st.write(e.get("chunk_text", ""))
                                        st.divider()
                        with col3:
                            conf = data.get("confidence")
                            if conf:
                                with st.expander(f"Confidence: {conf.get('overall_confidence', 0):.2f}"):
                                    st.markdown(f"*{conf.get('explanation', '')}*")
                                    if conf.get("factors"):
                                        for f in conf["factors"]:
                                            st.markdown(f"- {f}")
                                            
                        # Follow-up questions
                        follow_ups = data.get("follow_up_questions", [])
                        if follow_ups:
                            st.markdown("**Suggested Follow-ups:**")
                            for q in follow_ups:
                                st.button(q, key=f"new_{q}") # Button for follow-ups
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer, 
                            "data": data,
                            "id": str(uuid.uuid4())
                        })
                    except Exception as render_exc:
                        st.error(f"Frontend rendering crashed: {render_exc}")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"Frontend rendering crashed: {render_exc}",
                        })
                else:
                    answer = f"⚠️ {data.get('error_reason', 'Unknown error')}"
                    st.warning(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

            except requests.RequestException as exc:
                answer = f"⚠️ Could not reach the backend server: {exc}"
                st.error(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})