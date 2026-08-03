"""
Streamlit frontend
"""

import uuid
import requests
import streamlit as st
import streamlit.components.v1 as components
import re

def render_markdown_with_mermaid(text: str):
    if not text:
        return
        
    pattern = r'```mermaid\n(.*?)\n```'
    parts = re.split(pattern, text, flags=re.DOTALL)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            mermaid_code = part.strip()
            # Generate a unique ID for this mermaid diagram to avoid conflicts
            div_id = f"mermaid_{uuid.uuid4().hex[:8]}"
            html_code = f"""
            <script type="module">
              import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
              mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
            </script>
            <div class="mermaid" id="{div_id}">
              {mermaid_code}
            </div>
            """
            components.html(html_code, height=400, scrolling=True)

API_URL_RESEARCH = "http://localhost:8000/api/v1/query"
API_URL_PAPER_SEARCH = "http://localhost:8000/api/v1/papers/search"
API_URL_PAPER_SESSION = "http://localhost:8000/api/v1/paper/session"
API_URL_PAPER_QUERY = "http://localhost:8000/api/v1/query/paper"

st.set_page_config(page_title="CS Research Assistant", page_icon="📄")
st.title("📄 ResearchMind-AI")

# --- Session State Initialization ---
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "Research Assistant"

if "research_messages" not in st.session_state:
    st.session_state.research_messages = []
if "research_session_id" not in st.session_state:
    st.session_state.research_session_id = str(uuid.uuid4())

if "paper_messages" not in st.session_state:
    st.session_state.paper_messages = []
if "paper_session_id" not in st.session_state:
    st.session_state.paper_session_id = None
if "selected_paper_id" not in st.session_state:
    st.session_state.selected_paper_id = None
if "selected_paper_title" not in st.session_state:
    st.session_state.selected_paper_title = None

if "last_search_query" not in st.session_state:
    st.session_state.last_search_query = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Controls")
    
    st.session_state.current_mode = st.radio(
        "Mode",
        ["Research Assistant", "Paper Q&A"],
        index=0 if st.session_state.current_mode == "Research Assistant" else 1
    )
    
    if st.button("➕ New Chat"):
        if st.session_state.current_mode == "Research Assistant":
            st.session_state.research_messages = []
            st.session_state.research_session_id = str(uuid.uuid4())
        else:
            st.session_state.paper_messages = []
            if st.session_state.selected_paper_id:
                try:
                    resp = requests.post(API_URL_PAPER_SESSION, json={"paper_id": st.session_state.selected_paper_id})
                    resp.raise_for_status()
                    st.session_state.paper_session_id = resp.json()["session_id"]
                except Exception as e:
                    st.error(f"Failed to create new paper session: {e}")
        st.rerun()

# --- Paper Q&A Mode Configuration ---
if st.session_state.current_mode == "Paper Q&A":
    
    if not st.session_state.selected_paper_id:
        st.markdown("### 🔍 Search Paper")
        search_query = st.text_input("Search by title, author, or arXiv ID", value=st.session_state.last_search_query)
        
        # Trigger search automatically if length >= 3
        if len(search_query) >= 3 and search_query != st.session_state.last_search_query:
            st.session_state.last_search_query = search_query
            try:
                resp = requests.get(f"{API_URL_PAPER_SEARCH}?q={search_query}")
                resp.raise_for_status()
                st.session_state.search_results = resp.json()
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.session_state.search_results = []
        elif len(search_query) < 3 and search_query != st.session_state.last_search_query:
            st.session_state.last_search_query = search_query
            st.session_state.search_results = []
                
        # Display Results
        if len(search_query) >= 3:
            results = st.session_state.search_results
            if not results:
                st.info("No matching papers found.\n\nTry:\n• Different keywords\n• Author name\n• arXiv ID")
            else:
                for r in results:
                    with st.container(border=True):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{r['title']}**")
                            authors_str = ", ".join(r['authors']) if r['authors'] else "Unknown Authors"
                            st.caption(f"👤 {authors_str}")
                            
                            details = []
                            if r['publication_year']:
                                details.append(f"📅 {r['publication_year']}")
                            if r['arxiv_id']:
                                details.append(f"🔗 arXiv: {r['arxiv_id']}")
                            st.caption(" | ".join(details))
                        
                        with col2:
                            if st.button("Chat with Paper", key=f"select_{r['paper_id']}"):
                                with st.spinner("Creating paper session... Loading paper... Preparing chat..."):
                                    try:
                                        session_resp = requests.post(API_URL_PAPER_SESSION, json={"paper_id": r['paper_id']})
                                        session_resp.raise_for_status()
                                        st.session_state.paper_session_id = session_resp.json()["session_id"]
                                        st.session_state.selected_paper_id = r['paper_id']
                                        st.session_state.selected_paper_title = r['title']
                                        st.session_state.paper_messages = []
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to create session for paper: {e}")
    
    else:
        # Paper is selected
        col1, col2 = st.columns([4, 1])
        with col1:
            st.info(f"**Currently reading:** {st.session_state.selected_paper_title}")
        with col2:
            if st.button("Change Paper"):
                st.session_state.selected_paper_id = None
                st.session_state.selected_paper_title = None
                st.session_state.paper_session_id = None
                st.session_state.paper_messages = []
                st.rerun()

# --- Set active variables based on mode ---
if st.session_state.current_mode == "Research Assistant":
    active_messages = st.session_state.research_messages
    active_session_id = st.session_state.research_session_id
    api_url = API_URL_RESEARCH
    payload_builder = lambda query: {"query": query, "session_id": active_session_id}
else:
    active_messages = st.session_state.paper_messages
    active_session_id = st.session_state.paper_session_id
    api_url = API_URL_PAPER_QUERY
    payload_builder = lambda query: {"query": query, "session_id": active_session_id, "paper_id": st.session_state.selected_paper_id}

# --- Chat Display ---
for message in active_messages:
    with st.chat_message(message["role"]):
        render_markdown_with_mermaid(message.get("content", ""))
        
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
                    
            # Clean Evidence
            if data.get("citations") or data.get("evidence"):
                with st.expander("Evidence"):
                    # Combine unique citations based on paper, page, and section
                    seen = set()
                    for c in data.get("citations", []):
                        key = (c.get("title", "Unknown"), c.get("section", "Unknown"), c.get("page"))
                        if key not in seen:
                            seen.add(key)
                            page_str = f", Page {c['page']}" if c.get("page") else ""
                            st.markdown(f"- 📄 **{c.get('title', 'Unknown')}**")
                            st.caption(f"  📑 {c.get('section', 'Unknown')}{page_str}")
                    
                    for e in data.get("evidence", []):
                        key = (e.get("paper", "Unknown"), "Matched Text", e.get("page"))
                        if key not in seen:
                            seen.add(key)
                            page_str = f", Page {e['page']}" if e.get("page") else ""
                            st.markdown(f"- 📄 **{e.get('paper', 'Unknown')}**")
                            st.caption(f"  📑 Matched Text{page_str}")

# --- Input Box ---
if st.session_state.current_mode == "Paper Q&A" and not st.session_state.selected_paper_id:
    # No input box if in Paper mode and no paper selected
    pass
else:
    user_query = st.chat_input("Ask a question...")

    if user_query:
        active_messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = payload_builder(user_query)
                    response = requests.post(api_url, json=payload, timeout=600)
                    response.raise_for_status()
                    data = response.json()

                    if data.get("success"):
                        try:
                            answer = data.get("answer", "")
                            render_markdown_with_mermaid(answer)
                            
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
                                    
                            # Clean Evidence
                            if data.get("citations") or data.get("evidence"):
                                with st.expander("Evidence"):
                                    # Combine unique citations based on paper, page, and section
                                    seen = set()
                                    for c in data.get("citations", []):
                                        key = (c.get("title", "Unknown"), c.get("section", "Unknown"), c.get("page"))
                                        if key not in seen:
                                            seen.add(key)
                                            page_str = f", Page {c['page']}" if c.get("page") else ""
                                            st.markdown(f"- 📄 **{c.get('title', 'Unknown')}**")
                                            st.caption(f"  📑 {c.get('section', 'Unknown')}{page_str}")
                                    
                                    for e in data.get("evidence", []):
                                        key = (e.get("paper", "Unknown"), "Matched Text", e.get("page"))
                                        if key not in seen:
                                            seen.add(key)
                                            page_str = f", Page {e['page']}" if e.get("page") else ""
                                            st.markdown(f"- 📄 **{e.get('paper', 'Unknown')}**")
                                            st.caption(f"  📑 Matched Text{page_str}")
                                                
                            active_messages.append({
                                "role": "assistant", 
                                "content": answer, 
                                "data": data,
                                "id": str(uuid.uuid4())
                            })
                        except Exception as render_exc:
                            st.error(f"Frontend rendering crashed: {render_exc}")
                            active_messages.append({
                                "role": "assistant", 
                                "content": f"Frontend rendering crashed: {render_exc}",
                            })
                    else:
                        answer = f"⚠️ {data.get('error_reason', 'Unknown error')}"
                        st.warning(answer)
                        active_messages.append({"role": "assistant", "content": answer})

                except requests.RequestException as exc:
                    answer = f"⚠️ Could not reach the backend server: {exc}"
                    st.error(answer)
                    active_messages.append({"role": "assistant", "content": answer})