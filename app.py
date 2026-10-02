import streamlit as st
from pathlib import Path

from config import settings
from rag_pipeline import HireFlowRAG
from resume_loader import extract_pdf_text


st.set_page_config(
    page_title="HireFlow",
    page_icon="💼",
    layout="wide",
)

st.title("HireFlow")
st.caption("Simple AI-assisted candidate search using Gemini + LangChain + FAISS")

# Keep the object in Streamlit session state so the FAISS index is not rebuilt
# every time the page is refreshed.
if "hireflow" not in st.session_state:
    st.session_state.hireflow = None

with st.sidebar:
    st.header("Resume Index")

    resume_count = len(list(settings.resume_dir.glob("*.pdf"))) + len(
        list(settings.resume_dir.glob("*.docx"))
    )

    st.write(f"Resumes found: **{resume_count}**")

    if st.button("Build / Rebuild Index", use_container_width=True):
        with st.spinner("Reading resumes and creating embeddings..."):
            try:
                st.session_state.hireflow = HireFlowRAG()
                st.session_state.hireflow.build_index()
                st.success("Resume index is ready.")
            except Exception as exc:
                st.error(f"Could not build the index: {exc}")

    if st.button("Clear Index", use_container_width=True):
        st.session_state.hireflow = None
        st.info("Index cleared.")

st.subheader("Job Description")

jd_source = st.radio(
    "How do you want to provide the Job Description?",
    ["Paste text", "Upload PDF"],
    horizontal=True,
)

jd = ""
if jd_source == "Paste text":
    jd = st.text_area(
        "Paste the Job Description",
        height=260,
        placeholder="Example: We are looking for a Senior Backend Engineer with Node.js, Python, AWS and PostgreSQL experience...",
    )
else:
    jd_file = st.file_uploader("Upload the Job Description (PDF)", type=["pdf"])

    if jd_file is not None:
        try:
            jd = extract_pdf_text(jd_file)
        except Exception as exc:
            st.error(f"Could not read the PDF: {exc}")

        if jd:
            with st.expander("Preview extracted Job Description"):
                st.text(jd)
        else:
            st.warning("No text could be extracted from this PDF. It may be a scanned image.")

top_k = st.slider("Number of candidates to retrieve", min_value=1, max_value=10, value=5)

search_clicked = st.button("Find Candidates", type="primary", use_container_width=True)

if search_clicked:
    if not jd.strip():
        st.warning("Please paste a Job Description or upload a PDF.")
        st.stop()

    if st.session_state.hireflow is None:
        with st.spinner("Loading the resume index..."):
            try:
                st.session_state.hireflow = HireFlowRAG()
                # Reuses the saved FAISS index; only builds one if none exists.
                # Use "Build / Rebuild Index" after adding or removing resumes.
                st.session_state.hireflow.load_index()
            except Exception as exc:
                st.error(f"Could not load resumes: {exc}")
                st.stop()

    with st.spinner("Searching resumes and evaluating candidates..."):
        try:
            results = st.session_state.hireflow.search_and_evaluate(
                jd=jd,
                top_k=top_k,
            )
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            st.stop()

    if not results:
        st.info("No candidate information was found.")
        st.stop()

    st.subheader("Candidate Results")

    for index, result in enumerate(results, start=1):
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"### {index}. {result.candidate_name}")
                st.write(result.summary)

            with col2:
                st.metric("Match", f"{result.match_score}%")

            st.markdown("**Strengths**")
            for item in result.strengths:
                st.write(f"- {item}")

            st.markdown("**Missing / Unclear**")
            for item in result.missing_skills:
                st.write(f"- {item}")

            st.caption(
                "This result is an AI-generated screening aid. "
                "Review the resume and requirements before making a hiring decision."
            )
