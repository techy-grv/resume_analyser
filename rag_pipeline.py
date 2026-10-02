from collections import defaultdict
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from resume_loader import load_resumes
from schemas import CandidateEvaluation


class HireFlowRAG:
    """
    Basic RAG pipeline:

    1. Read resumes.
    2. Split resume text into chunks.
    3. Create Gemini embeddings.
    4. Store chunks in a local FAISS index.
    5. Embed the JD and retrieve relevant chunks.
    6. Group chunks by candidate.
    7. Ask Gemini to evaluate each candidate using their full resume text.
    """

    # How many candidates are evaluated by Gemini at the same time.
    MAX_CONCURRENCY = 5

    def __init__(self) -> None:
        settings.validate()

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
        )

        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0,
            google_api_key=settings.gemini_api_key,
        )

        self.vector_store: FAISS | None = None

        # Full resume text by candidate name. Retrieval finds *who* matches;
        # the evaluation then reads the whole resume, not just the matched chunks.
        self.resume_texts: dict[str, str] = {}

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def _load_resume_texts(self) -> list[Document]:
        resumes = load_resumes(settings.resume_dir)

        if not resumes:
            raise ValueError(
                f"No PDF or DOCX resumes found in: {settings.resume_dir}"
            )

        self.resume_texts = {
            resume.metadata["candidate_name"]: resume.page_content
            for resume in resumes
        }

        return resumes

    def build_index(self) -> None:
        resumes = self._load_resume_texts()

        chunks: list[Document] = []

        for resume in resumes:
            resume_chunks = self.splitter.split_documents([resume])

            for chunk in resume_chunks:
                chunk.metadata["candidate_name"] = resume.metadata["candidate_name"]
                chunk.metadata["source"] = resume.metadata["source"]

            chunks.extend(resume_chunks)

        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )

        self.vector_store.save_local(str(settings.faiss_dir))

    def load_index(self) -> None:
        if not (settings.faiss_dir / "index.faiss").exists():
            self.build_index()
            return

        self.vector_store = FAISS.load_local(
            str(settings.faiss_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        # Reading text from files is cheap; only the embeddings are expensive.
        self._load_resume_texts()

    def _retrieve_candidates(
        self,
        jd: str,
        top_k: int,
    ) -> dict[str, list[Document]]:
        if self.vector_store is None:
            self.load_index()

        total_chunks = self.vector_store.index.ntotal
        k = min(top_k * 3, total_chunks)

        # One strong candidate can take up many of the top chunks, so keep
        # widening the search until there are enough distinct candidates
        # or every chunk has been searched.
        while True:
            retrieved_docs = self.vector_store.similarity_search(jd, k=k)

            grouped: dict[str, list[Document]] = defaultdict(list)

            for document in retrieved_docs:
                candidate_name = document.metadata["candidate_name"]
                grouped[candidate_name].append(document)

            if len(grouped) >= top_k or k >= total_chunks:
                break

            k = min(k * 2, total_chunks)

        # Keep only the requested number of candidates.
        return dict(list(grouped.items())[:top_k])

    def _build_evaluation_input(
        self,
        jd: str,
        candidate_name: str,
        documents: list[Document],
    ) -> dict[str, str]:
        resume_text = self.resume_texts.get(candidate_name)

        # Fall back to the retrieved chunks if the resume file was removed
        # after the index was built.
        if not resume_text:
            resume_text = "\n\n".join(
                document.page_content for document in documents
            )

        return {
            "jd": jd,
            "candidate_name": candidate_name,
            "resume_text": resume_text,
        }

    def _evaluation_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are helping a recruiter review resumes.

Evaluate the candidate only using:
1. The Job Description.
2. The resume text supplied below.

Do not invent experience or skills.

Return a structured candidate evaluation.
The match score should represent how well the resume evidence
matches the requirements in the JD. Missing information should
be treated as "missing or unclear", not as proof that the candidate
does not have the skill.

This is a screening aid, not a final hiring decision.
""",
                ),
                (
                    "human",
                    """
JOB DESCRIPTION:
{jd}

CANDIDATE NAME:
{candidate_name}

RESUME:
{resume_text}
""",
                ),
            ]
        )

        structured_llm = self.llm.with_structured_output(
            CandidateEvaluation
        )

        return prompt | structured_llm

    def search_and_evaluate(
        self,
        jd: str,
        top_k: int = 5,
    ) -> list[CandidateEvaluation]:
        candidates = self._retrieve_candidates(jd, top_k)

        inputs = [
            self._build_evaluation_input(jd, candidate_name, documents)
            for candidate_name, documents in candidates.items()
        ]

        # Evaluate candidates in parallel instead of one Gemini call at a time.
        results = self._evaluation_chain().batch(
            inputs,
            config={"max_concurrency": self.MAX_CONCURRENCY},
        )

        results.sort(
            key=lambda item: item.match_score,
            reverse=True,
        )

        return results
