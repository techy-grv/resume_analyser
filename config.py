import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class Settings(BaseModel):
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-3.8-flash")
    embedding_model: str = Field(default="gemini-embedding-001")
    resume_dir: Path = Field(default=Path("resumes"))
    faiss_dir: Path = Field(default=Path("faiss_index"))
    chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=150)

    def validate(self) -> None:
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Add it to the .env file."
            )


settings = Settings(
    gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
    gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    embedding_model=os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "gemini-embedding-001",
    ),
    resume_dir=Path(os.getenv("RESUME_DIR", "resumes")),
    faiss_dir=Path(os.getenv("FAISS_DIR", "faiss_index")),
    chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
)

settings.resume_dir.mkdir(parents=True, exist_ok=True)
settings.faiss_dir.mkdir(parents=True, exist_ok=True)
