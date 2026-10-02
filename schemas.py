from pydantic import BaseModel, Field


class CandidateEvaluation(BaseModel):
    candidate_name: str = Field(
        description="Candidate name. Use the name from the resume when available."
    )
    match_score: int = Field(
        ge=0,
        le=100,
        description="Approximate match score based only on the JD and resume evidence.",
    )
    strengths: list[str] = Field(
        description="Important requirements that are supported by the resume."
    )
    missing_skills: list[str] = Field(
        description="Required skills or requirements that are missing or unclear."
    )
    summary: str = Field(
        description="Short explanation of the candidate's match with the JD."
    )
