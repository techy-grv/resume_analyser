# Task 1 - Requirements

### Functional Requirements

1. JD Upload- Recruiter should be able to upload or enter a Job Description (JD).
2. Resume Upload - Recruiter should be able to upload multiple candidate resumes in PDF/DOCX format.
3. Candidate Search- Recruiter should be able to search candidates using a JD based on skills, experience, education, and other requirements.
4. Candidate Ranking - System should rank candidates based on their relevance to the JD and generate a match/confidence score.
5. Candidate Evaluation - System should provide an explanation of each candidate's strengths, weaknesses, missing skills, and overall matching with the JD.

### Non-Functional Requirements

1. Performance- Candidate search should return results within an given latency.

2. Scalability- System should support thousands or millions of resumes and increasing numbers of recruiters without major performance degradation.

3. Availability- System should be highly available.

4. Security- Candidate resumes and personal information should be securely stored and protected using authentication, authorization, and encryption.

5. Reliability- Resume processing and candidate matching should handle failures without losing uploaded candidate data.

---

# Task 2 - High-Level Architecture

### Frontend

Provides the recruiter interface for:

- Uploading job descriptions
- Uploading resumes
- Searching candidates
- Viewing candidate rankings
- Reviewing generated explanations

### API Gateway

It will handle authentication, request routing, rate limiting, and basic request validation.

### JD Service

This will be responsible for storing and managing job descriptions. It can also extract useful information such as required skills, experience, education, and other requirements.

### Resume Service

Handles resume uploads and candidate-related operations.