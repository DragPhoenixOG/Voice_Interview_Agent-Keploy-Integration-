"""
memory/transcript_store.py
==========================
Interview session state management and transcript persistence.
Each session lives in memory during the interview and is flushed
to a JSON file on disk when the interview ends.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TRANSCRIPT_DIR = os.getenv("TRANSCRIPT_DIR", "transcripts")


class TranscriptStore:
    """
    Holds the state for a single interview session.
    Tracks conversation history, asked questions, state machine, and feedback.
    """

    def __init__(self, session_id: str = None):
        self.session_id    = session_id or str(uuid.uuid4())
        self.created_at    = datetime.now().isoformat()
        self.resume_data:  dict = {}
        self.conversation_history: list = []   # OpenAI-format messages for LLM context
        self.transcript:   list = []           # Rich entries for display + saving
        self.asked_questions: list = []        # Prevent repeated questions
        self.question_count:  int  = 0
        self.interview_state: str  = "not_started"
        # States: not_started → greeting → in_progress → closing → completed
        self.feedback: Optional[dict] = None

        Path(TRANSCRIPT_DIR).mkdir(parents=True, exist_ok=True)
        logger.info(f"Session created: {self.session_id[:8]}")

    # ── Resume ────────────────────────────────────────────────────────────────

    def set_resume(self, resume_data: dict):
        self.resume_data = resume_data
        logger.info(f"Resume set: {resume_data.get('name', 'Unknown')}")

    # ── Messages ──────────────────────────────────────────────────────────────

    def add_message(
        self,
        role: str,
        content: str,
        is_question: bool = False,
        metadata: dict = None,
    ):
        """
        Add a message to both:
          - conversation_history  (sent to the LLM for context)
          - transcript            (saved to disk + shown in UI)

        Args:
            role:        "user" (candidate) or "assistant" (interviewer)
            content:     Message text
            is_question: True if this is an interviewer question (tracks count)
            metadata:    Optional extra info (type, question_number, etc.)
        """
        timestamp = datetime.now().isoformat()

        # LLM context window entry
        self.conversation_history.append({"role": role, "content": content})

        # Full transcript entry
        self.transcript.append({
            "role":        role,
            "content":     content,
            "timestamp":   timestamp,
            "is_question": is_question,
            "metadata":    metadata or {},
        })

        # Track questions to avoid repetition
        if role == "assistant" and is_question:
            self.asked_questions.append(content)
            self.question_count += 1
            logger.info(f"Question #{self.question_count} logged")

    def get_conversation_history(self, last_n: int = None) -> list:
        """Return history in LLM format, optionally capped to last N messages."""
        if last_n:
            return self.conversation_history[-last_n:]
        return self.conversation_history

    def get_transcript(self) -> list:
        return self.transcript

    def get_asked_questions(self) -> list:
        return self.asked_questions

    # ── State Machine ─────────────────────────────────────────────────────────

    def set_state(self, state: str):
        valid = {"not_started","waiting_intro", "greeting", "in_progress", "closing", "completed"}
        if state not in valid:
            raise ValueError(f"Invalid state '{state}'. Must be one of: {valid}")
        self.interview_state = state
        logger.info(f"State → {state}")

    # ── Feedback ──────────────────────────────────────────────────────────────

    def set_feedback(self, feedback: dict):
        self.feedback = feedback
        self.set_state("completed")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_to_disk(self) -> Optional[str]:
        """Save the complete session to a timestamped JSON file."""
        try:
            stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_{self.session_id[:8]}_{stamp}.json"
            filepath = Path(TRANSCRIPT_DIR) / filename

            payload = {
                "session_id":      self.session_id,
                "created_at":      self.created_at,
                "saved_at":        datetime.now().isoformat(),
                "candidate_name":  self.resume_data.get("name", "Unknown"),
                "interview_state": self.interview_state,
                "question_count":  self.question_count,
                "resume_summary": {
                    "name":              self.resume_data.get("name"),
                    "skills":            self.resume_data.get("skills", []),
                    "experience_count":  len(self.resume_data.get("experience", [])),
                    "projects_count":    len(self.resume_data.get("projects", [])),
                },
                "transcript": self.transcript,
                "feedback":   self.feedback,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            logger.info(f"Session saved → {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return None

    def to_dict(self) -> dict:
        """Serialize current session state (for API responses)."""
        return {
            "session_id":      self.session_id,
            "interview_state": self.interview_state,
            "question_count":  self.question_count,
            "candidate_name":  self.resume_data.get("name", "Unknown"),
            "transcript":      self.transcript,
            "feedback":        self.feedback,
        }


# ── Global Session Registry ───────────────────────────────────────────────────
# Simple in-memory store; replace with Redis for multi-user production use.

_sessions: dict[str, TranscriptStore] = {}


def create_session() -> TranscriptStore:
    store = TranscriptStore()
    _sessions[store.session_id] = store
    return store


def get_session(session_id: str) -> Optional[TranscriptStore]:
    return _sessions.get(session_id)


def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    logger.info(f"Session {session_id[:8]} removed from memory")
