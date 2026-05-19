"""
agents/feedback_agent.py
========================
Generates a comprehensive post-interview feedback report using
the local Ollama LLM — completely offline, no API costs.

Analyzes the full transcript and produces:
  - Overall score
  - Strengths / weaknesses
  - Communication, technical, and behavioral scores
  - Actionable improvement tips
  - Hiring recommendation
"""

import json
import logging
from dotenv import load_dotenv
from utils.ollama_client import chat
from utils.prompts import get_feedback_prompt

load_dotenv()
logger = logging.getLogger(__name__)


def generate_feedback(resume_data: dict, conversation_history: list, transcript: list) -> dict:
    """
    Analyze the complete interview transcript and produce a structured report.

    Args:
        resume_data:          Parsed resume dict
        conversation_history: Full conversation in LLM message format
        transcript:           Rich transcript list with metadata

    Returns:
        Feedback dict with scores, strengths, weaknesses, recommendation, etc.
    """
    if not transcript or len(transcript) < 2:
        logger.warning("Transcript too short for meaningful feedback")
        return _minimal_feedback(resume_data)

    try:
        prompt = get_feedback_prompt(resume_data, transcript)

        logger.info("Generating feedback report via Ollama...")

        raw = chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=(
                "You are a senior hiring manager providing honest, constructive interview feedback. "
                """
                Return ONLY valid JSON.

                Rules:
                - No markdown
                - No explanations
                - No extra braces
                - No trailing commas
                - No text before or after JSON"""
            ),
            temperature=0.3,   # Low temperature → consistent, factual feedback
            max_tokens=800,
        )

        import re

        # ─────────────────────────────────────────────────────
        # Clean model response
        # ─────────────────────────────────────────────────────
        cleaned = raw.strip()

        # Remove markdown fences
        cleaned = cleaned.replace("```json", "")
        cleaned = cleaned.replace("```", "")
        cleaned = cleaned.strip()

        # Extract first valid JSON object only
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if not match:
            raise ValueError("No valid JSON object found")

        cleaned = match.group(0)

        # Remove accidental trailing garbage
        cleaned = cleaned.strip()

        # Fix common qwen issues
        cleaned = cleaned.replace("\n}", "\n}")
        cleaned = cleaned.replace("}\n}", "}")

        logger.info(f"Cleaned feedback JSON: {cleaned[:300]}")

        # Parse safely
        feedback = json.loads(cleaned)

        logger.info(
            f"Feedback generated successfully"
        )

        return _validate(feedback)

    except json.JSONDecodeError as e:
        logger.error(f"Feedback JSON parse error: {e}\nRaw: {raw[:400]}")
        return _fallback_feedback(resume_data, transcript)
    except Exception as e:
        logger.error(f"Feedback generation failed: {e}")
        return _fallback_feedback(resume_data, transcript)


# ── Validation & Fallbacks ───────────────────────────────────────────────────

def _validate(feedback: dict) -> dict:
    """Ensure all required keys are present, filling defaults where missing."""
    defaults = {
        "overall_score": 5,
        "strengths":     ["Participated in the full interview process"],
        "weaknesses":    ["Insufficient data for detailed assessment"],
        "communication_feedback": {
            "score":    5,
            "feedback": "Communication assessment requires more responses."
        },
        "technical_feedback": {
            "score":    5,
            "feedback": "Technical assessment requires more responses."
        },
        "behavioral_feedback": {
            "score":    5,
            "feedback": "Behavioral assessment requires more responses."
        },
        "areas_to_improve":      ["Practice answering with specific examples"],
        "hiring_recommendation": "MAYBE — Insufficient data for a confident recommendation",
        "closing_message":       "Thank you for the interview. Keep practicing!",
    }
    for key, default in defaults.items():
        if key not in feedback:
            feedback[key] = default
    return feedback


def _fallback_feedback(resume_data: dict, transcript: list) -> dict:
    """Basic feedback when the LLM fails to generate or parse JSON."""
    name  = resume_data.get("name", "Candidate")
    count = sum(1 for e in transcript if e.get("role") == "user")
    return {
        "overall_score": 5,
        "strengths":     [f"Completed the interview and provided {count} responses"],
        "weaknesses":    ["Full feedback could not be generated due to a system error"],
        "communication_feedback": {
            "score":    5,
            "feedback": f"{name} responded to {count} questions. Manual review recommended."
        },
        "technical_feedback": {
            "score":    5,
            "feedback": "Technical evaluation was not completed due to a system error."
        },
        "behavioral_feedback": {
            "score":    5,
            "feedback": "Behavioral evaluation was not completed due to a system error."
        },
        "areas_to_improve": [
            "Practice the STAR method for behavioral answers",
            "Prepare specific examples with measurable outcomes",
            "Research common interview questions in your field",
        ],
        "hiring_recommendation": "MAYBE — System error prevented full evaluation",
        "closing_message": (
            f"Thank you, {name}! There was a minor issue generating detailed feedback, "
            "but keep practicing — you'll do great!"
        ),
    }


def _minimal_feedback(resume_data: dict) -> dict:
    """Feedback returned when the interview was too short to evaluate."""
    name = resume_data.get("name", "Candidate")
    return {
        "overall_score": 0,
        "strengths":     ["Initiated the interview session"],
        "weaknesses":    ["Interview was too short for meaningful evaluation"],
        "communication_feedback": {
            "score":    0,
            "feedback": "Not enough responses to evaluate communication."
        },
        "technical_feedback": {
            "score":    0,
            "feedback": "Not enough responses to evaluate technical skills."
        },
        "behavioral_feedback": {
            "score":    0,
            "feedback": "Not enough responses to evaluate behavioral competencies."
        },
        "areas_to_improve": [
            "Complete a full interview session (at least 5-6 questions) for real feedback"
        ],
        "hiring_recommendation": "NO — Interview too brief for evaluation",
        "closing_message": (
            f"Hi {name}, it looks like your session was quite short. "
            "Try completing a full interview for detailed, actionable feedback!"
        ),
    }


def format_feedback_for_tts(feedback: dict) -> str:
    """
    Format key feedback highlights as a short spoken summary.
    Spoken aloud via pyttsx3 at the end of the interview.
    """
    score      = feedback.get("overall_score", "N/A")
    recommend  = feedback.get("hiring_recommendation", "")
    closing    = feedback.get("closing_message", "")
    strengths  = feedback.get("strengths", [])
    areas      = feedback.get("areas_to_improve", [])

    parts = [f"Your overall interview score is {score} out of 10."]

    if strengths:
        parts.append(f"A key strength: {strengths[0]}")

    if areas:
        parts.append(f"One area to work on: {areas[0]}")

    if closing:
        parts.append(closing)

    return " ".join(parts)
