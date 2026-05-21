"""
agents/interviewer_agent.py
============================
Core interview logic: question generation, follow-up detection, and closing.
All LLM calls go through the local Ollama client — 100% offline.
"""

import os
import logging
from dotenv import load_dotenv
from utils.ollama_client import chat
from utils.prompts import (
    get_question_generation_prompt,
    get_followup_assessment_prompt,
    get_followup_question_prompt,
    get_closing_prompt,
)
import ollama

client = ollama.Client(
    host="http://host.docker.internal:11434"
)

load_dotenv()
logger = logging.getLogger(__name__)

MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "8"))

# Fallback questions used when Ollama is unavailable
_FALLBACK_QUESTIONS = [
    "Can you walk me through a challenging technical problem you solved recently and how you approached it?",
    "Tell me about a project you're particularly proud of. What was your specific contribution?",
    "How do you approach debugging a complex issue in a production environment?",
    "Describe a time you had to learn a new technology quickly. How did you do it?",
    "Tell me about a disagreement with a teammate and how you resolved it.",
    "What does good code look like to you, and how do you ensure quality in your work?",
    "How do you prioritize tasks when you're working on multiple things at once?",
    "Where do you see yourself growing technically in the next two years?",
]


def generate_question(
    resume_data: dict,
    conversation_history: list,
    asked_questions: list,
) -> str:
    """
    Generate the next interview question, personalized to the candidate's resume
    and contextual to what has been discussed so far.

    Args:
        resume_data:          Parsed resume data
        conversation_history: Full conversation for context
        asked_questions:      Questions already asked (to avoid repeating)

    Returns:
        Next question text
    """
    try:
        prompt = get_question_generation_prompt(
            resume_data, asked_questions, conversation_history
        )

        question = chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt = (
                "You are a friendly senior technical interviewer conducting a real mock interview. "

                "Your job is to ask natural, conversational interview questions based on the candidate's resume, "
                "projects, skills, and previous answers. "

                "Rules:\n"
                "- Ask ONLY ONE question\n"
                "- Maximum 15 words\n"
                "- Keep questions short and conversational\n"
                "- Sound like a real human interviewer\n"
                "- Avoid robotic or corporate language\n"
                "- Do not say 'How can I assist you today'\n"
                "- Do not explain the question\n"
                "- Do not ask multiple questions together\n"
                "- Focus on technical depth, projects, problem-solving, or behavioral experience\n"
                "- Prefer practical real-world interview questions\n"
                "- Never repeat previous questions\n"
                "- Return ONLY the question text\n"

                "Good examples:\n"
                "- Tell me about your React project.\n"
                "- How did you optimize API performance?\n"
                "- Describe a challenging bug you fixed.\n"
                "- Explain your role in the Salesforce project.\n"
                "- How did you handle deployment issues?\n"
            ),
            temperature = 0.3,
            max_tokens = 25
        )

        # Strip accidental prefixes from the model
        for prefix in ["Question:", "Q:", "Next question:", "Interview question:"]:
            if question.lower().startswith(prefix.lower()):
                question = question[len(prefix):].strip()

        logger.info(f"Question generated: '{question[:80]}'")
        return question

    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        # Use a fallback that hasn't been asked yet
        for q in _FALLBACK_QUESTIONS:
            if q not in asked_questions:
                return q
        return "What do you consider your greatest professional strength?"


def should_ask_followup(last_question: str, last_answer: str) -> bool:
    """
    Ask the local LLM to assess if the candidate's answer was shallow.
    Very short answers (<15 words) are automatically flagged.

    Returns:
        True  → ask a follow-up probe
        False → move on to a new topic
    """
    # Quick heuristic: very short answer is always shallow
    if len(last_answer.split()) < 20:
        logger.info("Answer < 15 words — triggering follow-up")
        return True

    try:
        prompt = get_followup_assessment_prompt(last_question, last_answer)

        assessment = chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You evaluate interview answers. Respond with exactly one word.",
            temperature=0.0,   # Deterministic assessment
            max_tokens=10,
        ).strip().upper()

        # Normalize — local models sometimes add punctuation
        for word in ["SHALLOW", "ADEQUATE", "THOROUGH"]:
            if word in assessment:
                logger.info(f"Answer assessment: {word}")
                return word == "SHALLOW"

        logger.warning(f"Unexpected assessment output: '{assessment}' — defaulting to no follow-up")
        return False

    except Exception as e:
        logger.error(f"Follow-up assessment failed: {e}")
        return False


def generate_followup_question(last_question: str, last_answer: str, resume_data: dict) -> str:
    """
    Generate a targeted follow-up question to probe a shallow answer deeper.

    Returns:
        Follow-up question text
    """
    try:
        prompt = get_followup_question_prompt(last_question, last_answer)

        followup = chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a skilled interviewer. Ask one sharp follow-up question.",
            temperature=0.7,
            max_tokens=120,
        )

        logger.info(f"Follow-up: '{followup[:80]}'")
        return followup

    except Exception as e:
        logger.error(f"Follow-up generation failed: {e}")
        return "Can you give me a specific example of that from your experience?"


def generate_closing_statement(candidate_name: str, question_count: int) -> str:
    """
    Generate a warm closing message to wrap up the interview before feedback.

    Returns:
        Closing statement text
    """
    try:
        prompt = get_closing_prompt(candidate_name, question_count)

        closing = chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are Alex, wrapping up a mock interview. Sound warm and genuine.",
            temperature=0.7,
            max_tokens=130,
        )

        return closing

    except Exception as e:
        logger.error(f"Closing generation failed: {e}")
        first = candidate_name.split()[0] if candidate_name else "there"
        return (
            f"Thank you so much, {first}! You've done a great job today. "
            f"I'm now putting together your detailed feedback report — just a moment."
        )


def is_interview_complete(question_count: int) -> bool:
    """Return True when the maximum number of questions has been reached."""
    return question_count >= MAX_QUESTIONS
