"""
agents/greeter_agent.py
=======================
Generates a warm, personalized opening greeting using the local Ollama LLM.
Called once at the start of each interview session.
"""

import logging
from dotenv import load_dotenv
from utils.ollama_client import chat
from utils.prompts import get_greeting_prompt

load_dotenv()
logger = logging.getLogger(__name__)


def generate_greeting(resume_data: dict) -> str:
    """
    Generate a personalized greeting for the candidate based on their resume.

    Args:
        resume_data: Parsed resume dict (name, skills, experience, …)

    Returns:
        Greeting text to display and speak via TTS
    """
    try:
        prompt = get_greeting_prompt(resume_data)

        greeting = chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=(
                "You are Alex, a warm and professional AI interviewer. "
                "Write natural, concise, encouraging responses."
                "Speak naturally"
                "Keep responses under 2 short sentences"
                "Maximum 20 words"
                "No long explanations"
                "No filler text"
                "No corporate language"
                "No AI-sounding phrases"
            ),
            temperature=0.7,
            max_tokens=250,
        )

        logger.info(f"Greeting generated for: {resume_data.get('name', 'Unknown')}")
        return greeting

    except Exception as e:
        logger.error(f"Greeting generation failed: {e}")
        # Friendly fallback if Ollama is slow/unavailable
        name = resume_data.get("name", "there")
        first = name.split()[0] if name != "there" else name
        return (
            f"Hello {first}! I'm Alex, your AI interviewer today. "
            f"I've reviewed your resume and I'm excited to learn more about you. "
            f"We'll cover some technical and behavioral questions — take your time with each answer. "
            f"Are you ready to get started?"
        )
