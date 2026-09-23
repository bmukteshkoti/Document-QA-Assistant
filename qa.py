import os
import time

from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GET GEMINI API KEY
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. Add it to your .env file."
    )


# ============================================================
# CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=api_key)


# ============================================================
# GEMINI MODEL
# ============================================================

MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, context_chunks):

    # --------------------------------------------------------
    # Check whether relevant context was retrieved
    # --------------------------------------------------------

    if not context_chunks:

        return (
            "I couldn't find the answer in the uploaded documents."
        )


    # --------------------------------------------------------
    # Combine retrieved PDF chunks
    # --------------------------------------------------------

    context = "\n\n".join(
        [
            f"Source {i + 1}:\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        ]
    )


    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are a Document Question-Answering Assistant.

Your job is to answer the user's question using ONLY the
information provided in the document context below.

IMPORTANT RULES:

1. Use ONLY the provided document context.
2. Do NOT use your general knowledge.
3. Do NOT add facts that are not present in the context.
4. Do NOT make assumptions.
5. If the answer cannot be found in the context, respond exactly:
   "I couldn't find the answer in the uploaded documents."
6. Start directly with the answer.
7. Do NOT say "Based on the provided documents".
8. Use simple and clear language.
9. Use numbered points when explaining multiple concepts.
10. Use bullet points when appropriate.
11. Keep the answer focused on the user's question.
12. Do NOT use unnecessary headings.
13. Do NOT use Markdown bold formatting such as **text**.
14. Do NOT mention the retrieved sources unless necessary.
15. Answer like a clear college/exam-style explanation.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Now answer the question using ONLY the document context.
"""


    # ========================================================
    # GEMINI REQUEST WITH RETRIES
    # ========================================================

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            # ------------------------------------------------
            # Make sure a response was actually returned
            # ------------------------------------------------

            if response and response.text:

                return response.text.strip()

            return (
                "I couldn't generate an answer from the "
                "uploaded documents."
            )


        # ====================================================
        # HANDLE API ERRORS
        # ====================================================

        except Exception as e:

            error_message = str(e)


            # ------------------------------------------------
            # 429 - QUOTA EXCEEDED
            # ------------------------------------------------

            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            ):

                return (
                    "Gemini API quota has been reached for the "
                    "current free-tier limit. Please try again "
                    "after the quota resets."
                )


            # ------------------------------------------------
            # 503 - TEMPORARILY UNAVAILABLE
            # ------------------------------------------------

            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                if attempt < max_retries - 1:

                    wait_time = 2 ** attempt

                    time.sleep(wait_time)

                    continue

                return (
                    "Gemini is temporarily unavailable because "
                    "the AI service is experiencing high demand. "
                    "Please try again later."
                )


            # ------------------------------------------------
            # OTHER API ERRORS
            # ------------------------------------------------

            return (
                "An error occurred while generating the answer. "
                "Please try again."
            )