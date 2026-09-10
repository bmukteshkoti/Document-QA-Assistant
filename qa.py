import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_answer(question, relevant_chunks):

    context = "\n\n".join(relevant_chunks)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information provided in the context below.

If the answer is not present in the context, say:
"I couldn't find this information in the uploaded documents."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text