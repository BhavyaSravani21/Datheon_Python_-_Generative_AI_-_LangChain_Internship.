
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("API key not found. Check your .env file for GEMINI_API_KEY or GOOGLE_API_KEY.")
else:
    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents="Reply with exactly: Gemini connected successfully!"
        )
        print(response.text)
    except Exception as e:
        print("Gemini connection failed:")
        print(e)
