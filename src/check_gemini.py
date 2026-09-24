import os, sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Minimal call: should print a short reply, not a 403
resp = client.models.generate_content(
    model="gemini-3.5-flash-lite", contents="Say OK."
)
print(resp.text)