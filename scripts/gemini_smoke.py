import os

from dotenv import load_dotenv

load_dotenv()
from google import genai

key = os.environ.get("GEMINI_API_KEY")
if not key:
    raise SystemExit("GEMINI_API_KEY is missing in .env")

client = genai.Client(api_key=key)
print("Flash models available to your key:")
for m in client.models.list():
    if "flash" in (m.name or ""):
        print("  ", m.name)

model = os.environ.get("GEMINI_MODEL")
if model:
    r = client.models.generate_content(model=model, contents="Reply with the single word: ready")
    print("Model reply:", r.text)
else:
    print("Now set GEMINI_MODEL in .env to one name above (without the 'models/' prefix), then rerun.")
