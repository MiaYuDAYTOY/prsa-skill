import os

OPENAI_KEY = os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", "YOUR_DEEPSEEK_KEY"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")

# Preserve PRSA's original two-model design
PRSA_STRONG_MODEL = os.getenv("PRSA_STRONG_MODEL", "deepseek-v4-pro")
PRSA_FAST_MODEL = os.getenv("PRSA_FAST_MODEL", "deepseek-v4-flash")

# Backward-compatible fallback for any old code that still reads OPENAI_MODEL
OPENAI_MODEL = os.getenv("OPENAI_MODEL", PRSA_STRONG_MODEL)
