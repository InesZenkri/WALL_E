# Install

```bash
virtualenv .venv -p python3.11
source .venv/bin/activate
pip install -e .
```

```
touch .env
OPENAI_API_KEY=sk-or-v1-token
OPENAI_MODEL=openai/gpt-4o
MAX_TOKENS=10000
uvicorn brain.routes:app --reload --log-level=critical --host=0.0.0.0
```
