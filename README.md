# Lore — Python API Backend

FastAPI backend for the Lore AI storytelling engine.  
Streaming story generation + structured A/B/C choices via Claude.

---

## Quick start

```bash
# 1. Clone / copy this folder
cd lore-backend

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env → ANTHROPIC_API_KEY=sk-ant-...

# 5. Run
python run.py
# → http://localhost:8000
# → http://localhost:8000/docs   (Swagger UI)
```

---

## API Reference

### `GET /health`
```json
{ "status": "ok", "timestamp": "...", "active_sessions": 3 }
```

### `GET /genres`
Returns all 4 genres with metadata (id, label, description, icon, recommended).

---

### `POST /session/start`
Start a new story session. Returns the opening scene synchronously.

**Request**
```json
{ "genre": "fantasy", "player_name": "Aria" }
```

**Response**
```json
{
  "session_id": "uuid",
  "genre": "fantasy",
  "turn": 1,
  "chapter_title": "The Iron Gate",
  "story": "The gate groans open...",
  "choices": [
    { "key": "A", "title": "Enter the cave", "subtitle": "Darkness, risk, possible treasure" },
    { "key": "B", "title": "Follow the torchlight", "subtitle": "Someone else is down here" },
    { "key": "C", "title": "Hold position", "subtitle": "Information before action" }
  ]
}
```

---

### `POST /session/{session_id}/choose`
Player makes a choice. Returns **Server-Sent Events** stream.

**Request** (choice-based)
```json
{ "choice_key": "A" }
```

**Request** (free text)
```json
{ "free_text": "I search the body for clues" }
```

**Request** (hybrid — choice + rider)
```json
{ "choice_key": "B", "free_text": "but I keep one hand on my sword" }
```

**SSE stream format**
```
data: {"type": "story_chunk", "text": "The cave swallows "}
data: {"type": "story_chunk", "text": "you whole..."}
data: {"type": "choices", "choices": [...], "turn": 2, "chapter_title": "..."}
data: {"type": "done"}
```

On error:
```
data: {"type": "error", "message": "...", "code": 500}
data: {"type": "done"}
```

---

### `GET /session/{session_id}`
Returns current session state (for reconnecting clients).

### `DELETE /session/{session_id}`
Ends session and frees memory.

---

## Frontend integration

```javascript
// 1. Start session
const { session_id, story, choices } = await fetch('/session/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ genre: 'fantasy', player_name: 'Aria' })
}).then(r => r.json());

// 2. Render opening scene + choices
renderStory(story);
renderChoices(choices);

// 3. On choice click — consume SSE stream
async function makeChoice(choiceKey) {
  lockButtons();
  showLoadingBar();

  const res = await fetch(`/session/${session_id}/choose`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ choice_key: choiceKey })
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const lines = decoder.decode(value).split('\n');
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const event = JSON.parse(line.slice(6));
      if (event.type === 'story_chunk') appendStoryText(event.text);
      if (event.type === 'choices')    renderChoices(event.choices);
      if (event.type === 'done')       unlockButtons();
      if (event.type === 'error')      showError(event.message);
    }
  }
}
```

---

## File structure

```
lore-backend/
├── main.py          ← FastAPI app, routes, streaming core
├── models.py        ← Pydantic request/response types
├── config.py        ← Settings (env vars / .env)
├── session_store.py ← In-memory session store with TTL
├── prompts.py       ← All Claude system prompts + builders
├── run.py           ← Entry point (uvicorn)
├── test_api.py      ← pytest test suite
├── requirements.txt
└── .env.example
```

---

## Running tests

```bash
pip install pytest httpx
pytest test_api.py -v
```

---

## Swapping the session store for Redis

Replace `session_store.py` with:

```python
import redis, json
from models import SessionState

class SessionStore:
    def __init__(self, **_):
        self.r = redis.Redis(host='localhost', port=6379, db=0)
        self.ttl = 3600  # seconds

    def get(self, sid):
        raw = self.r.get(sid)
        return SessionState(**json.loads(raw)) if raw else None

    def save(self, sid, session):
        self.r.setex(sid, self.ttl, session.model_dump_json())

    def delete(self, sid):
        self.r.delete(sid)

    def count(self):
        return self.r.dbsize()
```

No changes needed in `main.py` — the store interface is identical.

---

## Deploying to production

```bash
# Multiple workers (no reload in prod)
python run.py --workers 4

# Or directly with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Behind nginx — set proxy headers
uvicorn main:app --proxy-headers --forwarded-allow-ips='*'
```

Add `CORS_ORIGINS=["https://yourdomain.com"]` to `.env` for production.
"# AI-Powered-Smart-Energy-system" 
