"""
main.py — FastAPI application for SIH26091 AI Business Advisory Assistant.

Endpoints:
  POST   /api/chat               → Main chat endpoint (3-tier memory)
  GET    /api/history/{user_id}  → Full chat history
  GET    /api/profile/{user_id}  → Structured business profile
  DELETE /api/memory/{user_id}   → Wipe user vector memory
  GET    /api/stats              → ChromaDB stats (admin)
  GET    /                       → Serves frontend index.html
"""

import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from dotenv import load_dotenv

import google.generativeai as genai

from database import (
    init_db,
    get_db,
    get_or_create_profile,
    save_message,
    get_recent_messages,
    get_all_messages,
    UserProfile,
    Message,
)
from memory_manager import (
    store_interaction,
    retrieve_relevant_history,
    format_rag_context,
    delete_user_memory,
    get_collection_stats,
)
from profile_updater import run_profile_update, should_run_profile_update

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set. API calls will fail.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

CHAT_MODEL        = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
PROFILE_UPDATE_INTERVAL = int(os.getenv("PROFILE_UPDATE_INTERVAL", "5"))
FRONTEND_DIR      = Path(__file__).parent.parent / "frontend"

# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

LANGUAGE_HINTS = {
    "en": "Respond in clear, simple English. Avoid jargon.",
    "hi": "हिंदी में जवाब दें। सरल और स्पष्ट भाषा का उपयोग करें।",
    "mr": "मराठीत उत्तर द्या. सोप्या भाषेत समजावा.",
    "ta": "தமிழில் பதிலளிக்கவும். எளிமையான மொழியைப் பயன்படுத்தவும்.",
    "te": "తెలుగులో సమాధానం ఇవ్వండి. సులభమైన భాషలో.",
    "kn": "ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ. ಸರಳ ಭಾಷೆ ಬಳಸಿ.",
    "bn": "বাংলায় উত্তর দিন। সহজ ভাষায় বলুন।",
    "gu": "ગુજરાતીમાં જવાબ આપો. સરળ ભાષામાં.",
}

BASE_SYSTEM_INSTRUCTION = """You are a friendly, expert business advisor and financial counsellor
for rural micro-entrepreneurs and small business owners in India.

Your responsibilities:
1. Provide practical, actionable advice on business operations, growth, and financial management.
2. Inform users about relevant government schemes (PM SVANidhi, Mudra Loan, PMEGP, Stand-Up India,
   PM Vishwakarma, Startup India, etc.) and guide them through application processes.
3. Help with basic financial planning: budgeting, savings, loans, working capital, etc.
4. Offer guidance on digital payments (UPI, BHIM), GST registration, business licences.
5. Suggest supply chain improvements, marketing strategies, and customer acquisition.
6. Be culturally sensitive, patient, and use relatable examples from rural/semi-urban India.
7. Keep answers concise but complete. Use numbered lists and bullet points for clarity.
8. NEVER give unethical advice. NEVER recommend illegal activities.
9. If you don't know something, honestly say so and suggest where the user can get help.

Always greet by name if you know it. Reference the user's business context where appropriate.
"""


def build_system_instruction(profile: UserProfile, language: str = "en") -> str:
    """Construct the full system instruction with injected user profile."""
    lang_hint = LANGUAGE_HINTS.get(language, LANGUAGE_HINTS["en"])

    profile_block = ""
    if profile:
        profile_text = profile.to_profile_text()
        profile_block = f"""
=== ENTREPRENEUR'S BUSINESS PROFILE (always use this context) ===
{profile_text}
=== END OF PROFILE ===
"""

    return f"""{BASE_SYSTEM_INSTRUCTION}

{lang_hint}

{profile_block}"""


# ---------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initialising database...")
    init_db()
    logger.info("Database ready.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SIH26091 Business Advisory API",
    description="AI-Driven Hyper-Local Business Advisory for Rural Micro-Entrepreneurs",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id:  str = Field(..., min_length=1, max_length=128, description="Unique user identifier")
    message:  str = Field(..., min_length=1, max_length=4096, description="User's chat message")
    language: str = Field("en", description="ISO 639-1 language code: en|hi|mr|ta|te|kn|bn|gu")


class ChatResponse(BaseModel):
    response:                      str
    relevant_retrieved_history_count: int
    turn_count:                    int
    profile_updated:               bool


class HistoryResponse(BaseModel):
    user_id:  str
    messages: list[dict]
    total:    int


class ProfileResponse(BaseModel):
    user_id: str
    profile: dict


# ---------------------------------------------------------------------------
# Background task: store in vector DB
# ---------------------------------------------------------------------------

def _background_store(
    user_id: str,
    user_message: str,
    bot_response: str,
    message_id: int,
    timestamp: str,
):
    """Store interaction in ChromaDB (runs in background thread)."""
    try:
        store_interaction(
            user_id=user_id,
            user_message=user_message,
            bot_response=bot_response,
            message_id=message_id,
            timestamp=timestamp,
        )
    except Exception as exc:
        logger.error(f"Background ChromaDB store failed: {exc}")


def _background_profile_update(user_id: str):
    """Run profile update in a fresh DB session (background thread)."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        run_profile_update(db, user_id)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Main chat endpoint
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    3-tier memory chat pipeline:
      A) Load user profile → inject into system instruction
      B) Semantic RAG retrieval from ChromaDB (top 3)
      C) Recent messages buffer (last 6 from DB)
      D) Gemini API call
      E) Async storage to DB + ChromaDB
      F) Profile update every PROFILE_UPDATE_INTERVAL turns
    """
    user_id  = request.user_id.strip()
    message  = request.message.strip()
    language = request.language.strip().lower()

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured.")

    # ── A: Profile ──────────────────────────────────────────────────────────
    profile = get_or_create_profile(db, user_id)
    system_instruction = build_system_instruction(profile, language)

    # ── B: RAG retrieval ────────────────────────────────────────────────────
    rag_hits = retrieve_relevant_history(user_id, message, top_k=3)
    rag_context = format_rag_context(rag_hits)

    # ── C: Short-term buffer (last 6 messages) ───────────────────────────
    recent_msgs = get_recent_messages(db, user_id, limit=6)

    # ── D: Build Gemini payload ──────────────────────────────────────────
    # Convert DB messages to Gemini history format
    gemini_history = []
    for msg in recent_msgs:
        gemini_history.append({
            "role": "user" if msg.role == "user" else "model",
            "parts": [{"text": msg.content}],
        })

    # Prepend RAG context as a synthetic model turn if available
    if rag_context:
        rag_injection = (
            f"[Advisory System: Relevant past context retrieved from memory]\n{rag_context}"
        )
        # Insert after first user message if history exists, else at start
        gemini_history.insert(0, {"role": "user",  "parts": [{"text": "Context?"}]})
        gemini_history.insert(1, {"role": "model", "parts": [{"text": rag_injection}]})

    # ── Gemini API call ────────────────────────────────────────────────────
    try:
        model = genai.GenerativeModel(
            model_name=CHAT_MODEL,
            system_instruction=system_instruction,
        )
        chat_session = model.start_chat(history=gemini_history)
        gemini_response = chat_session.send_message(
            message,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024,
            ),
        )
        bot_response = gemini_response.text.strip()

    except Exception as exc:
        logger.error(f"Gemini API call failed: {exc}")
        raise HTTPException(status_code=502, detail=f"AI service error: {str(exc)}")

    # ── E: Persist to DB ────────────────────────────────────────────────
    save_message(db, user_id, "user",      message,      language)
    bot_msg = save_message(db, user_id, "assistant", bot_response, language)

    # Update turn count
    profile.turn_count = (profile.turn_count or 0) + 1
    db.commit()

    turn_count = profile.turn_count

    # Async: store in ChromaDB
    background_tasks.add_task(
        _background_store,
        user_id=user_id,
        user_message=message,
        bot_response=bot_response,
        message_id=bot_msg.id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # ── F: Profile update every N turns ─────────────────────────────────
    profile_updated = False
    if should_run_profile_update(turn_count, PROFILE_UPDATE_INTERVAL):
        background_tasks.add_task(_background_profile_update, user_id)
        profile_updated = True
        logger.info(f"Profile update scheduled for user {user_id} (turn {turn_count})")

    return ChatResponse(
        response=bot_response,
        relevant_retrieved_history_count=len(rag_hits),
        turn_count=turn_count,
        profile_updated=profile_updated,
    )


# ---------------------------------------------------------------------------
# History endpoint
# ---------------------------------------------------------------------------

@app.get("/api/history/{user_id}", response_model=HistoryResponse)
def get_history(user_id: str, db: Session = Depends(get_db)):
    """Return full chat history for a user, oldest-first."""
    messages = get_all_messages(db, user_id)
    return HistoryResponse(
        user_id=user_id,
        messages=[m.to_dict() for m in messages],
        total=len(messages),
    )


# ---------------------------------------------------------------------------
# Profile endpoint
# ---------------------------------------------------------------------------

@app.get("/api/profile/{user_id}", response_model=ProfileResponse)
def get_profile(user_id: str, db: Session = Depends(get_db)):
    """Return the structured business profile for a user."""
    profile = get_or_create_profile(db, user_id)
    return ProfileResponse(user_id=user_id, profile=profile.to_dict())


# ---------------------------------------------------------------------------
# Memory wipe (GDPR / privacy)
# ---------------------------------------------------------------------------

@app.delete("/api/memory/{user_id}")
def delete_memory(user_id: str, db: Session = Depends(get_db)):
    """Delete all vector memory for a user (irreversible)."""
    deleted_count = delete_user_memory(user_id)
    return {"user_id": user_id, "vector_records_deleted": deleted_count}


# ---------------------------------------------------------------------------
# Admin stats
# ---------------------------------------------------------------------------

@app.get("/api/stats")
def stats():
    """Return ChromaDB collection stats."""
    return get_collection_stats()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model":  CHAT_MODEL,
        "time":   datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------

# NOTE: StaticFiles mount is registered AFTER all API routes so that
# /api/* routes are never intercepted by the catch-all file server.
if (FRONTEND_DIR / "style.css").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found. Place index.html in /frontend</h1>", status_code=404)


RESERVED_ROUTES = {"docs", "redoc", "openapi.json"}

@app.get("/{filename}")
async def serve_static(filename: str):
    if filename in RESERVED_ROUTES:
        raise HTTPException(status_code=404, detail="File not found")
    file_path = FRONTEND_DIR / filename
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    raise HTTPException(status_code=404, detail="File not found")


# ---------------------------------------------------------------------------
# Entry point for local dev
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info",
    )
