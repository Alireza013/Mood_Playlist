from pathlib import Path
from typing import Optional, Dict, cast
from contextlib import asynccontextmanager

from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool

from mood_playlist.service import MoodService
from mood_playlist.recommender import MediaType

# Global state to hold our service instance
ml_models: Dict[str, MoodService] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load ML models on startup and clean up on shutdown.
    This prevents reloading the model for every single request.
    """
    print("Loading MoodService and ML models...")
    ml_models["service"] = MoodService(use_transformers=True)
    yield
    # Clean up resources if needed
    ml_models.clear()
    print("MoodService unloaded.")

BASE_DIR = Path(__file__).parent
app = FastAPI(title="Mood Playlist", version="0.1.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

UI_TEXT: Dict[str, Dict[str, object]] = {
    "en": {
        "eyebrow": "Text-based Playlist • Persian & English",
        "headline": "Tell me about your day.",
        "accent": "Get a mood-based playlist.",
        "lede": "Write a short note, we infer the mood, and suggest songs or movies.",
        "text_label": "Text",
        "placeholder": "I felt a bit anxious before the meeting but it went well.",
        "type_label": "Type",
        "type_any": "Songs + Movies",
        "type_song": "Songs only",
        "type_movie": "Movies only",
        "cta": "Analyze & Recommend",
        "detected_language": "Detected language",
        "mood": "Mood",
        "model_meta": "model label",
        "source_meta": "source",
        "lang_label": "Language",
        "lang_en": "English",
        "lang_fa": "فارسی",
        "cards_lang": "lang",
        "response_lang_label": "Response language",
        "response_lang_any": "Persian & English",
        "response_lang_fa": "Persian",
        "response_lang_en": "English",
        "moods": ["joy", "sadness", "anger", "fear", "neutral", "excitement"],
        "mood_labels": {
            "joy": "joy",
            "sadness": "sadness",
            "anger": "anger",
            "fear": "fear",
            "neutral": "neutral",
            "excitement": "excitement",
        },
        "language_labels": {
            "en": "English",
            "fa": "Persian"
        },
        # Analytics Section
        "analytics_title": "Dataset Analytics",
        "chart_mood": "Mood Distribution",
        "chart_lang": "Content by Language",
    },
    "fa": {
        "eyebrow": "پلی لیست متن محور • فارسی و انگلیسی",
        "headline": "از روزت بگو.",
        "accent": "یک پلی‌لیست بر اساس مودت بگیر.",
        "lede": "یک یادداشت کوتاه بنویس، مود را تشخیص می‌دهیم و آهنگ/فیلم پیشنهاد می‌کنیم.",
        "text_label": "متن",
        "placeholder": "امروز خیلی پرانرژی بودم و کارهام را زودتر از موعد انجام دادم.",
        "type_label": "نوع محتوا",
        "type_any": "آهنگ و فیلم",
        "type_song": "فقط آهنگ",
        "type_movie": "فقط فیلم",
        "cta": "تحلیل و پیشنهاد",
        "detected_language": "زبان تشخیص‌داده‌شده",
        "mood": "مود",
        "model_meta": "برچسب مدل",
        "source_meta": "منبع",
        "lang_label": "زبان",
        "lang_en": "English",
        "lang_fa": "فارسی",
        "cards_lang": "زبان",
        "response_lang_label": "زبان پاسخ",
        "response_lang_any": "فارسی و انگلیسی",
        "response_lang_fa": "فارسی",
        "response_lang_en": "English",
        "moods": ["شادی", "غم", "خشم", "ترس", "طبیعی", "هیجان"],
        "mood_labels": {
            "joy": "شادی",
            "sadness": "غم",
            "anger": "خشم",
            "fear": "ترس",
            "neutral": "طبیعی",
            "excitement": "هیجان",
        },
        "language_labels": {
            "en": "انگلیسی",
            "fa": "فارسی"
        },
        # Analytics Section
        "analytics_title": "آمار دیتاست",
        "chart_mood": "توزیع مودها",
        "chart_lang": "محتوا بر اساس زبان",
    },
}


def resolve_ui_lang(lang: Optional[str]) -> str:
    return "fa" if lang == "fa" else "en"


def resolve_theme(theme: Optional[str]) -> str:
    return "light" if theme == "light" else "dark"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, lang: Optional[str] = "en", theme: Optional[str] = "dark"):
    ui_lang = resolve_ui_lang(lang)
    theme_resolved = resolve_theme(theme)
    
    # Fetch stats
    service = ml_models.get("service")
    stats = service.get_stats() if service else {}

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "use_transformers": False,
            "ui": UI_TEXT[ui_lang],
            "ui_lang": ui_lang,
            "theme": theme_resolved,
            "stats": stats,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    text: str = Form(...),
    media_type: Optional[str] = Form(None),
    response_language: Optional[str] = Form("both"),
    ui_lang: Optional[str] = Form("en"),
    theme: Optional[str] = Form("dark"),
    lang: Optional[str] = None,
):
    ui_lang_resolved = resolve_ui_lang(lang or ui_lang)
    theme_resolved = resolve_theme(theme)
    
    # Retrieve the pre-loaded service
    service = ml_models["service"]
    
    mt: Optional[MediaType] = None
    if media_type in {"song", "movie"}:
        mt = cast(MediaType, media_type)

    result = await run_in_threadpool(
        service.analyze_and_recommend,
        text=text,
        media_type=mt,
        response_language=response_language if response_language in {"fa", "en"} else None,
    )
    
    stats = service.get_stats()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "text": text,
            "media_type": media_type,
            "response_language": response_language,
            "ui": UI_TEXT[ui_lang_resolved],
            "ui_lang": ui_lang_resolved,
            "theme": theme_resolved,
            "stats": stats,
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}