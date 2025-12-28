# Mood Playlist (Text-first, FA/EN)

A multimodal, mood-based recommender system designed for text input in **English** and **Persian**. It predicts the user's mood from a short note about their day and suggests songs or movies that match that emotional state.

The project features a hybrid classification engine (Rule-based + ML), a responsive bilingual UI with dark/light modes, and a built-in analytics dashboard.

## 🚀 Features

* **Bilingual NLP Engine**: Automatically detects language (English/Persian) and routes text to the appropriate analyzer.
* **Hybrid Classification**:
    * **Lightweight Mode**: Uses a keyword-based lexicon (no heavy downloads required).
    * **ML Mode**: Uses Transformer models (DistilBERT) for high-accuracy emotion detection if dependencies are installed.
* **Unified Mood System**: Maps various model labels to 6 core moods: `joy`, `sadness`, `anger`, `fear`, `neutral`, `excitement`.
* **Recommendation Engine**: Filters a curated JSON catalog of songs and movies based on mood, media type, and language preference.
* **Modern UI**:
    * FastAPI + Jinja2 frontend.
    * **Analytics Dashboard**: Visualizes mood distribution and catalog statistics using Chart.js.
    * **Dark/Light Themes**: Toggleable visual themes.
    * **RTL/LTR Support**: Proper layout adjustments for Persian vs. English.

## 📸 Screenshots

| **Dark Theme (English)** | **Light Theme (Persian)** |
|:------------------------:|:-------------------------:|
| ![Dark Theme UI](screenshots/dark-theme.png) | ![Light Theme UI](screenshots/light-theme.png) |

## 📂 Project Structure

```text
Mood_Playlist/
├── mood_playlist/
│   ├── data/
│   │   └── catalog.json          # Curated database of songs/movies
│   ├── static/
│   │   └── styles.css            # CSS variables for themes & layout
│   ├── templates/
│   │   └── index.html            # Main UI with Jinja2 templating
│   ├── __init__.py               # Defines SUPPORTED_MOODS
│   ├── app.py                    # FastAPI application & UI routes
│   ├── emotion_classifier.py     # NLP logic (Lexicon + Transformers)
│   ├── language.py               # Language detection (Regex + langdetect)
│   ├── mapping.py                # Maps raw model labels to core moods
│   ├── recommender.py            # Catalog filtering logic
│   └── service.py                # Orchestrator for detection & recommendation
├── requirements.txt              # Core dependencies
├── requirements-ml.txt           # Optional ML dependencies (Torch/Transformers)
└── README.md
```

## Quickstart
1) Create a venv and install base deps:
```bash
pip install -r requirements.txt
```
2) (To use transformer model) install ML deps:
```bash
pip install -r requirements-ml.txt
# (requirements-ml already pins torch==2.9.1 CPU from the official PyTorch index)
# If you want to install it manually instead:
# pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

3) Run the FastAPI app:
```bash
uvicorn mood_playlist.app:app --reload
```
Open http://localhost:8000. Toggle "Use transformer models" in the UI when you’ve downloaded models.

## Models (recommended, all free)
These are referenced by name only; download/caching happens via `transformers` when the env var `USE_TRANSFORMERS=1` is set (UI toggle sets it for the process).
- English: `bhadresh-savani/distilbert-base-uncased-emotion` (6 emotions, small)
- Persian: `Negark/distilbert-fa-armanemo`
If you prefer smaller footprints, swap in other text-emotion models and update `DEFAULT_MODELS` in `mood_playlist/emotion_classifier.py`.

## How it works
- `LanguageDetector` wraps `langdetect` to pick FA vs EN (falls back to EN).
- `TextEmotionClassifier` chooses a model per language when `USE_TRANSFORMERS=1` **and** `transformers` is installed & models are cached. Otherwise a fallback keyword lexicon handles both languages.
- Model labels are mapped into the unified mood set; unknown labels → `neutral`.
- `Recommender` filters the catalog (`mood_playlist/data/catalog.json`) by mood and media type and returns a small, shuffled list.

## Extending
- Add more moods: update `SUPPORTED_MOODS` + mapping tables and tag new catalog entries.
- Add more items: append to `data/catalog.json` with `title`, `creator`, `type`, `mood`, `language`.
- Swap in your own models: change `DEFAULT_MODELS` and label mappings.
- Add persistence / APIs: hook the core into databases or external content APIs (keep tokens out of the repo; use env vars).

## Security & publishing notes
- No private APIs or secrets are checked in. Keep any API keys in environment variables if you later add providers.
- The default path uses offline-friendly heuristics; transformer models only load when present locally.

## Dev/testing
- Lint/type-check as you prefer (`ruff`, `mypy`).
- Minimal unit tests can be run via `pytest`
