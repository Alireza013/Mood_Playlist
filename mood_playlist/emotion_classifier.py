import os
import json
from typing import Dict, Optional, Callable, Any, cast

from mood_playlist import SUPPORTED_MOODS
from huggingface_hub import snapshot_download
from mood_playlist.mapping import map_label_to_mood

DEFAULT_MODELS = {
    "en": "bhadresh-savani/distilbert-base-uncased-emotion",
    "fa": "Negark/distilbert-fa-armanemo",
}

ARMANEMO_LABELS = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "anger",
    4: "neutral",
    5: "sadness",
    6: "excitement"
}

class TextEmotionClassifier:
    def __init__(self, use_transformers: bool = False, model_names: Optional[Dict[str, str]] = None):
        self.use_transformers = use_transformers
        self.model_names = model_names or DEFAULT_MODELS
        self.pipelines: Dict[str, Callable[[str], Any]] = {}
        self.id2label: Dict[str, Dict[int, str]] = {}
        
        if use_transformers:
            self._load_pipelines()

    def _load_pipelines(self) -> None:
        try:
            from transformers import pipeline
        except ImportError:
            print("❌ Transformers library not found. Falling back to lexicon.")
            self.use_transformers = False
            return

        print("⏳ Loading ML models (this may take a while initially)...")
        for lang, model_name in self.model_names.items():
            try:
                print(f"   ...loading {lang} model: {model_name}")
                pipe = self._load_patched_pipeline(model_name, lang)
                self.pipelines[lang] = pipe
                
                if lang == "fa":
                    self.id2label[lang] = ARMANEMO_LABELS
                    print(f"      🛠️ Applied manual labels for {lang}: {self.id2label[lang]}")
                elif hasattr(pipe.model, "config") and hasattr(pipe.model.config, "id2label"):
                    self.id2label[lang] = cast(Dict[int, str], pipe.model.config.id2label)
                    print(f"      🏷️ Model labels for {lang}: {self.id2label[lang]}")
                
                print(f"   ✅ {lang} model loaded successfully.")
            except Exception as e:
                print(f"   ⚠️ Failed to load {lang} model ({model_name}): {e}")
                print("      -> Falling back to lexicon for this language.")
                self.pipelines.pop(lang, None)
        
        if not self.pipelines:
            self.use_transformers = False

    def _load_patched_pipeline(self, model_name: str, lang: str):
        from transformers import pipeline
        model_path = snapshot_download(repo_id=model_name)
        config_path = os.path.join(model_path, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if "add_special_tokens" in config:
                del config["add_special_tokens"]
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)

        return pipeline(
            "text-classification",
            model=model_path, 
            top_k=None, 
            truncation=True,
            use_fast=True 
        )

    def predict(self, text: str, language: str) -> Dict[str, str]:
        cleaned = (text or "").strip()
        if not cleaned:
            return {"label": "neutral", "mood": "neutral", "source": "fallback"}

        lex_label = self._lexicon_label(cleaned, language)
        lex_mood = map_label_to_mood(language, lex_label)

        if self.use_transformers and language in self.pipelines:
            try:
                outputs = self.pipelines[language](cleaned)
                result = outputs[0] if outputs and isinstance(outputs[0], dict) else outputs[0][0]
                raw_label = result.get("label", "neutral") if isinstance(result, dict) else "neutral"
                
                resolved_label = self._resolve_label(raw_label, language)
                
                print(f"🔍 [DEBUG] Lang: {language} | Raw: {raw_label} -> Resolved: {resolved_label} | Lexicon: {lex_mood}")

                mood = map_label_to_mood(language, resolved_label)
                chosen_label, chosen_mood, source = self._resolve_with_lexicon(resolved_label, mood, lex_label, lex_mood)
                return {"label": chosen_label, "mood": chosen_mood, "source": source}
            except Exception as e:
                print(f"Prediction error for {language}: {e}")
                pass

        return {"label": lex_label, "mood": lex_mood, "source": "lexicon"}

    def _resolve_label(self, label: str, lang: str) -> str:
        if label.upper().startswith("LABEL_") and lang in self.id2label:
            try:
                idx = int(label.split("_")[-1])
                return self.id2label[lang].get(idx, label)
            except ValueError:
                return label
        return label

    @staticmethod
    def _resolve_with_lexicon(tf_label: str, tf_mood: str, lex_label: str, lex_mood: str) -> tuple[str, str, str]:
        if tf_mood == lex_mood:
            return tf_label, tf_mood, "transformer"
        
        if lex_mood in {"anger", "sadness", "fear"} and tf_mood in {"neutral", "joy", "excitement"}:
            return lex_label, lex_mood, "lexicon_override_neg"

        if lex_mood in {"joy", "excitement"} and tf_mood in {"sadness", "anger", "fear", "neutral"}:
             return lex_label, lex_mood, "lexicon_override_pos"
            
        return tf_label, tf_mood, "transformer"

    def _lexicon_label(self, text: str, language: str) -> str:
        txt = text.lower()
        lexicon = PERSIAN_LEXICON if language == "fa" else ENGLISH_LEXICON
        scores = {label: 0 for label in lexicon}
        for label, keywords in lexicon.items():
            for word in keywords:
                if word in txt:
                    scores[label] += 1
        best_label = max(scores.items(), key=lambda x: x[1])[0]
        if scores[best_label] == 0:
            return "neutral"
        return best_label


ENGLISH_LEXICON = {
    "joy": ["happy", "joy", "grateful", "great", "awesome", "good", "love"],
    "sadness": ["sad", "down", "blue", "upset", "depressed", "cry"],
    "anger": ["angry", "mad", "furious", "irritated", "rage", "hate"],
    "fear": ["afraid", "scared", "anxious", "worried", "nervous"],
    "neutral": ["ok", "fine", "meh", "normal"],
    "excitement": ["excited", "hyped", "stoked", "thrilled", "pumped"],
}

PERSIAN_LEXICON = {
    "joy": ["خوشحال", "شاد", "خوب", "عالی", "راضی", "عشق", "خنده"],
    "sadness": ["غمگین", "ناراحت", "بی حوصله", "دلگیر", "گریه", "غصه", "بدبخت"],
    "anger": ["عصبانی", "خشم", "حرص", "کلافه", "متنفر", "دعوا"],
    "fear": ["می ترسم", "ترس", "استرس", "نگران", "وحشت"],
    "neutral": ["معمولی", "بد نیست", "اوکی", "نرمال", "سلام"],
    "excitement": ["هیجان", "متحمس", "ذوق", "انرژی", "باحال"]
}