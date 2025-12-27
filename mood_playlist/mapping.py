from typing import Dict

SUPPORTED_MOODS = {"joy", "sadness", "anger", "fear", "neutral", "excitement"}

LABEL_MAP: Dict[str, str] = {
    # --- Joy / Happiness ---
    "joy": "joy",
    "happy": "joy",
    "happiness": "joy",
    "love": "joy",
    "cheerful": "joy",
    "delighted": "joy",
    "admiration": "joy",
    
    # --- Sadness ---
    "sad": "sadness",
    "sadness": "sadness",
    "grief": "sadness",
    "sorrow": "sadness",
    "disappointment": "sadness",
    
    # --- Anger / Hatred ---
    "anger": "anger",
    "angry": "anger",
    "rage": "anger",
    "hatred": "anger",
    "hate": "anger",
    "annoyance": "anger",
    "furious": "anger",
    
    # --- Fear ---
    "fear": "fear",
    "scared": "fear",
    "frightened": "fear",
    "anxious": "fear",
    "nervous": "fear",
    
    # --- Excitement / Wonder / Surprise ---
    "excitement": "excitement",
    "excited": "excitement",
    "surprise": "excitement",
    "surprised": "excitement",
    "wonder": "excitement",
    "amazed": "excitement",
    "enthusiasm": "excitement",
    
    # --- Neutral / Other ---
    "neutral": "neutral",
    "no emotion": "neutral",
    "other": "neutral",
}

def map_label_to_mood(language: str, label: str) -> str:
    """Map model label to one of the SUPPORTED_MOODS; default to neutral."""
    label_norm = (str(label) or "").strip().lower()
    
    # 1. Direct lookup
    if label_norm in LABEL_MAP:
        return LABEL_MAP[label_norm]

    # 2. Substring matching
    for key, val in LABEL_MAP.items():
        if key in label_norm:
             return val
                
    return "neutral"