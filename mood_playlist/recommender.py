import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict

from mood_playlist import SUPPORTED_MOODS

MediaType = Literal["song", "movie"]


def _load_catalog() -> List[dict]:
    """
    Load catalog from env var MOOD_DATA_DIR if set, otherwise default location.
    """
    env_dir = os.getenv("MOOD_DATA_DIR")
    if env_dir:
        data_path = Path(env_dir) / "catalog.json"
    else:
        data_path = Path(__file__).parent / "data" / "catalog.json"
        
    if not data_path.exists():
        return []

    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Recommendation:
    title: str
    creator: str
    type: MediaType
    mood: str
    language: str


class Recommender:
    def __init__(self, catalog: Optional[List[dict]] = None):
        self.catalog = catalog or _load_catalog()

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Calculate distribution stats for the dashboard."""
        moods = {}
        languages = {}
        
        for item in self.catalog:
            # Count Moods
            m = item.get("mood", "neutral")
            moods[m] = moods.get(m, 0) + 1
            
            # Count Languages
            l = item.get("language", "en")
            languages[l] = languages.get(l, 0) + 1
            
        return {"moods": moods, "languages": languages}

    def recommend(
        self,
        mood: str,
        media_type: Optional[MediaType] = None,
        limit: int = 6,
        response_language: Optional[str] = None,
    ) -> List[Recommendation]:
        if mood not in SUPPORTED_MOODS:
            mood = "neutral"
        
        # Filter by mood
        filtered = [item for item in self.catalog if item.get("mood") == mood]
        
        # Filter by media type
        if media_type:
            filtered = [item for item in filtered if item.get("type") == media_type]
            
        # Filter by content language
        if response_language in {"fa", "en"}:
            filtered = [item for item in filtered if item.get("language") == response_language]
            
        random.shuffle(filtered)
        picks = filtered[:limit]
        return [Recommendation(**item) for item in picks]