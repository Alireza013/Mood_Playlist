from typing import Dict, Optional, TypedDict, List, Any, cast

from mood_playlist.language import LanguageDetector
from mood_playlist.recommender import Recommender, MediaType
from mood_playlist.emotion_classifier import TextEmotionClassifier


class PredictionResult(TypedDict):
    label: str
    mood: str
    source: str


class ServiceResult(TypedDict):
    language: str
    prediction: PredictionResult
    recommendations: List[Dict[str, Any]]


class MoodService:
    def __init__(
        self,
        use_transformers: Optional[bool] = None,
    ):
        # Default to transformers for better accuracy; fallback to lexicon if models not available
        self.use_transformers = True if use_transformers is None else use_transformers
        self.lang_detector = LanguageDetector()
        self.classifier = TextEmotionClassifier(use_transformers=self.use_transformers)
        self.recommender = Recommender()

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        return self.recommender.get_stats()

    def analyze_and_recommend(
        self,
        text: str,
        media_type: Optional[MediaType] = None,
        limit: int = 6,
        response_language: Optional[str] = None,
    ) -> ServiceResult:
        language = self.lang_detector.detect(text)
        prediction = cast(PredictionResult, self.classifier.predict(text, language))
        recommendations = self.recommender.recommend(
            prediction["mood"],
            media_type=media_type,
            limit=limit,
            response_language=response_language,
        )
        rec_dicts: List[Dict[str, Any]] = [rec.__dict__ for rec in recommendations]
        return cast(
            ServiceResult,
            {
                "language": language,
                "prediction": prediction,
                "recommendations": rec_dicts,
            },
        )