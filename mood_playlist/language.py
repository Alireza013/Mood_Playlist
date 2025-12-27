import re
from langdetect import detect, DetectorFactory, LangDetectException

# langdetect has randomness by default; setting seed makes results stable.
DetectorFactory.seed = 42


class LanguageDetector:
    supported = {"fa", "en"}
    _persian_pattern = re.compile(r"[\u0600-\u06FF]")

    def detect(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return "en"
        # Quick heuristic: any Persian script character → fa
        if self._persian_pattern.search(cleaned):
            return "fa"
        try:
            lang = detect(cleaned)
        except LangDetectException:
            return "en"
        return lang if lang in self.supported else "en"
