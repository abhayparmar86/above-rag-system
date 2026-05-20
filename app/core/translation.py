"""
Detects the language and converts to english for llm processing and response generation. Translates the final response into detected language if detected language is not english.
"""

import langdetect
from langdetect import detect, DetectorFactory

# Seed to ensure consistent language detection
DetectorFactory.seed = 0 

LANG_MAP = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", 
    "hi": "Hindi", "ar": "Arabic", "zh-cn": "Chinese", "ja": "Japanese",
    "ru": "Russian", "pt": "Portuguese", "it": "Italian"
}

def detect_language(text: str) -> str:
    """Detects the language of the input text and returns the English name."""
    try:
        lang_code = detect(text)
        return LANG_MAP.get(lang_code, "English")
    except langdetect.lang_detect_exception.LangDetectException:
        # Fallback to English if the text is too short/ambiguous or empty
        return "English"

async def translate_to_english(llm, text: str, source_lang: str) -> str:
    """Translates text from the source language to English using the LLM."""
    prompt = f"""[INST] You are expert translator.Translate the following text from {source_lang} to English.
              STRICT RULE: Output ONLY the translated English Text. Do NOT add quotes, conversational filler, or explanations.
              
              Text: {text}
              English: Translation: [\INST]"""
              
    return (await llm.ainvoke(prompt)).strip()

async def translate_to_native(llm, text: str, target_lang: str) -> str:
    """Translates English text to the target native language using the LLM."""
    prompt = f"""[INST] You are an expert Translator. Translate the following English text into {lang}.
            STRICT RULE: Output ONLY the translated {lang} text. Do NOT add Quotes, conversational filler, or explanations.
            
            English Text: {text}
            {target_lang} Translation: [\INST]"""
            
    return (await llm.ainvoke(prompt)).strip()