"""
Detects the language and converts to english for llm processing and response generation. Translates the final response into detected language if detected language is not english.
"""

import langdetect
from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

NLLB_MAP = {
    "English": "eng_Latn", "Spanish": "spa_Latn", "French": "fra_Latn", 
    "German": "deu_Latn", "Hindi": "hin_Deva", "Arabic": "arb_Arab", 
    "Chinese": "zho_Hans", "Japanese": "jpn_Jpan", "Russian": "rus_Cyrl", 
    "Portuguese": "por_Latn", "Italian": "ita_Latn"
}

def translate(text: str, src: str, tgt: str) -> str:
    if src == tgt: return text
    tokenizer.src_lang = NLLB_MAP.get(src, "eng_Latn")
    tgt_code = NLLB_MAP.get(tgt, "eng_Latn")
    inputs = tokenizer(text, return_tensors="pt")
    out = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code), max_length=512)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]

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

async def translate_to_english(text: str, source_lang: str) -> str:
    return translate(text, source_lang, "English")

async def translate_to_native(text: str, target_lang: str) -> str:
    return translate(text, "English", target_lang)