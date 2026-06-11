"""
Detects the language and converts to english for llm processing and response generation. Translates the final response into detected language if detected language is not english.
Uses CTranslate2 for high concurrency scaling
"""

import langdetect
from langdetect import detect, DetectorFactory
import langdetect.lang_detect_exception
import ctranslate2
import transformers
import os, threading, asyncio
import torch

MODEL_NAME = "facebook/nllb-200-distilled-600M"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# If running locally without docker, it falls back to local directory
CT_MODEL_DIR = "/models/nllb_ctranslate2" if os.path.exists("/models") else os.path.join(CURRENT_DIR, "nllb_ctranslate2")

NLLB_MAP = {
    "English": "eng_Latn", "Spanish": "spa_Latn", "French": "fra_Latn", 
    "German": "deu_Latn", "Hindi": "hin_Deva", "Arabic": "arb_Arab", 
    "Chinese": "zho_Hans", "Japanese": "jpn_Jpan", "Russian": "rus_Cyrl", 
    "Portuguese": "por_Latn", "Italian": "ita_Latn"
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
compute_type = "float16" if device == "cuda" else "int8"

translator = ctranslate2.Translator(
    CT_MODEL_DIR,
    device = device,
    compute_type = compute_type,
    inter_threads = 8,
    intra_threads = 1
)

thread_local = threading.local()

def get_tokenizer():
    """Returns a thread-local instance of the Hugging Face tokenizer."""
    if not hasattr(thread_local, "tokenizer"):
        thread_local.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    return thread_local.tokenizer


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

def translate(text: str,src: str, tgt: str) -> str:
    """Synchronous translation core running natively in c++"""
    if src == tgt:
        return text
    
    src_code = NLLB_MAP.get(src, "eng_Latn")
    tgt_code = NLLB_MAP.get(tgt, "eng_Latn")
    
    tokenizer = get_tokenizer()
    tokenizer.src_lang = src_code
    
    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    
    results = translator.translate_batch([source_tokens], target_prefix=[[tgt_code]])
    target_tokens = results[0].hypotheses[0]
    
    if tgt_code in target_tokens:
        target_tokens = [t for t in target_tokens if t != tgt_code]
        
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens), skip_special_tokens=True)    


async def translate_to_english(text: str, source_lang: str) -> str:
    """Asynchronous wrapper to offload CPU/GPU execution from FastAPI event loop."""
    return await asyncio.to_thread(translate, text, source_lang, "English")

async def translate_to_native(text: str, target_lang: str) -> str:
    """Asynchronous wrapper to offload CPU/GPU execution from FastAPI event loop."""
    return await asyncio.to_thread(translate, text, "English", target_lang)