"""
Detects the language and converts to english for llm processing and response generation. Translates the final response into detected language if detected language is not english.
Uses CTranslate2 for high concurrency scaling
"""

import langdetect
from langdetect import detect, DetectorFactory
import langdetect.lang_detect_exception
import ctranslate2
import transformers
import os, threading, asyncio, queue, time
import torch
import subprocess
import tempfile, shutil
from core.logger import get_logger

logger = get_logger(__name__)

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass 

MODEL_NAME = "facebook/nllb-200-distilled-600M"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CT_MODEL_DIR = "/models/nllb_ctranslate2" if os.path.exists("/models") else os.path.join(CURRENT_DIR, "nllb_ctranslate2")

# --- AUTOMATED ONE-TIME SETUP AND CONVERSION ---
# if not os.path.exists(CT_MODEL_DIR):
# if not os.path.exists(os.path.join(CT_MODEL_DIR, "model.bin")):    
#     print(f"[SYSTEM] CTranslate2 model not found at: {CT_MODEL_DIR}")
#     print("[SYSTEM] Initiating automated download and conversion to Float16 (this occurs only once)...")
#     try:
#         subprocess.run([
#             "python3", "-m", "ctranslate2.converters.transformers",
#             "--model", MODEL_NAME,
#             "--output_dir", CT_MODEL_DIR,
#             "--quantization", "float16"
#         ], check=True)
#         print("[SYSTEM] Model download and conversion completed successfully!")
#     except Exception as e:
#         print(f"[SYSTEM] ❌ Automated model conversion failed: {e}")
#         if os.path.exists(CT_MODEL_DIR):
#             shutil.rmtree(CT_MODEL_DIR)
#         raise e

if not os.path.exists(os.path.join(CT_MODEL_DIR, "model.bin")):
    print(f"[SYSTEM] CTranslate2 model not found at: {CT_MODEL_DIR}")
    print("[SYSTEM] Initiating automated download and conversion to Float16 (this occurs only once)...")
    
    # 1. Ensure the final target directory exists
    os.makedirs(CT_MODEL_DIR, exist_ok=True)
    # tmp_dir = tempfile.mkdtemp()
    tmp_dir = tempfile.mkdtemp(prefix="nllb_convert_")
    
    try:
        subprocess.run([
            "python3", "-m", "ctranslate2.converters.transformers",
            "--model", MODEL_NAME,
            "--output_dir", tmp_dir,
            "--quantization", "float16",
            "--force"                
        ], check=True)
        # Now copy the converted files into the mounted volume
        for f in os.listdir(tmp_dir):
            shutil.copy2(os.path.join(tmp_dir, f), CT_MODEL_DIR)
            
        print("[SYSTEM] Model download and conversion completed successfully!")
    except Exception as e:
        print(f"[SYSTEM] ❌ Automated model conversion failed: {e}")
        # REMOVED shutil.rmtree — never delete a mounted volume
        raise e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # clean up temp

# -----------------------------------------------

#-----------------------------------------------------------------------------
# SUPPORTED LANGUAGES — single source of truth
#-----------------------------------------------------------------------------
# Deliberately scoped to 5 major international languages instead of a wide
# spread of locales. Every language here is fully supported end-to-end:
# text translation (NLLB), langdetect auto-detection, Whisper STT hints,
# and browser TTS (SpeechSynthesisUtterance.lang / BCP-47 locale).
# Anything outside this list soft-falls-back to English with a UI warning
# rather than silently mistranslating or erroring.
#
# "whisper" codes are ISO-639-1, used as the `language=` hint to faster-whisper.
# "bcp47" is the locale tag the frontend sets on SpeechSynthesisUtterance so
# the browser picks a matching voice for read-aloud.
#-----------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    "English":    {"nllb": "eng_Latn", "detect_codes": ["en"], "whisper": "en", "bcp47": "en-US"},
    "Spanish":    {"nllb": "spa_Latn", "detect_codes": ["es"], "whisper": "es", "bcp47": "es-ES"},
    "French":     {"nllb": "fra_Latn", "detect_codes": ["fr"], "whisper": "fr", "bcp47": "fr-FR"},
    "German":     {"nllb": "deu_Latn", "detect_codes": ["de"], "whisper": "de", "bcp47": "de-DE"},
    "Portuguese": {"nllb": "por_Latn", "detect_codes": ["pt"], "whisper": "pt", "bcp47": "pt-PT"},
    "Hindi":      {"nllb": "hin_Deva", "detect_codes": ["hi"], "whisper": "hi", "bcp47": "hi-IN"},
}

NLLB_MAP = {name: cfg["nllb"] for name, cfg in SUPPORTED_LANGUAGES.items()}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
compute_type = "float16" if device == "cuda" else "int8"

print("=" * 60)
print(f"[CUDA DIAGNOSTIC] PyTorch version: {torch.__version__}")
print(f"[CUDA DIAGNOSTIC] Is CUDA available to PyTorch?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[CUDA DIAGNOSTIC] Detected GPU: {torch.cuda.get_device_name(0)}")
print(f"[CUDA DIAGNOSTIC] CTranslate2 is loading model on: {device.upper()}")
print(f"[CUDA DIAGNOSTIC] CTranslate2 compute type: {compute_type}")
print("=" * 60)
logger.info("Loading NLLB translation model | device=%s compute_type=%s", device, compute_type)

translator = ctranslate2.Translator(
    CT_MODEL_DIR,
    device=device,
    compute_type=compute_type,
    inter_threads=4,
    intra_threads=1 if device == "cuda" else 2,
    max_queued_batches=-1,
)

# ─── CHANGE 1: Replace thread_local + get_tokenizer with TokenizerPool ───────
# REMOVED: thread_local = threading.local()
# REMOVED: def get_tokenizer(): ...

TOKENIZER_POOL_SIZE = 4  # Must match inter_threads and max_parallel_batches

class TokenizerPool:
    def __init__(self, model_name: str, size: int):
        self._pool = queue.Queue()
        print(f"[SYSTEM] Pre-warming {size} tokenizer instances...", flush=True)
        for i in range(size):
            tok = transformers.AutoTokenizer.from_pretrained(model_name)
            self._pool.put(tok)
            print(f"[SYSTEM] Tokenizer {i+1}/{size} ready", flush=True)
        print(f"[SYSTEM] ✅ Tokenizer pool ready.", flush=True)
        logger.info("Tokenizer pool ready | size=%d", size)

    def acquire(self):
        return self._pool.get()  # Blocks if all in use — correct behavior

    def release(self, tokenizer):
        self._pool.put(tokenizer)

# Initialized once at import time — cold start paid here, never again
tokenizer_pool = TokenizerPool(MODEL_NAME, size=TOKENIZER_POOL_SIZE)
# ─────────────────────────────────────────────────────────────────────────────

DetectorFactory.seed = 0 

LANG_MAP = {
    code: name
    for name, cfg in SUPPORTED_LANGUAGES.items()
    for code in cfg["detect_codes"]
}

def detect_language(text: str) -> tuple[str, bool]:
    """
    Detects the language of `text` and reports whether it's one we actually
    support end-to-end.

    Returns (language_name, is_supported):
      - Detected + supported        -> (name, True)
      - Detected but NOT supported  -> ("English", False)  <- soft fallback,
        caller should surface a "language not supported, showing English"
        warning banner instead of silently mistranslating.
      - Detection failed (text too short/ambiguous, e.g. "ok", "3", "")
                                     -> ("English", True)   <- not a real
        unsupported-language case, so no warning is shown.
    """
    try:
        lang_code = detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        return "English", True

    name = LANG_MAP.get(lang_code)
    if name:
        return name, True
    return "English", False


def is_language_supported(language_name: str) -> bool:
    return language_name in SUPPORTED_LANGUAGES


def get_bcp47(language_name: str) -> str:
    """BCP-47 locale tag for TTS (SpeechSynthesisUtterance.lang). Falls back to en-US."""
    return SUPPORTED_LANGUAGES.get(language_name, SUPPORTED_LANGUAGES["English"])["bcp47"]


def get_whisper_code(language_name: str) -> str | None:
    """ISO-639-1 code for the Whisper STT language hint. None means auto-detect."""
    cfg = SUPPORTED_LANGUAGES.get(language_name)
    return cfg["whisper"] if cfg else None


def list_supported_languages() -> list[dict]:
    """Full locale info for the frontend's /languages dropdown — one source, no duplicate hardcoded map in JS."""
    return [
        {"name": name, "bcp47": cfg["bcp47"], "whisper": cfg["whisper"]}
        for name, cfg in SUPPORTED_LANGUAGES.items()
    ]


# ─── DYNAMIC BATCHER ─────────────────────────────────────────────────────────

class AsyncDynamicBatcher:
    def __init__(self, max_batch_size: int = 32, batch_timeout_ms: float = 5.0, max_parallel_batches: int = 4):
        self.max_batch_size = max_batch_size
        self.batch_timeout_s = batch_timeout_ms / 1000.0
        self.max_parallel_batches = max_parallel_batches
        self.queue = []
        self.lock = asyncio.Lock()
        self._active_workers = 0

    async def add_request(self, text: str, src: str, tgt: str) -> str:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        async with self.lock:
            self.queue.append({"text": text, "src": src, "tgt": tgt, "future": future})
            if self._active_workers < self.max_parallel_batches:
                self._active_workers += 1
                asyncio.create_task(self._batch_worker(skip_wait=False))
                
        return await future

    # ─── CHANGE 2: Add skip_wait parameter to avoid unnecessary 5ms sleep ────
    async def _batch_worker(self, skip_wait: bool = False):
        if not skip_wait:
            await asyncio.sleep(self.batch_timeout_s)
        
        async with self.lock:
            if not self.queue:
                self._active_workers -= 1
                return
            
            batch_requests = self.queue[:self.max_batch_size]
            self.queue = self.queue[self.max_batch_size:]
            
            # ─── CHANGE 3: Immediately spawn sibling workers for remaining items ──
            # Previous code only spawned in `finally` — after current batch finished.
            # Now we spawn immediately so all 4 translator slots run in parallel.
            while self.queue and self._active_workers < self.max_parallel_batches:
                self._active_workers += 1
                asyncio.create_task(self._batch_worker(skip_wait=True))  # No wait — queue already full
        # ─────────────────────────────────────────────────────────────────────
        
        print(f"[BATCHER] Worker firing | batch_size={len(batch_requests)} | active_workers={self._active_workers} | remaining_queue={len(self.queue)}", flush=True)
        logger.debug(
            "Batch worker firing | batch_size=%d active_workers=%d remaining_queue=%d",
            len(batch_requests), self._active_workers, len(self.queue)
        )
        
        try:
            results = await asyncio.to_thread(self._execute_translation_batch, batch_requests)
            
            for req, translated_text in zip(batch_requests, results):
                if not req["future"].done():
                    req["future"].set_result(translated_text)
        except Exception as e:
            for req in batch_requests:
                if not req["future"].done():
                    req["future"].set_exception(e)
        finally:
            async with self.lock:
                self._active_workers -= 1
                if self.queue and self._active_workers < self.max_parallel_batches:
                    self._active_workers += 1
                    asyncio.create_task(self._batch_worker(skip_wait=True))

    # ─── CHANGE 4: Replace get_tokenizer() with tokenizer_pool ───────────────
    def _execute_translation_batch(self, batch_requests) -> list[str]:
        t0 = time.time()
        tokenizer = tokenizer_pool.acquire()  # Instant — pre-warmed, no init cost
        t1 = time.time()
        
        print(f"[CT2] thread={threading.current_thread().name} | tokenizer_acquire={t1-t0:.4f}s | batch_size={len(batch_requests)} | translator_active={translator.num_active_batches}", flush=True)
        
        try:
            source_batch = []
            target_prefixes = []
            
            for req in batch_requests:
                src_code = NLLB_MAP.get(req["src"], "eng_Latn")
                tgt_code = NLLB_MAP.get(req["tgt"], "eng_Latn")
                tokenizer.src_lang = src_code
                tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(req["text"]))
                source_batch.append(tokens)
                target_prefixes.append([tgt_code])
            
            t2 = time.time()
            print(f"[CT2] tokenization={t2-t1:.4f}s", flush=True)
            
            translated_results = translator.translate_batch(
                source_batch,
                target_prefix=target_prefixes,
                max_decoding_length=256,
                beam_size=1,
                asynchronous=False,
            )
            
            t3 = time.time()
            print(f"[CT2] ct2_inference={t3-t2:.4f}s", flush=True)
            
            decoded_results = []
            for req, res in zip(batch_requests, translated_results):
                tgt_code = NLLB_MAP.get(req["tgt"], "eng_Latn")
                target_tokens = res.hypotheses[0]
                if tgt_code in target_tokens:
                    target_tokens = [t for t in target_tokens if t != tgt_code]
                decoded = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(target_tokens),
                    skip_special_tokens=True
                )
                decoded_results.append(decoded)
            
            t4 = time.time()
            print(f"[CT2] decoding={t4-t3:.4f}s | total={t4-t0:.4f}s", flush=True)
            logger.debug(
                "CT2 batch timing | batch_size=%d tokenize_s=%.4f inference_s=%.4f decode_s=%.4f total_s=%.4f",
                len(batch_requests), t2 - t1, t3 - t2, t4 - t3, t4 - t0
            )
            
            return decoded_results
        
        finally:
            tokenizer_pool.release(tokenizer)  # Always return — even on exception
    # ─────────────────────────────────────────────────────────────────────────

dynamic_batcher = AsyncDynamicBatcher(max_batch_size=8, batch_timeout_ms=5.0, max_parallel_batches=4)

# ─── END DYNAMIC BATCHER ─────────────────────────────────────────────────────

async def translate_to_english(text: str, source_lang: str) -> str:
    return await dynamic_batcher.add_request(text, source_lang, "English")

async def translate_to_native(text: str, target_lang: str) -> str:
    return await dynamic_batcher.add_request(text, "English", target_lang)