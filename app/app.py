"""
====================================================================
@file: app.py
@description: Backend API Server xử lý nhận dạng giọng nói (ASR) bằng Flask. 
              Tích hợp mô hình HuggingFace Wav2Vec2, xử lý âm thanh (FFmpeg/Librosa), 
              nhận diện người nói (Diarizer) và Post-Processing.
@author: Nguyễn Trí Thượng
@project: VietASR Pro
@email: nguyentrithuong471@gmail.com
@github: CheeseThuong
@version: 2.0.0
====================================================================
"""

#from IPython.core import pylabtools
import sys
import logging
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Force UTF-8 for stdout/stderr on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Tải biến môi trường từ file .env (phải chạy trước khi import bất kỳ thứ gì khác)
try:
    from dotenv import load_dotenv as _load_dotenv
    import pathlib as _pathlib
    _env_file = _pathlib.Path(__file__).resolve().parent.parent / ".env"
    _load_dotenv(dotenv_path=_env_file, override=False)
    print(f"[OK] .env loaded from: {_env_file}")
except ImportError:
    print("[WARN] python-dotenv không cài — dùng biến môi trường hệ thống")

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
from io import BytesIO
import time
import torch
import torchaudio
import librosa
import numpy as np
import tempfile
import json
import platform
import traceback
import random
import os 

# FIX: Optimize for low-memory environments (Render Free Tier)
if os.environ.get('RENDER'):
    torch.set_num_threads(1)

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
)
from datasets import load_from_disk
from datetime import datetime
from pathlib import Path
import re
import shutil
from pydub import AudioSegment
from werkzeug.utils import secure_filename
import soundfile as sf
import io

# Import Gemini client tập trung (quản lý rate limit, cache, retry, chunking)
try:
    from gemini_client import (
        summarize_transcript as _gemini_summarize,
        test_gemini_connection as _gemini_test,
        call_gemini as _gemini_call,
    )
    GEMINI_CLIENT_AVAILABLE = True
except ImportError:
    _gemini_summarize = None
    _gemini_test = None
    _gemini_call = None
    GEMINI_CLIENT_AVAILABLE = False

# Import DOCX functionality
try:
    from docx import Document
    from docx.shared import Inches, Pt
except ImportError:
    Document = None
    Inches = None
    Pt = None

# ===================== FFMPEG AUTO-DETECTION =====================
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
_FFMPEG_PATH = os.path.join(_PROJECT_ROOT, "tools", "ffmpeg", "bin", "ffmpeg.exe")
_FFPROBE_PATH = os.path.join(_PROJECT_ROOT, "tools", "ffmpeg", "bin", "ffprobe.exe")

FFMPEG_AVAILABLE = False

if os.path.exists(_FFMPEG_PATH):
    AudioSegment.converter = _FFMPEG_PATH
    if os.path.exists(_FFPROBE_PATH):
        AudioSegment.ffprobe = _FFPROBE_PATH
    ffmpeg_dir = os.path.dirname(_FFMPEG_PATH)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    FFMPEG_AVAILABLE = True
    print(f"[OK] ffmpeg found at: {_FFMPEG_PATH}")
elif shutil.which("ffmpeg"):
    FFMPEG_AVAILABLE = True
    print(f"[OK] ffmpeg found in system PATH: {shutil.which('ffmpeg')}")
else:
    print("[WARN] ffmpeg NOT found! Audio format conversion will not work.")
    print("       Run setup_ffmpeg.bat to install ffmpeg automatically.")
    print("       Or install ffmpeg manually and add to PATH.")

try:
    from post_processor import PostProcessor
except ImportError:
    PostProcessor = None

try:
    from jiwer import wer as compute_wer
except Exception:
    compute_wer = None

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ===== ENCODING CONFIG =====
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.secret_key = 'vietasr-pro-2025'

@app.after_request
def set_utf8_header(response):
    ct = response.content_type
    if ct.startswith('text/html') and 'charset' not in ct:
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response

# Upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg', 'aac', 'wma', 'opus'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Chunking config
CHUNK_DURATION_S = 30   # seconds per chunk
CHUNK_OVERLAP_S = 1     # overlap between chunks
MAX_WORKERS = 4         # parallel chunk processing

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# ===================== LOGGING =====================
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setFormatter(log_formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)

logger = logging.getLogger('VietASR')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# ===================== RATE LIMITING =====================
_rate_limit_store = defaultdict(list)  # IP -> [timestamps]
RATE_LIMIT_MAX = 10   # max requests
RATE_LIMIT_WINDOW = 60  # per 60 seconds

def check_rate_limit(ip):
    now = time.time()
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_limit_store[ip].append(now)
    return True

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables for model
MODEL = None
PROCESSOR = None
DEVICE = None
DEVICE_PREFERENCE = "auto"
MODEL_SOURCE = None
LOCAL_DATASET = None
MODEL_LOAD_STATUS = "not_loaded"  # "not_loaded", "loading", "ready", "error"
MODEL_LOAD_ERROR = None

# === UPGRADE 2025: Đọc cấu hình từ biến môi trường ===
USE_GEMINI = os.environ.get("USE_GEMINI", "false").lower() in ("true", "1", "yes")
DEFAULT_LIVE_MODE = os.environ.get("DEFAULT_LIVE_MODE", "browser")  # 'browser' hoặc 'wav2vec2'
SERVER_VERSION = "2.1.0"

# Post-processing pipeline
POST_PROCESSOR = None
try:
    if PostProcessor:
        POST_PROCESSOR = PostProcessor()
        logger.info("Post-processor pipeline loaded")
except Exception as _pp_err:
    logger.warning(f"Post-processor init failed: {_pp_err}")

# Speaker Diarization
SPEAKER_DIARIZER = None
try:
    from diarizer import SpeakerDiarizer
    SPEAKER_DIARIZER = SpeakerDiarizer()
    if SPEAKER_DIARIZER.available:
        logger.info(f"Speaker Diarizer loaded (backend: {SPEAKER_DIARIZER.backend_name})")
    else:
        logger.warning("Speaker Diarizer: no backend installed")
        SPEAKER_DIARIZER = None
except ImportError:
    logger.warning("Speaker Diarizer not available (diarizer.py not found)")
except Exception as _d_err:
    logger.warning(f"Speaker Diarizer init failed: {_d_err}")

DEFAULT_HF_MODEL_SOURCE = "https://huggingface.co/datasets/doof-ferb/vlsp2020_vinai_100h"


def normalize_hf_source(source):
    """Convert a Hugging Face URL or repo reference to a repo id."""
    if not source:
        return None

    source = source.strip()
    if not source:
        return None

    match = re.search(r"huggingface\.co/(?:datasets/|models/)?([^/?#]+/[^/?#]+)", source)
    if match:
        return match.group(1)

    return source


def get_configured_hf_source():
    """Read the preferred Hugging Face source from environment variables."""
    for env_name in (
        "ASR_MODEL_SOURCE",
        "HUGGINGFACE_MODEL_ID",
        "HF_MODEL_ID",
        "HF_REPO_ID",
        "HUGGINGFACE_MODEL_URL",
        "HF_MODEL_URL",
    ):
        value = os.getenv(env_name)
        if value:
            return normalize_hf_source(value)

    return normalize_hf_source(DEFAULT_HF_MODEL_SOURCE)


def get_local_dataset_path():
    """Return the local dataset export path used for demo samples."""
    web_demo_root = Path(__file__).resolve().parent
    return web_demo_root.parent / "Data" / "train"


def load_local_demo_dataset():
    """Load the exported DatasetDict from Data/train if available."""
    global LOCAL_DATASET

    if LOCAL_DATASET is not None:
        return LOCAL_DATASET

    dataset_path = get_local_dataset_path()
    dataset_dict_file = dataset_path / "dataset_dict.json"
    if not dataset_dict_file.exists():
        return None

    try:
        LOCAL_DATASET = load_from_disk(str(dataset_path))
        return LOCAL_DATASET
    except Exception as e:
        print(f"✗ Failed to load local demo dataset: {e}")
        return None


def extract_audio_sample(audio_sample):
    """Extract a waveform and sampling rate from a dataset audio sample."""
    if not audio_sample:
        return None, None

    audio_bytes = audio_sample.get("bytes")
    audio_path = audio_sample.get("path")

    if audio_bytes:
        return sf.read(BytesIO(audio_bytes))

    if audio_path and Path(audio_path).exists():
        return sf.read(audio_path)

    return None, None


def transcribe_audio_array(speech, sampling_rate=16000):
    """Transcribe an in-memory waveform using the loaded model."""
    if MODEL_LOAD_STATUS == "loading":
        return {"success": False, "error": "Model is still loading in background"}
    if MODEL is None or PROCESSOR is None:
        return {"success": False, "error": f"Model not loaded. Error: {MODEL_LOAD_ERROR}"}

    try:
        if speech is None:
            return {"success": False, "error": "No audio data provided"}

        speech = np.asarray(speech)
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)

        if sampling_rate != 16000:
            speech = librosa.resample(speech.astype(np.float32), orig_sr=sampling_rate, target_sr=16000)
            sampling_rate = 16000

        speech = speech.astype(np.float32)
        speech = speech / (np.max(np.abs(speech)) + 1e-6)

        input_values = PROCESSOR(
            speech,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_values.to(DEVICE)

        with torch.no_grad():
            logits = MODEL(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = PROCESSOR.batch_decode(predicted_ids)[0].strip().lower()

        if not transcription:
            return {"success": False, "error": "Model returned empty transcription"}

        # --- Post-processing pipeline ---
        original_transcription = transcription
        pp_steps = []
        if POST_PROCESSOR:
            try:
                pp_result = POST_PROCESSOR.process(transcription)
                transcription = pp_result.get("processed", transcription)
                pp_steps = pp_result.get("steps_applied", [])
            except Exception as pp_e:
                print(f"[WARN] Post-processing failed: {pp_e}")

        return {
            "success": True,
            "transcription": transcription,
            "original_transcription": original_transcription,
            "post_processing_steps": pp_steps,
            "word_count": len(transcription.split()),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def load_processor(model_id, token=None):
    """Load processor with fallback for tokenizer config issues."""
    try:
        if token:
            return Wav2Vec2Processor.from_pretrained(model_id, token=token)
        return Wav2Vec2Processor.from_pretrained(model_id)
    except Exception as e:
        message = str(e)
        if "extra_special_tokens" in message:
            print("  ⚠ Processor config error, trying manual tokenizer/feature_extractor...")
            if token:
                tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                    model_id,
                    token=token,
                    extra_special_tokens=[]
                )
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, token=token)
            else:
                tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                    model_id,
                    extra_special_tokens=[]
                )
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        raise

def get_device_info():
    """Collect available device info (CUDA/ROCm/MPS/CPU)."""
    devices = []

    # CUDA / ROCm
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            backend = "rocm" if torch.version.hip else "cuda"
            devices.append({
                "type": backend,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2)
            })
        except Exception:
            devices.append({"type": "cuda", "name": "CUDA GPU", "total_memory_gb": None})

    # Apple Silicon (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append({"type": "mps", "name": "Apple Silicon (MPS)", "total_memory_gb": None})

    # CPU fallback
    cpu_name = platform.processor() or "CPU"
    devices.append({"type": "cpu", "name": cpu_name, "total_memory_gb": None})

    return devices

def select_device(preference="auto"):
    """Select the best available device based on preference."""
    devices = get_device_info()
    device_types = [d["type"] for d in devices]

    if preference == "cuda" and "cuda" in device_types:
        return torch.device("cuda"), "CUDA GPU"
    if preference == "rocm" and "rocm" in device_types:
        return torch.device("cuda"), "ROCm GPU"
    if preference == "mps" and "mps" in device_types:
        return torch.device("mps"), "MPS"
    if preference == "cpu":
        return torch.device("cpu"), "CPU"

    # Auto: prefer CUDA/ROCm, then MPS, then CPU
    if "cuda" in device_types or "rocm" in device_types:
        return torch.device("cuda"), "CUDA/ROCm GPU"
    if "mps" in device_types:
        return torch.device("mps"), "MPS"
    return torch.device("cpu"), "CPU"

# Dialect mapping - Chuyển đổi từ vùng miền sang tiếng Việt chuẩn
DIALECT_MAPPING = {
    # Miền Nam
    "dạ": "vâng",
    "ui": "vâng",
    "trẹo": "rẽ",
    "nhắn": "nhấn",
    
    # Miền Bắc
    "vậy": "vậy",
    "thế": "vậy",
    
    # Common variations
    "chi": "gì",
    "zô": "vào",
    "dô": "vào"
}

def load_model():
    """Load ASR model from a Hugging Face repo or local model folder."""
    global MODEL, PROCESSOR, DEVICE, MODEL_SOURCE
    
    try:
        MODEL_SOURCE = None
        web_demo_root = Path(__file__).resolve().parent
        project_root = web_demo_root.parent
        local_candidates = [
            project_root / "final_model",  # Kaggle-trained model (WER ~13.3%)
            web_demo_root / "results" / "final_model",
            project_root / "results" / "final_model",
            project_root / "results" / "models" / "wav2vec2-vietnamese" / "final_model",
            project_root / "models" / "wav2vec2-vietnamese-asr" / "final_model",
        ]
        model_path = None
        configured_hf_source = get_configured_hf_source()
        token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        
        # Try the configured Hugging Face source first so the app can load a shared model.
        if configured_hf_source:
            try:
                print(f"\nAttempting Hugging Face source: {configured_hf_source}...")
                _ = load_processor(configured_hf_source, token=token)
                model_path = configured_hf_source
                MODEL_SOURCE = configured_hf_source
                print(f"  ✓ SUCCESS! Using Hugging Face source: {configured_hf_source}")
            except Exception as e:
                print(f"  ✗ Hugging Face source failed: {str(e)[:120]}")

        # Try local model folders next
        if not model_path:
            for candidate in local_candidates:
                if candidate.exists():
                    model_path = str(candidate)
                    MODEL_SOURCE = model_path
                    print(f"✓ Found local model: {model_path}")
                    break

        # Try multiple fallback models if neither the configured source nor the local model is available
        # FIX: Put 'base' model first for Render to avoid OOM (512MB RAM limit)
        fallback_models = [
            "nguyenvulebinh/wav2vec2-base-vi",             # Vietnamese - base (Lightweight ~300MB)
            "nguyenvulebinh/wav2vec2-large-vi-vlsp2020",  # Vietnamese - large (~1.2GB)
            "facebook/wav2vec2-base-960h"                  # English fallback
        ]

        if not model_path:
            print("\n⚠ WARNING: No configured Hugging Face source or local model found")
            print("  → Switching to HIGH-QUALITY pretrained fallback models...\n")
            
            # Try each fallback
            for fallback in fallback_models:
                try:
                    print(f"  Attempting: {fallback}...")
                    # Quick test if model exists (robust processor load)
                    _ = load_processor(fallback, token=token)
                    model_path = fallback
                    MODEL_SOURCE = fallback
                    print(f"  ✓ SUCCESS! Using pretrained model: {fallback}")
                    print(f"  → This model has MUCH better quality than undertrained local models\n")
                    break
                except Exception as e:
                    print(f"  ✗ Failed: {str(e)[:100]}")
                    continue
            else:
                # No model worked - return False
                print("  ✗ All fallback models failed")
                return False
        
        # Select device
        DEVICE, device_label = select_device(DEVICE_PREFERENCE)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            print(f"🚀 Using {device_label}")
        elif DEVICE.type == "mps":
            print("🚀 Using Apple Silicon (MPS)")
        else:
            print("🐌 CPU mode - continuing anyway")
        
        print(f"\nLoading model from: {model_path}...")
        
        PROCESSOR = load_processor(model_path, token=token)
        if token:
            MODEL = Wav2Vec2ForCTC.from_pretrained(model_path, token=token).to(DEVICE)
        else:
            MODEL = Wav2Vec2ForCTC.from_pretrained(model_path).to(DEVICE)
        MODEL.eval()
        
        print("✓ Model loaded successfully!\n")
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("  Continuing without model - web interface will still work\n")
        return False

def background_load_model():
    """Load model in background thread to avoid blocking server startup"""
    global MODEL_LOAD_STATUS, MODEL_LOAD_ERROR
    MODEL_LOAD_STATUS = "loading"
    print("[ASR Model] Starting background loading...")
    try:
        success = load_model()
        if success:
            MODEL_LOAD_STATUS = "ready"
            print("[ASR Model] Loaded successfully in background!")
        else:
            MODEL_LOAD_STATUS = "error"
            MODEL_LOAD_ERROR = "Failed to load model. Check server logs."
            print("[ASR Model] Loading failed (load_model returned False)")
    except Exception as e:
        MODEL_LOAD_STATUS = "error"
        MODEL_LOAD_ERROR = str(e)
        print(f"[ASR Model] Exception during background loading: {e}")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(audio_path):
    """Convert audio file to WAV format (16kHz mono) for Wav2Vec2 inference."""
    try:
        # Get file extension
        ext = audio_path.rsplit('.', 1)[1].lower() if '.' in audio_path else ''

        if not FFMPEG_AVAILABLE and ext != 'wav':
            raise EnvironmentError(
                "ffmpeg không tìm thấy. Chạy setup_ffmpeg.bat hoặc cài ffmpeg thủ công."
            )

        # If already WAV, return as is (librosa will handle resampling)
        if ext == 'wav':
            return audio_path

        # Convert using pydub → 16 kHz mono WAV
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)

        wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
        audio.export(wav_path, format='wav')

        return wav_path
    except EnvironmentError:
        raise  # re-raise ffmpeg-specific errors
    except FileNotFoundError:
        raise EnvironmentError(
            "ffmpeg không tìm thấy. Chạy setup_ffmpeg.bat hoặc cài ffmpeg thủ công."
        )
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None

def preprocess_audio(audio_path, target_sr=16000):
    """Preprocess audio file for inference"""
    try:
        # Convert to WAV if needed
        wav_path = convert_to_wav(audio_path)
        if wav_path is None:
            return None, None
        
        # Load audio
        speech, sr = librosa.load(wav_path, sr=target_sr)
        
        # Normalize
        speech = speech / (np.max(np.abs(speech)) + 1e-6)
        
        # Clean up converted file if different from original
        if wav_path != audio_path and os.path.exists(wav_path):
            os.unlink(wav_path)
        
        return speech, sr
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None, None

def transcribe_audio(audio_path):
    """Transcribe audio file using loaded model"""
    try:
        speech, sr = preprocess_audio(audio_path)
        if speech is None:
            return {"success": False, "error": "Failed to process audio"}
        return transcribe_audio_array(speech, sr)
    except Exception as e:
        return {"success": False, "error": str(e)}


def transcribe_chunk(chunk_data):
    """Transcribe a single audio chunk. Used by ThreadPoolExecutor."""
    idx, speech_chunk, sr, offset = chunk_data
    result = transcribe_audio_array(speech_chunk, sr)
    if result.get("success"):
        duration = len(speech_chunk) / sr
        result["chunk_index"] = idx
        result["start"] = round(offset, 2)
        result["end"] = round(offset + duration, 2)
    return result


def transcribe_audio_chunked(audio_path, enable_diarization=False,
                              min_speakers=2, max_speakers=4):
    """
    Transcribe long audio with smart 30s chunking + 1s overlap.
    Optionally run speaker diarization in parallel.

    Returns dict with full transcript, chunks with timestamps,
    and diarization results if enabled.
    """
    try:
        wav_path = convert_to_wav(audio_path)
        if wav_path is None:
            return {"success": False, "error": "Failed to convert audio"}

        speech, sr = librosa.load(wav_path, sr=16000)
        if speech is None:
            return {"success": False, "error": "Failed to load audio"}

        speech = speech.astype(np.float32)
        speech = speech / (np.max(np.abs(speech)) + 1e-6)
        duration = len(speech) / sr

        # Clean up converted file
        if wav_path != audio_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except Exception:
                pass

        # If short enough, transcribe directly
        if duration <= CHUNK_DURATION_S + 5:
            result = transcribe_audio_array(speech, sr)
            if result.get("success"):
                result["chunks"] = [{
                    "text": result["transcription"],
                    "start": 0.0,
                    "end": round(duration, 2),
                }]
                result["duration"] = round(duration, 2)
                result["chunk_count"] = 1
            return result

        # Split into chunks
        chunk_samples = int(CHUNK_DURATION_S * sr)
        overlap_samples = int(CHUNK_OVERLAP_S * sr)
        step = chunk_samples - overlap_samples

        chunk_tasks = []
        offset = 0
        idx = 0
        while offset < len(speech):
            end = min(offset + chunk_samples, len(speech))
            chunk = speech[offset:end]
            if len(chunk) < sr:  # Skip chunks < 1s
                break
            chunk_tasks.append((idx, chunk, sr, offset / sr))
            offset += step
            idx += 1

        total_chunks = len(chunk_tasks)
        logger.info(f"Chunking: {total_chunks} chunks, duration={duration:.1f}s")

        # Parallel transcription
        chunk_results = list(executor.map(transcribe_chunk, chunk_tasks))

        # Merge results
        transcript_parts = []
        transcript_chunks = []
        for cr in chunk_results:
            if cr.get("success") and cr.get("transcription"):
                transcript_parts.append(cr["transcription"])
                transcript_chunks.append({
                    "text": cr["transcription"],
                    "start": cr.get("start", 0),
                    "end": cr.get("end", 0),
                })

        full_transcript = " ".join(transcript_parts)

        # Post-processing on full transcript
        original_transcript = full_transcript
        pp_steps = []
        if POST_PROCESSOR and full_transcript:
            try:
                pp_result = POST_PROCESSOR.process(full_transcript)
                full_transcript = pp_result.get("processed", full_transcript)
                pp_steps = pp_result.get("steps_applied", [])
            except Exception as pp_e:
                logger.warning(f"Post-processing failed: {pp_e}")

        result = {
            "success": True,
            "transcription": full_transcript,
            "original_transcription": original_transcript,
            "post_processing_steps": pp_steps,
            "word_count": len(full_transcript.split()),
            "duration": round(duration, 2),
            "chunk_count": total_chunks,
            "chunks": transcript_chunks,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }

        # Speaker Diarization
        if enable_diarization and SPEAKER_DIARIZER and SPEAKER_DIARIZER.available:
            try:
                diar_path = wav_path if wav_path != audio_path and os.path.exists(wav_path) else audio_path
                # Need a WAV for diarization - re-save if needed
                temp_wav = os.path.join(UPLOAD_FOLDER, f"diar_{int(time.time())}.wav")
                sf.write(temp_wav, speech, sr)

                segments = SPEAKER_DIARIZER.diarize(
                    temp_wav, min_speakers=min_speakers, max_speakers=max_speakers
                )
                dialogue = SPEAKER_DIARIZER.assign_transcript_to_speakers(
                    segments, transcript_chunks
                )
                stats = SPEAKER_DIARIZER.get_speaker_stats(dialogue)

                result["dialogue"] = dialogue
                result["speaker_stats"] = stats
                result["speaker_count"] = len(stats)
                result["diarization_segments"] = segments

                # Cleanup temp wav
                if os.path.exists(temp_wav):
                    os.unlink(temp_wav)

                logger.info(f"Diarization: {len(stats)} speakers detected")
            except Exception as diar_e:
                logger.warning(f"Diarization failed: {diar_e}")
                result["diarization_error"] = str(diar_e)

        return result

    except Exception as e:
        logger.error(f"Chunked transcription error: {e}")
        return {"success": False, "error": str(e)}


def normalize_dialect(text):
    """Chuyển đổi tiếng địa phương sang tiếng Việt chuẩn"""
    words = text.split()
    normalized = []
    
    for word in words:
        # Check dialect mapping
        normalized_word = DIALECT_MAPPING.get(word.lower(), word)
        normalized.append(normalized_word)
    
    return " ".join(normalized)

def phonetic_transcribe(text):
    """Phiên âm tiếng Việt sang IPA (đơn giản)"""
    # Simplified Vietnamese to IPA mapping
    phonetic_map = {
        "a": "a", "ă": "ă", "â": "ə", "e": "ɛ", "ê": "e",
        "i": "i", "o": "ɔ", "ô": "o", "ơ": "ɤ", "u": "u",
        "ư": "ɯ", "y": "i",
        "b": "ɓ", "c": "k", "d": "z", "đ": "d", "g": "ɣ",
        "h": "h", "k": "k", "l": "l", "m": "m", "n": "n",
        "ng": "ŋ", "nh": "ɲ", "p": "p", "ph": "f", "qu": "kw",
        "r": "z", "s": "s", "t": "t", "th": "tʰ", "tr": "ʈ",
        "v": "v", "x": "s", "ch": "c", "gh": "ɣ", "gi": "z",
        "kh": "x", "ngh": "ŋ"
    }
    
    result = []
    i = 0
    text_lower = text.lower()
    
    while i < len(text_lower):
        # Try 2-char combinations first
        if i < len(text_lower) - 1:
            two_char = text_lower[i:i+2]
            if two_char in phonetic_map:
                result.append(phonetic_map[two_char])
                i += 2
                continue
        
        # Single char
        char = text_lower[i]
        result.append(phonetic_map.get(char, char))
        i += 1
    
    return "".join(result)

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    """API endpoint for audio transcription"""
    try:
        # Check if audio file exists
        if 'audio' not in request.files:
            return jsonify({
                "success": False,
                "error": "No audio file provided"
            }), 400
        
        audio_file = request.files['audio']
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        
        # Transcribe
        result = transcribe_audio(tmp_path)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/chatbot', methods=['POST'])
def api_chatbot():
    """Chatbot API - Hỗ trợ phiên âm và chuẩn hóa"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        action = data.get('action', 'general')
        
        if not user_message:
            return jsonify({
                "success": False,
                "error": "Empty message"
            }), 400
        
        response = ""
        
        # Phiên âm
        if action == 'phonetic' or 'phiên âm' in user_message.lower():
            text_to_transcribe = data.get('text', user_message)
            phonetic = phonetic_transcribe(text_to_transcribe)
            response = f"Phiên âm: /{phonetic}/"
        
        # Chuẩn hóa vùng miền
        elif action == 'normalize' or 'chuẩn hóa' in user_message.lower():
            text_to_normalize = data.get('text', user_message)
            normalized = normalize_dialect(text_to_normalize)
            response = f"Tiếng Việt chuẩn: {normalized}"
        
        # Đếm từ
        elif action == 'count' or 'đếm' in user_message.lower():
            text = data.get('text', user_message)
            word_count = len(text.split())
            char_count = len(text)
            response = f"Số từ: {word_count} | Số ký tự: {char_count}"
        
        # General help
        else:
            response = """Tôi có thể giúp bạn:
            
1. Phiên âm tiếng Việt sang IPA
   Ví dụ: "phiên âm xin chào"
   
2. Chuẩn hóa từ vùng miền
   Ví dụ: "chuẩn hóa dạ ui chi"
   
3. Đếm từ và ký tự
   Ví dụ: "đếm xin chào các bạn"
   
Bạn cần tôi giúp gì?"""
        
        return jsonify({
            "success": True,
            "response": response,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ===================== REALTIME WEBSOCKET (SOCKET.IO) =====================
SESSION_BUFFERS = {}

# --- HẰNG SỐ CẤU HÌNH VAD VÀ ỔN ĐỊNH NHẬN DẠNG ---
# Tùy thuộc vào độ nhạy của micro và độ ồn của phòng, bạn có thể cần điều chỉnh các thông số này.
VAD_RMS_THRESHOLD = 0.003       # Ngưỡng năng lượng âm thanh tối thiểu để nhận biết người nói (thấp hơn = nhạy hơn)
VAD_SILENCE_SECONDS = 1.0       # Thời gian im lặng tối thiểu (giây) trước khi chốt câu tự động
STABILITY_CYCLES = 3            # Số chu kỳ dịch liên tiếp có kết quả chuẩn hóa tương đương để coi là ổn định
FORCE_COMMIT_SECONDS = 6.0      # Thời gian tối đa (giây) nói liên tục mà chưa chốt câu để ép chốt câu tránh trễ
MIN_COMMIT_WORDS = 2            # Số lượng từ tối thiểu để chốt câu (tránh tiếng ồn hay từ rác siêu ngắn)
INFERENCE_INTERVAL_S = 1.2      # Khoảng cách tối thiểu (giây) giữa các lần chạy mô hình Wav2Vec2 (tiết kiệm CPU)

def clean_text_for_commit(text):
    if not text:
        return ""
    # Keep letters, numbers, spaces (unicode-aware)
    cleaned = re.sub(r'[^\w\s]', '', text)
    # Remove multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def normalize_text_for_stability(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def is_duplicate_or_garbage(text, last_text):
    text = text.strip().lower()
    last_text = last_text.strip().lower()
    
    # 1. Short valid Vietnamese responses exception
    valid_short_responses = {'có', 'không', 'đúng', 'rồi', 'vâng', 'chưa', 'được'}
    if text in valid_short_responses:
        return False
        
    # 2. Garbage words filter (unstable short fragments)
    garbage_words = {'tan', 'an', 'ta', 'tâ', 'tân', 'a', 'ă', 'â', 'e', 'ê', 'i', 'o', 'ô', 'ơ', 'u', 'ư', 'y'}
    words = text.split()
    
    if all(w in garbage_words for w in words):
        logger.info(f"[ASR-Filter] Filtered garbage segment: '{text}'")
        return True
        
    # 3. Too short filter: ignore phrases with less than MIN_COMMIT_WORDS
    if len(words) < MIN_COMMIT_WORDS:
        logger.info(f"[ASR-Filter] Filtered too short segment: '{text}'")
        return True
        
    # 4. Exact duplicate check
    if text == last_text:
        logger.info(f"[ASR-Filter] Filtered exact duplicate: '{text}'")
        return True
        
    # 5. Overlap duplicate check
    if text in last_text or last_text in text:
        words_last = last_text.split()
        intersection = set(words).intersection(set(words_last))
        if len(intersection) / max(len(words), len(words_last)) > 0.7:
            logger.info(f"[ASR-Filter] Filtered high overlap duplicate: '{text}' vs '{last_text}'")
            return True
            
    return False

@socketio.on('connect')
def handle_connect():
    SESSION_BUFFERS[request.sid] = {
        'audio': np.array([], dtype=np.float32),
        'confirmed_text': '',
        'last_committed_text': '',
        'interim_text': '',
        'silence_duration': 0.0,
        'continuous_speech_duration': 0.0,
        'last_predictions': [],
        'last_inference_time': 0.0,
        'chunk_id': 0
    }
    emit('server_ready', {'message': 'Connected to ASR Server'})

@socketio.on('disconnect')
def handle_disconnect():
    SESSION_BUFFERS.pop(request.sid, None)

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    if not isinstance(data, bytes):
        return

    pcm_data = np.frombuffer(data, dtype=np.float32)

    # Guard: ignore empty or corrupted chunks
    if pcm_data.size == 0:
        return

    session = SESSION_BUFFERS.get(request.sid)
    if not session:
        return

    # 1. Simple VAD-lite (RMS energy calculation)
    rms = np.sqrt(np.mean(pcm_data**2))
    
    # Calculate chunk duration (PCM float32, 16000Hz, mono)
    chunk_duration = pcm_data.size / 16000
    
    is_silent = rms < VAD_RMS_THRESHOLD
    
    if is_silent:
        session['silence_duration'] = session.get('silence_duration', 0.0) + chunk_duration
    else:
        session['silence_duration'] = 0.0
        session['continuous_speech_duration'] = session.get('continuous_speech_duration', 0.0) + chunk_duration

    # 2. Append new data to buffer
    session['audio'] = np.concatenate([session['audio'], pcm_data])

    # Keep max 5 seconds of audio for sliding window
    MAX_BUFFER = 16000 * 5
    if len(session['audio']) > MAX_BUFFER:
        session['audio'] = session['audio'][-MAX_BUFFER:]

    # 3. Optimize CPU: Only run Wav2Vec2 inference every INFERENCE_INTERVAL_S seconds
    now = time.time()
    last_infer = session.get('last_inference_time', 0.0)
    
    if now - last_infer < INFERENCE_INTERVAL_S:
        # Emit current status to frontend so it doesn't feel stuck
        status_msg = "Đang nghe..."
        if session.get('silence_duration', 0.0) > 0.5:
            status_msg = "Đang nghe... (im lặng)"
        else:
            status_msg = "Đang nghe... (phát hiện âm thanh)"
            
        emit('transcript_update', {
            'text': session.get('interim_text', ''),
            'full_text': session['confirmed_text'].strip(),
            'chunk_id': session['chunk_id'],
            'is_final': False,
            'status_message': status_msg,
            'latency': 0
        })
        return

    session['last_inference_time'] = now
    started_at = time.perf_counter()

    # 4. Transcribe the buffer
    result = transcribe_audio_array(session['audio'], 16000)
    new_transcript = result.get('transcription', '').strip()
    
    # 5. Stability Check using normalized text
    norm_new = normalize_text_for_stability(new_transcript)
    last_predictions = session.get('last_predictions', [])
    last_predictions.append(norm_new)
    if len(last_predictions) > STABILITY_CYCLES + 1:
        last_predictions.pop(0)
    session['last_predictions'] = last_predictions

    is_stable = False
    if len(last_predictions) >= STABILITY_CYCLES and norm_new:
        recent = last_predictions[-STABILITY_CYCLES:]
        if len(set(recent)) == 1:
            is_stable = True

    interim_text = new_transcript
    session['interim_text'] = interim_text
    
    # 6. Silence / Stability / Force Commit decision
    should_commit = False
    commit_reason = ""
    
    if session['silence_duration'] >= VAD_SILENCE_SECONDS and interim_text:
        should_commit = True
        commit_reason = f"Đã chốt 1 đoạn (Khoảng lặng {session['silence_duration']:.1f}s)"
    elif session.get('continuous_speech_duration', 0.0) >= FORCE_COMMIT_SECONDS and interim_text:
        should_commit = True
        commit_reason = f"Đã force commit sau {FORCE_COMMIT_SECONDS} giây"
    elif is_stable and interim_text:
        last_committed = session.get('last_committed_text', '')
        if normalize_text_for_stability(interim_text) != normalize_text_for_stability(last_committed):
            should_commit = True
            commit_reason = f"Đã chốt 1 đoạn (Kết quả ổn định)"

    if should_commit:
        cleaned_text = clean_text_for_commit(interim_text)
        last_committed = session.get('last_committed_text', '')
        
        if cleaned_text and not is_duplicate_or_garbage(cleaned_text, last_committed):
            session['confirmed_text'] += (' ' + cleaned_text) if session['confirmed_text'] else cleaned_text
            session['last_committed_text'] = cleaned_text
            session['chunk_id'] += 1
            
            latency = round((time.perf_counter() - started_at) * 1000)
            
            logger.info(f"[ASR-Commit] Reason: {commit_reason}. Committed final segment: '{cleaned_text}'")
            emit('transcript_update', {
                'text': cleaned_text,
                'full_text': session['confirmed_text'].strip(),
                'chunk_id': session['chunk_id'],
                'is_final': True,
                'status_message': commit_reason,
                'latency': latency
            })
            
        # Reset rolling buffer and stability states
        session['audio'] = np.array([], dtype=np.float32)
        session['silence_duration'] = 0.0
        session['continuous_speech_duration'] = 0.0
        session['last_predictions'] = []
        session['interim_text'] = ""
        
    else:
        # Emit as interim result
        latency = round((time.perf_counter() - started_at) * 1000)
        cleaned_interim = clean_text_for_commit(interim_text)
        
        # Build status message
        status_msg = "Đang xử lý Wav2Vec2..."
        if session.get('silence_duration', 0.0) > 0.0:
            status_msg = f"Đang nghe... (Khoảng lặng: {session['silence_duration']:.1f}s)"
        else:
            status_msg = f"Đang nghe... (Phát hiện âm thanh {session.get('continuous_speech_duration', 0.0):.1f}s)"
            
        emit('transcript_update', {
            'text': cleaned_interim,
            'full_text': session['confirmed_text'].strip(),
            'chunk_id': session['chunk_id'],
            'is_final': False,
            'status_message': status_msg,
            'latency': latency
        })

@socketio.on('stop_recording')
def handle_stop_recording():
    session = SESSION_BUFFERS.get(request.sid)
    if not session:
        return
        
    logger.info("[ASR-State] Stop Recording event received from client.")
    
    # Force transcribe and commit any remaining audio buffer
    if len(session['audio']) > 0:
        result = transcribe_audio_array(session['audio'], 16000)
        final_text = result.get('transcription', '').strip()
        cleaned = clean_text_for_commit(final_text)
        last_committed = session.get('last_committed_text', '')
        
        if cleaned and not is_duplicate_or_garbage(cleaned, last_committed):
            session['confirmed_text'] += (' ' + cleaned) if session['confirmed_text'] else cleaned
            session['last_committed_text'] = cleaned
            session['chunk_id'] += 1
            
            logger.info(f"[ASR-Commit-Stop] Committed remaining segment on stop: '{cleaned}'")
            emit('transcript_update', {
                'text': cleaned,
                'full_text': session['confirmed_text'].strip(),
                'chunk_id': session['chunk_id'],
                'is_final': True,
                'status_message': "Đã dừng và chốt đoạn cuối",
                'latency': 0
            })
            
    # Reset session buffers
    session['audio'] = np.array([], dtype=np.float32)
    session['silence_duration'] = 0.0
    session['continuous_speech_duration'] = 0.0
    session['last_predictions'] = []
    session['interim_text'] = ""

@socketio.on('correct_text')
def handle_correct_text(data):
    """
    FIX (NEW): SocketIO event handler for Web Speech API final text segments.
    Applies post-processing (dictionary correction + optional Gemini) and
    emits corrected text back to the same client session.
    """
    text = data.get('text', '').strip() if isinstance(data, dict) else ''
    if not text:
        return

    corrected = text
    steps = []
    if POST_PROCESSOR:
        try:
            pp_result = POST_PROCESSOR.process(text)
            corrected = pp_result.get('processed', text)
            steps = pp_result.get('steps_applied', [])
        except Exception as pp_e:
            logger.warning(f'[RT] Post-processing failed: {pp_e}')

    emit('text_corrected', {
        'original': text,
        'corrected': corrected,
        'steps_applied': steps,
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    """Check model status"""
    device_info = get_device_info()
    return jsonify({
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else "N/A",
        "status": MODEL_LOAD_STATUS,
        "model_source": MODEL_SOURCE,
        "available_devices": device_info,
        "error": MODEL_LOAD_ERROR
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check alias for Render and other deployment platforms"""
    return api_status()


# ===================== UPGRADE 2025: NEW ENDPOINTS =====================

@app.route('/api/health', methods=['GET'])
def api_health():
    """
    Health check endpoint dành cho launcher và monitoring.
    Trả về trạng thái chi tiết hơn /health để launcher biết server đã sẵn sàng.
    """
    model_ok = MODEL is not None
    return jsonify({
        "status": MODEL_LOAD_STATUS,
        "model_loaded": model_ok,
        "model_source": MODEL_SOURCE,
        "device": str(DEVICE) if DEVICE else "cpu",
        "server_version": SERVER_VERSION,
        "use_gemini": USE_GEMINI,
        "default_live_mode": DEFAULT_LIVE_MODE,
        "ffmpeg_available": FFMPEG_AVAILABLE,
        "diarization_available": SPEAKER_DIARIZER is not None,
        "post_processing_available": POST_PROCESSOR is not None,
    })


@app.route('/api/gemini-status', methods=['GET'])
def api_gemini_status():
    """
    Trả về trạng thái Gemini cho status card trên giao diện.
    Không thực sự gọi Gemini — chỉ kiểm tra cấu hình.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    has_key = bool(api_key) and api_key not in (
        "your_gemini_api_key_here",
        "PASTE_MY_NEW_GEMINI_API_KEY_HERE",
        "",
    )
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

    if not GEMINI_CLIENT_AVAILABLE:
        return jsonify({
            "status": "unavailable",
            "enabled": False,
            "has_key": False,
            "model": model_name,
            "error": "google-genai library not installed"
        })

    if not has_key:
        return jsonify({
            "status": "no_key",
            "enabled": False,
            "has_key": False,
            "model": model_name,
            "error": "GEMINI_API_KEY chưa được cấu hình trong .env"
        })

    if not USE_GEMINI:
        return jsonify({
            "status": "disabled",
            "enabled": False,
            "has_key": True,
            "model": model_name,
            "error": "USE_GEMINI=false trong .env"
        })

    return jsonify({
        "status": "ready",
        "enabled": True,
        "has_key": True,
        "model": model_name,
        "error": None
    })


@app.route('/api/config', methods=['GET'])
def api_config():
    """
    Trả về cấu hình hiện tại của server để frontend biết chế độ hoạt động.
    KHÔNG trả về API key hay thông tin nhạy cảm.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    has_gemini_key = bool(api_key) and api_key not in (
        "your_gemini_api_key_here",
        "PASTE_MY_NEW_GEMINI_API_KEY_HERE",
        "",
    )
    return jsonify({
        "server_version": SERVER_VERSION,
        "model_source": MODEL_SOURCE,
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else "cpu",
        "default_live_mode": DEFAULT_LIVE_MODE,
        "use_gemini": USE_GEMINI,
        "gemini_configured": has_gemini_key,
        "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        "ffmpeg_available": FFMPEG_AVAILABLE,
        "diarization_available": SPEAKER_DIARIZER is not None,
        "post_processing_available": POST_PROCESSOR is not None,
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "allowed_extensions": sorted(list(ALLOWED_EXTENSIONS)),
        "chunk_duration_s": CHUNK_DURATION_S,
    })


@app.route('/api/export-transcript', methods=['POST'])
def api_export_transcript():
    """
    Xuất transcript và/hoặc summary thành file text.
    Input JSON: { transcript, summary, filename, format }
    Output: file text/plain để download.
    """
    try:
        data = request.get_json() or {}
        transcript = data.get('transcript', '').strip()
        summary = data.get('summary', '').strip()
        filename = data.get('filename', 'vietasr_export') or 'vietasr_export'
        export_format = data.get('format', 'transcript')  # 'transcript' | 'summary' | 'full'

        if not transcript and not summary:
            return jsonify({"success": False, "error": "Không có nội dung để xuất"}), 400

        # Tạo nội dung file
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []

        if export_format == 'summary' and summary:
            lines.append(f"=== VietASR Pro — Tóm Tắt ===")
            lines.append(f"Thời gian: {now_str}")
            lines.append("")
            lines.append(summary)
        elif export_format == 'full':
            lines.append(f"=== VietASR Pro — Kết Quả Nhận Dạng ===")
            lines.append(f"Thời gian: {now_str}")
            lines.append("")
            if transcript:
                lines.append("--- TRANSCRIPT ---")
                lines.append(transcript)
                lines.append("")
            if summary:
                lines.append("--- TÓM TẮT ---")
                lines.append(summary)
        else:
            # Mặc định: chỉ transcript
            lines.append(f"=== VietASR Pro — Transcript ===")
            lines.append(f"Thời gian: {now_str}")
            lines.append("")
            lines.append(transcript or summary)

        content = "\n".join(lines)
        safe_filename = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        from io import BytesIO
        buf = BytesIO(content.encode('utf-8'))
        buf.seek(0)

        return send_file(
            buf,
            mimetype='text/plain; charset=utf-8',
            as_attachment=True,
            download_name=safe_filename,
        )
    except Exception as e:
        logger.error(f"[Export] Lỗi xuất file: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/device-info', methods=['GET'])
def api_device_info():
    """Get detailed device information"""
    devices = get_device_info()
    memory_info = {}
    if DEVICE and DEVICE.type == "cuda":
        memory_info = {
            "allocated_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / (1024 ** 3), 2)
        }
    return jsonify({
        "current_device": str(DEVICE) if DEVICE else "cpu",
        "device_preference": DEVICE_PREFERENCE,
        "devices": devices,
        "memory": memory_info,
        "model_loaded": MODEL is not None,
        "model_source": MODEL_SOURCE,
        "local_dataset_path": str(get_local_dataset_path()),
        "local_dataset_loaded": LOCAL_DATASET is not None or get_local_dataset_path().joinpath("dataset_dict.json").exists()
    })

@app.route('/api/device-select', methods=['POST'])
def api_device_select():
    """Select device preference and reload model"""
    global MODEL, PROCESSOR, DEVICE, DEVICE_PREFERENCE, MODEL_SOURCE
    try:
        data = request.get_json() or {}
        preference = data.get("device", "auto")
        DEVICE_PREFERENCE = preference

        # Reset model and reload
        MODEL = None
        PROCESSOR = None
        DEVICE = None
        MODEL_SOURCE = None
        loaded = load_model()

        return jsonify({
            "success": loaded,
            "device_preference": DEVICE_PREFERENCE,
            "device": str(DEVICE) if DEVICE else "cpu"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/normalize', methods=['POST'])
def api_normalize():
    """Normalize dialect text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        normalized = normalize_dialect(text)
        
        return jsonify({
            "success": True,
            "original": text,
            "normalized": normalized
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Upload and transcribe audio file with chunking + diarization"""
    if MODEL_LOAD_STATUS == "loading":
        return jsonify({
            "success": False,
            "error": "Mô hình Wav2Vec2 đang được tải ở chế độ nền. Vui lòng thử lại sau.",
            "error_code": "MODEL_LOADING"
        }), 503
    elif MODEL is None or PROCESSOR is None:
        return jsonify({
            "success": False,
            "error": f"Mô hình Wav2Vec2 chưa được tải thành công. Lỗi: {MODEL_LOAD_ERROR}",
            "error_code": "MODEL_NOT_LOADED"
        }), 503

    filepath = None
    try:
        # Rate limiting
        client_ip = request.remote_addr or '127.0.0.1'
        if not check_rate_limit(client_ip):
            return jsonify({
                "success": False,
                "error": "Vượt quá giới hạn. Thử lại sau 1 phút.",
                "error_code": "RATE_LIMIT"
            }), 429

        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided", "error_code": "NO_FILE"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected", "error_code": "NO_FILE_SELECTED"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"Dinh dang khong ho tro. Cho phep: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                "error_code": "UNSUPPORTED_FORMAT"
            }), 400

        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext != 'wav' and not FFMPEG_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "ffmpeg khong tim thay. Chay setup_ffmpeg.bat de cai dat.",
                "error_code": "FFMPEG_NOT_FOUND"
            }), 500

        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Check file size
        file_size = os.path.getsize(filepath)
        if file_size > MAX_FILE_SIZE:
            os.unlink(filepath)
            return jsonify({
                "success": False,
                "error": f"File qua lon ({file_size // (1024*1024)}MB). Gioi han: {MAX_FILE_SIZE // (1024*1024)}MB",
                "error_code": "FILE_TOO_LARGE"
            }), 400

        ground_truth = (
            request.form.get('ground_truth', '')
            or request.form.get('reference_text', '')
            or ''
        ).strip()

        # Diarization params
        enable_diarization = request.form.get('enable_diarization', 'false').lower() in ('true', '1', 'on')
        min_speakers = int(request.form.get('min_speakers', 2))
        max_speakers = int(request.form.get('max_speakers', 4))

        started_at = time.perf_counter()

        # Use chunked transcription for all files
        result = transcribe_audio_chunked(
            filepath,
            enable_diarization=enable_diarization,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        processing_time_seconds = round(time.perf_counter() - started_at, 2)
        result['processing_time_seconds'] = processing_time_seconds
        if processing_time_seconds > 2.0:
            logger.warning(f"[PERF] Slow inference: {processing_time_seconds:.2f}s exceeds 2s target")

        if result.get('success'):
            full_transcript = result.get('transcription', '')

            if ground_truth:
                result['ground_truth'] = ground_truth
                if compute_wer:
                    try:
                        result['wer'] = round(float(compute_wer(ground_truth, full_transcript)), 4)
                    except Exception:
                        result['wer'] = None
                else:
                    result['wer'] = None
            else:
                result['wer'] = None

            result['file_info'] = {
                'filename': file.filename,
                'size': file_size,
                'format': file_ext.upper()
            }

            # Gọi Gemini để tóm tắt SAU KHI transcript hoàn chỉnh.
            # Nếu Gemini thất bại, vẫn trả về transcript cho frontend.
            # KHÔNG gọi Gemini cho từng audio chunk rải rộc.
            auto_summarize = request.form.get('auto_summarize', 'false').lower() in ('true', '1', 'on')
            result['summary'] = None
            result['summary_error'] = None
            result['model'] = None

            if auto_summarize and GEMINI_CLIENT_AVAILABLE and full_transcript:
                gemini_key = os.environ.get('GEMINI_API_KEY', '').strip()
                if gemini_key:
                    try:
                        logger.info(f"[Upload] Bắt đầu tóm tắt tự động, transcript={len(full_transcript)} ký tự")
                        gem_result = _gemini_summarize(
                            transcript=full_transcript,
                            mode='summary',
                            api_key=gemini_key,
                        )
                        result['summary'] = gem_result.get('summary')
                        result['model'] = gem_result.get('model')
                        logger.info("[Upload] Tóm tắt tự động thành công")
                    except PermissionError as _p:
                        result['summary_error'] = 'forbidden_403'
                        logger.warning(f"[Upload] Gemini 403: {_p}")
                    except RuntimeError as _r:
                        result['summary_error'] = 'rate_limit_429'
                        logger.warning(f"[Upload] Gemini 429 exhausted: {_r}")
                    except Exception as _e:
                        result['summary_error'] = str(_e)[:120]
                        logger.warning(f"[Upload] Gemini thất bại: {_e}")
                else:
                    result['summary_error'] = 'no_api_key'

        logger.info(f"Upload: {file.filename} ({file_size}B) -> {result.get('word_count', 0)} words in {result.get('processing_time_seconds', 0)}s")
        return jsonify(result)

    except EnvironmentError as env_err:
        logger.error(f"Upload env error: {env_err}")
        return jsonify({"success": False, "error": str(env_err), "error_code": "FFMPEG_NOT_FOUND"}), 500
    except Exception as e:
        logger.error(f"Upload error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Loi xu ly file: {str(e)}", "error_code": "PROCESSING_ERROR"}), 500
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.unlink(filepath)
            except Exception:
                pass


@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    """
    Tóm tắt transcript bằng Gemini API.
    Hỗ trợ transcript dài bằng cách tách chunks và tổng hợp.
    Chỉ gọi Gemini SAU KHI transcript hoàn chỉnh.
    """
    # Lấy dữ liệu trước try để except blocks có thể truy cập
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    mode = data.get('mode', 'summary')

    try:
        if not text:
            return jsonify({
                "success": False,
                "error": "Không có văn bản để tóm tắt",
                "transcript": "",
                "summary": None,
                "chunks_count": 0,
                "model": None,
                "summary_error": "empty_text"
            }), 400

        # Kiểm tra gemini_client có sẵn không
        if not GEMINI_CLIENT_AVAILABLE or _gemini_summarize is None:
            return jsonify({
                "success": False,
                "transcript": text,
                "summary": None,
                "chunks_count": 0,
                "model": None,
                "summary_error": "gemini_client_unavailable",
                "error": "gemini_client chưa được cài đặt. pip install google-genai"
            }), 503

        # Đọc API key từ biến môi trường — KHÔNG lấy từ frontend
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "PASTE_MY_NEW_GEMINI_API_KEY_HERE":
            return jsonify({
                "success": False,
                "transcript": text,
                "summary": None,
                "chunks_count": 0,
                "model": None,
                "summary_error": "no_api_key",
                "error": "GEMINI_API_KEY chưa được cấu hình trong file .env"
            }), 503

        logger.info(f"[Summarize] Gọi Gemini, mode={mode}, text={len(text)} ký tự")

        # Gọi Gemini — hỗ trợ chunking cho transcript dài
        gem_result = _gemini_summarize(
            transcript=text,
            mode=mode,
            api_key=api_key,
        )

        summary = gem_result.get('summary', '')
        chunks_count = gem_result.get('chunks_count', 1)
        model_used = gem_result.get('model', '')

        logger.info(
            f"[Summarize] Thành công: chunks={chunks_count}, "
            f"summary={len(summary)} ký tự, model={model_used}"
        )
        return jsonify({
            "success": True,
            "transcript": text,
            "summary": summary,
            "chunks_count": chunks_count,
            "model": model_used,
            "summary_error": None,
            "mode": mode,
        })

    except PermissionError as perm_err:
        # Lỗi 403: API key không hợp lệ
        logger.error(f"[Summarize] 403 Forbidden: {perm_err}")
        return jsonify({
            "success": False,
            "transcript": text,
            "summary": None,
            "chunks_count": 0,
            "model": None,
            "summary_error": "forbidden_403",
            "error": str(perm_err)
        }), 403

    except RuntimeError as rate_err:
        # Lỗi 429 sau khi hết số lần retry
        logger.error(f"[Summarize] Rate limit exhausted: {rate_err}")
        return jsonify({
            "success": False,
            "transcript": text,
            "summary": None,
            "chunks_count": 0,
            "model": None,
            "summary_error": "rate_limit_429",
            "error": "Gemini API đang quá tải (429). Vui lòng thử lại sau vài phút."
        }), 429

    except ValueError as val_err:
        # Thiếu API key
        logger.error(f"[Summarize] Thiếu API key: {val_err}")
        return jsonify({
            "success": False,
            "transcript": text,
            "summary": None,
            "chunks_count": 0,
            "model": None,
            "summary_error": "no_api_key",
            "error": str(val_err)
        }), 503

    except TimeoutError as timeout_err:
        # Timeout
        logger.error(f"[Summarize] Timeout: {timeout_err}")
        return jsonify({
            "success": False,
            "transcript": text,
            "summary": None,
            "chunks_count": 0,
            "model": None,
            "summary_error": "timeout",
            "error": "Gemini API phản hồi quá chậm. Vui lòng thử lại."
        }), 504

    except Exception as e:
        logger.error(f"[Summarize] Lỗi không xác định: {e}")
        return jsonify({
            "success": False,
            "transcript": text,
            "summary": None,
            "chunks_count": 0,
            "model": None,
            "summary_error": "unknown_error",
            "error": str(e)
        }), 500


@app.route('/api/gemini-test', methods=['GET'])
def api_gemini_test():
    """
    Kiểm tra kết nối Gemini API bằng cách gửi prompt ngắn.
    Trả về: success, model, response, error
    API key được đọc từ biến môi trường — KHÔNG lộ ra response.
    """
    if not GEMINI_CLIENT_AVAILABLE or _gemini_test is None:
        return jsonify({
            "success": False,
            "model": None,
            "response": None,
            "error": "gemini_client chưa được cài đặt. pip install google-genai"
        }), 503

    # Đọc API key từ env — KHÔNG nhận từ query param hay body (bảo mật)
    result = _gemini_test()
    status_code = 200 if result.get('success') else 503
    return jsonify(result), status_code


@app.route('/api/demo/local-sample', methods=['GET'])
def api_demo_local_sample():
    """Transcribe a random or indexed sample from the local Data/train dataset."""
    dataset = load_local_demo_dataset()
    if dataset is None:
        return jsonify({
            "success": False,
            "error": f"Local dataset not found at {get_local_dataset_path()}"
        }), 404

    split_name = request.args.get('split', 'train')
    index_param = request.args.get('index', '').strip()

    if split_name not in dataset:
        return jsonify({
            "success": False,
            "error": f"Split '{split_name}' not found in local dataset"
        }), 400

    split_dataset = dataset[split_name]
    if len(split_dataset) == 0:
        return jsonify({
            "success": False,
            "error": f"Split '{split_name}' is empty"
        }), 400

    if index_param:
        try:
            sample_index = max(0, min(int(index_param), len(split_dataset) - 1))
        except ValueError:
            return jsonify({
                "success": False,
                "error": "index must be an integer"
            }), 400
    else:
        sample_index = random.randint(0, len(split_dataset) - 1)

    sample = split_dataset[sample_index]
    audio_sample = sample.get('audio') or {}
    transcript = sample.get('transcription') or sample.get('text') or ''
    speech, sr = extract_audio_sample(audio_sample)

    if speech is None or sr is None:
        return jsonify({
            "success": False,
            "error": "Could not read audio from the selected sample"
        }), 400

    result = transcribe_audio_array(speech, sr)
    result.update({
        "split": split_name,
        "index": sample_index,
        "ground_truth": transcript,
        "audio_path": audio_sample.get('path'),
        "audio_has_bytes": audio_sample.get('bytes') is not None,
    })
    return jsonify(result)

@app.route('/api/phonetic', methods=['POST'])
def api_phonetic():
    """Get phonetic transcription"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        phonetic = phonetic_transcribe(text)
        
        return jsonify({
            "success": True,
            "text": text,
            "phonetic": phonetic
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ===================== POST-PROCESSING API =====================

@app.route('/api/post-process/config', methods=['GET'])
def api_pp_config_get():
    """Get current post-processing configuration."""
    if not POST_PROCESSOR:
        return jsonify({"success": False, "error": "Post-processor not available"}), 503
    return jsonify({
        "success": True,
        "config": POST_PROCESSOR.config
    })


@app.route('/api/post-process/config', methods=['POST'])
def api_pp_config_set():
    """Update post-processing configuration."""
    if not POST_PROCESSOR:
        return jsonify({"success": False, "error": "Post-processor not available"}), 503
    try:
        data = request.get_json() or {}
        pipeline = data.get('pipeline')
        llm = data.get('llm')

        if pipeline:
            POST_PROCESSOR.config.setdefault('pipeline', {})
            POST_PROCESSOR.config['pipeline'].update(pipeline)
        if llm:
            POST_PROCESSOR.config.setdefault('llm', {})
            POST_PROCESSOR.config['llm'].update(llm)

        POST_PROCESSOR.save_config()
        POST_PROCESSOR.reload()

        return jsonify({"success": True, "config": POST_PROCESSOR.config})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/post-process/dictionary', methods=['GET'])
def api_pp_dict_get():
    """Get current correction dictionary."""
    if not POST_PROCESSOR:
        return jsonify({"success": False, "error": "Post-processor not available"}), 503
    raw = POST_PROCESSOR._load_json(POST_PROCESSOR.dict_path, {})
    return jsonify({"success": True, "dictionary": raw})


@app.route('/api/post-process/dictionary', methods=['POST'])
def api_pp_dict_update():
    """Add or update entries in the correction dictionary."""
    if not POST_PROCESSOR:
        return jsonify({"success": False, "error": "Post-processor not available"}), 503
    try:
        data = request.get_json() or {}
        entries = data.get('entries', {})
        raw = POST_PROCESSOR._load_json(POST_PROCESSOR.dict_path, {})
        raw.update(entries)
        with open(POST_PROCESSOR.dict_path, 'w', encoding='utf-8') as f:
            json.dump(raw, f, ensure_ascii=False, indent=4)
        POST_PROCESSOR.reload()
        return jsonify({"success": True, "dictionary": raw})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/post-process', methods=['POST'])
def api_post_process():
    """Execute post-processing on given text."""
    if not POST_PROCESSOR:
        return jsonify({"success": False, "error": "Post-processor not available"}), 503
    try:
        data = request.get_json() or {}
        text = data.get('text', '')
        system_prompt = data.get('system_prompt')
        
        if not text:
            return jsonify({'success': False, 'error': 'Không có văn bản'}), 400
            
        result = POST_PROCESSOR.process(text, system_prompt=system_prompt)
        # Assuming result['processed'] is the final text
        # If the user script expects 'text', we should return 'text'
        return jsonify({"success": True, "text": result.get("processed", text)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/post-process/test', methods=['POST'])
def api_pp_test():
    """Test post-processing on given text."""
    if not POST_PROCESSOR:
        return jsonify({"success": False, "error": "Post-processor not available"}), 503
    try:
        data = request.get_json() or {}
        text = data.get('text', '')
        result = POST_PROCESSOR.process(text)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("VietASR Pro — Server")
    print("=" * 70)

    # Start background model loading thread
    import threading
    t = threading.Thread(target=background_load_model)
    t.daemon = True
    t.start()

    print("\n✓ Server đang khởi động ở chế độ tải model nền (background model loading)!")
    print(f"  Mở trình duyệt tại: http://{os.environ.get('FLASK_HOST', '127.0.0.1')}:{os.environ.get('FLASK_PORT', os.environ.get('PORT', '5000'))}")
    print("\nNhấn Ctrl+C để dừng\n")

    # UPGRADE 2025: Đọc host/port từ biến môi trường
    # Ưu tiên: FLASK_HOST/FLASK_PORT -> PORT (Render) -> mặc định
    _host = os.environ.get("FLASK_HOST", "0.0.0.0")
    _port = int(
        os.environ.get("FLASK_PORT")
        or os.environ.get("PORT")
        or 5000
    )
    socketio.run(app, debug=False, host=_host, port=_port, allow_unsafe_werkzeug=True)
