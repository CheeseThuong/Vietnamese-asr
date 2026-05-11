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

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
from io import BytesIO
import time
import torch
import torchaudio
import librosa
import numpy as np
import os
import tempfile
import json
import platform
import traceback
import random
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
    global MODEL, PROCESSOR, DEVICE

    if MODEL is None or PROCESSOR is None:
        return {"success": False, "error": "Model not loaded"}

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
        fallback_models = [
            "nguyenvulebinh/wav2vec2-large-vi-vlsp2020",  # Vietnamese - large
            "nguyenvulebinh/wav2vec2-base-vi",             # Vietnamese - base
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

def get_new_tokens(prev_text: str, new_text: str) -> str:
    """
    So sánh prev_text và new_text, trả về phần MỚI của new_text
    không bị trùng với đuôi của prev_text.
    Dùng longest common suffix matching.
    """
    if not prev_text:
        return new_text.strip()
        
    prev_words = prev_text.strip().split()
    new_words = new_text.strip().split()
    
    max_overlap = min(len(prev_words), len(new_words), 10)
    overlap_len = 0
    
    for n in range(max_overlap, 0, -1):
        if prev_words[-n:] == new_words[:n]:
            overlap_len = n
            break
            
    new_part = new_words[overlap_len:]
    return ' '.join(new_part)

@socketio.on('connect')
def handle_connect():
    SESSION_BUFFERS[request.sid] = {
        'audio': np.array([], dtype=np.float32),
        'confirmed_text': '',
        'chunk_id': 0
    }
    emit('server_ready', {'message': 'Connected to ASR Server'})

@socketio.on('disconnect')
def handle_disconnect():
    SESSION_BUFFERS.pop(request.sid, None)

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    # FIX: Removed 5 debug print statements that spammed console on every chunk
    if not isinstance(data, bytes):
        return

    pcm_data = np.frombuffer(data, dtype=np.float32)

    # Guard: ignore empty or corrupted chunks
    if pcm_data.size == 0:
        return

    started_at = time.perf_counter()
    session = SESSION_BUFFERS.get(request.sid)
    if not session:
        return

    # Append new data to buffer
    session['audio'] = np.concatenate([session['audio'], pcm_data])

    # FIX: Keep max 4 seconds of audio for sliding window (16000 sr * 4s = 64000)
    # Previously the buffer grew unbounded when no new tokens were emitted
    MAX_BUFFER = 16000 * 4
    if len(session['audio']) > MAX_BUFFER:
        session['audio'] = session['audio'][-MAX_BUFFER:]

    # Transcribe the buffer
    result = transcribe_audio_array(session['audio'], 16000)
    new_transcript = result.get('transcription', '')

    new_part = get_new_tokens(session['confirmed_text'], new_transcript)
    if new_part:
        session['confirmed_text'] += (' ' + new_part) if session['confirmed_text'] else new_part
        session['chunk_id'] += 1

        latency = round((time.perf_counter() - started_at) * 1000)

        emit('transcript_update', {
            'text': new_part.strip(),
            'full_text': session['confirmed_text'].strip(),
            'chunk_id': session['chunk_id'],
            'is_final': False,
            'latency': latency
        })


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
        "status": "ready" if MODEL else "not_loaded",
        "model_source": MODEL_SOURCE,
        "available_devices": device_info
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check alias for Render and other deployment platforms"""
    return api_status()

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
            if ground_truth:
                result['ground_truth'] = ground_truth
                if compute_wer:
                    try:
                        result['wer'] = round(float(compute_wer(ground_truth, result['transcription'])), 4)
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
    """Summarize transcript using Gemini via PostProcessor."""
    if not POST_PROCESSOR:
        return jsonify({"success": False, "error": "Post-processor not available"}), 503
    try:
        data = request.get_json() or {}
        text = data.get('text', '').strip()
        mode = data.get('mode', 'summary')

        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400

        prompts = {
            'summary': "Hay tom tat noi dung chinh cua doan van ban ASR sau day. Trinh bay chuyen nghiep, ngan gon:",
            'meeting': "Hay tao bien ban cuoc hop tu doan van ban ASR sau. Bao gom: chu de, noi dung chinh, quyet dinh, hanh dong tiep theo:",
            'notes': "Hay tao ghi chu hoc tap tu doan van ban ASR sau. Trinh bay dang bullet points:",
            'translate': "Dich doan van ban tieng Viet sau sang tieng Anh. Chi tra ve ban dich:",
        }
        system_prompt = prompts.get(mode, prompts['summary'])

        result = POST_PROCESSOR.process(text, system_prompt=system_prompt)
        return jsonify({
            "success": True,
            "text": result.get("processed", text),
            "mode": mode,
            "steps_applied": result.get("steps_applied", []),
        })
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


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
    print("Vietnamese ASR Web Demo")
    print("=" * 70)
    
    # Load model
    model_loaded = load_model()
    
    if model_loaded:
        print("\n✓ Server ready!")
        print("  Open: http://localhost:5000")
        print("\nPress Ctrl+C to stop\n")
    else:
        print("\n⚠ Server starting without model")
        print("  Transcription will not work until model is loaded")
        print("  Web interface will still be available\n")
    
    # Run Flask app with SocketIO
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, debug=False, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
