"""
Diarization Configuration
=========================
Cấu hình cho Speaker Diarization engine.
"""

import os

# Backend: "simple-diarizer" (offline) hoặc "resemblyzer" (fallback)
# KHÔNG dùng pyannote (cần HuggingFace token)
DIARIZATION_BACKEND = os.getenv("DIARIZATION_BACKEND", "simple-diarizer")

# HuggingFace token — chỉ cần nếu dùng pyannote (disabled)
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Speaker limits
MIN_SPEAKERS = 2
MAX_SPEAKERS = 4

# Minimum audio duration for reliable diarization (seconds)
MIN_AUDIO_DURATION = 10.0

# Speaker color/label mapping
SPEAKER_COLORS = {
    "SPEAKER_00": {
        "hex": "#1565C0",
        "rgb": "21,101,192",
        "name": "Người nói 1",
    },
    "SPEAKER_01": {
        "hex": "#B71C1C",
        "rgb": "183,28,28",
        "name": "Người nói 2",
    },
    "SPEAKER_02": {
        "hex": "#1B5E20",
        "rgb": "27,94,32",
        "name": "Người nói 3",
    },
    "SPEAKER_03": {
        "hex": "#E65100",
        "rgb": "230,81,0",
        "name": "Người nói 4",
    },
}

# Unknown speaker (confidence < threshold)
UNKNOWN_SPEAKER = {
    "hex": "#7B1FA2",
    "rgb": "123,31,162",
    "name": "Người nói ?",
}

# Confidence threshold for speaker assignment
CONFIDENCE_THRESHOLD = 0.6
