"""
====================================================================
@file: diarizer.py
@description: Speaker Diarization Engine
              Nhận diện và phân biệt người nói trong file âm thanh.
              Hỗ trợ 2 backend: 
               - simple-diarizer (x-vector)
               - resemblyzer + sklearn KMeans clustering
@author: Nguyễn Trí Thượng
@project: VietASR Pro
@email: nguyentrithuong471@gmail.com
@github: CheeseThuong
@version: 2.0.0
====================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

# Import config
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config.diarization_config import (
        DIARIZATION_BACKEND,
        MIN_SPEAKERS,
        MAX_SPEAKERS,
        MIN_AUDIO_DURATION,
        SPEAKER_COLORS,
        UNKNOWN_SPEAKER,
        CONFIDENCE_THRESHOLD,
    )
except ImportError:
    # Fallback defaults if config not found
    DIARIZATION_BACKEND = "resemblyzer"
    MIN_SPEAKERS = 2
    MAX_SPEAKERS = 4
    MIN_AUDIO_DURATION = 10.0
    CONFIDENCE_THRESHOLD = 0.6
    SPEAKER_COLORS = {
        "SPEAKER_00": {"hex": "#1565C0", "rgb": "21,101,192", "name": "Người nói 1"},
        "SPEAKER_01": {"hex": "#B71C1C", "rgb": "183,28,28", "name": "Người nói 2"},
        "SPEAKER_02": {"hex": "#1B5E20", "rgb": "27,94,32", "name": "Người nói 3"},
        "SPEAKER_03": {"hex": "#E65100", "rgb": "230,81,0", "name": "Người nói 4"},
    }
    UNKNOWN_SPEAKER = {"hex": "#7B1FA2", "rgb": "123,31,162", "name": "Nguoi noi ?"}

# Detect available backend
_BACKEND = None
_SIMPLE_DIARIZER_AVAILABLE = False
_RESEMBLYZER_AVAILABLE = False

try:
    from simple_diarizer.diarizer import Diarizer as SimpleDiarizer
    _SIMPLE_DIARIZER_AVAILABLE = True
except ImportError:
    pass

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from sklearn.cluster import KMeans, SpectralClustering
    _RESEMBLYZER_AVAILABLE = True
except ImportError:
    pass

if DIARIZATION_BACKEND == "simple-diarizer" and _SIMPLE_DIARIZER_AVAILABLE:
    _BACKEND = "simple-diarizer"
elif _RESEMBLYZER_AVAILABLE:
    _BACKEND = "resemblyzer"
elif _SIMPLE_DIARIZER_AVAILABLE:
    _BACKEND = "simple-diarizer"
else:
    _BACKEND = None

print(f"[Diarizer] Backend: {_BACKEND or 'NONE (not installed)'}")
if _BACKEND is None:
    print("[Diarizer] Install: pip install simple-diarizer  OR  pip install resemblyzer scikit-learn")


def _format_time(seconds: float) -> str:
    """Format seconds to M:SS or H:MM:SS."""
    seconds = max(0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class SpeakerDiarizer:
    """
    Speaker Diarization engine.

    Phân biệt người nói trong file âm thanh.
    Hỗ trợ 2-4 người nói.
    """

    SPEAKER_COLORS = {
        k: v["hex"] for k, v in SPEAKER_COLORS.items()
    }
    SPEAKER_LABELS = {
        k: v["name"] for k, v in SPEAKER_COLORS.items()
    } if isinstance(list(SPEAKER_COLORS.values())[0], dict) else {}

    def __init__(self):
        # Re-read from the module-level config dicts
        self._colors = {}
        self._labels = {}
        for key, val in SPEAKER_COLORS.items():
            if isinstance(val, dict):
                self._colors[key] = val["hex"]
                self._labels[key] = val["name"]
            else:
                self._colors[key] = val
                self._labels[key] = key

        self._encoder = None  # Lazy-load resemblyzer encoder
        self._simple_diarizer = None  # Lazy-load simple-diarizer

    def _get_color(self, speaker_id: str) -> str:
        return self._colors.get(speaker_id, UNKNOWN_SPEAKER["hex"])

    def _get_label(self, speaker_id: str) -> str:
        return self._labels.get(speaker_id, UNKNOWN_SPEAKER["name"])

    @property
    def available(self) -> bool:
        return _BACKEND is not None

    @property
    def backend_name(self) -> str:
        return _BACKEND or "none"

    # ==================================================================
    # Main diarization entry point
    # ==================================================================
    def diarize(
        self,
        audio_path: str,
        min_speakers: int = MIN_SPEAKERS,
        max_speakers: int = MAX_SPEAKERS,
    ) -> List[Dict]:
        """
        Diarize audio file.

        Args:
            audio_path: Path to audio file (WAV 16kHz mono preferred)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect

        Returns:
            List of segments: [{"speaker": "SPEAKER_00", "start": 0.0, "end": 3.2,
                                "color": "#1565C0", "label": "Nguoi noi 1"}]
        """
        if not self.available:
            print("[Diarizer] No backend available, returning empty segments")
            return []

        audio_path = str(audio_path)
        if not os.path.exists(audio_path):
            print(f"[Diarizer] File not found: {audio_path}")
            return []

        # Check audio duration
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            if duration < MIN_AUDIO_DURATION:
                print(f"[Diarizer] Audio too short ({duration:.1f}s < {MIN_AUDIO_DURATION}s)")
                # Return single speaker for short audio
                return [{
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": duration,
                    "color": self._get_color("SPEAKER_00"),
                    "label": self._get_label("SPEAKER_00"),
                }]
        except Exception as e:
            print(f"[Diarizer] Could not get duration: {e}")

        min_speakers = max(1, min(min_speakers, MAX_SPEAKERS))
        max_speakers = max(min_speakers, min(max_speakers, MAX_SPEAKERS))

        try:
            if _BACKEND == "simple-diarizer":
                return self._diarize_simple(audio_path, min_speakers, max_speakers)
            elif _BACKEND == "resemblyzer":
                return self._diarize_resemblyzer(audio_path, min_speakers, max_speakers)
            else:
                return []
        except Exception as e:
            print(f"[Diarizer] Diarization failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    # ==================================================================
    # Backend 1: simple-diarizer
    # ==================================================================
    def _diarize_simple(
        self, audio_path: str, min_speakers: int, max_speakers: int
    ) -> List[Dict]:
        """Diarize using simple-diarizer library."""
        if self._simple_diarizer is None:
            self._simple_diarizer = SimpleDiarizer(embed_model="xvec")

        print(f"[Diarizer] simple-diarizer: processing {audio_path}")
        start_time = time.time()

        segments = self._simple_diarizer.diarize(
            audio_path,
            num_speakers=None,  # auto-detect
            threshold=0.5,
        )

        elapsed = time.time() - start_time
        print(f"[Diarizer] Done in {elapsed:.1f}s, {len(segments)} segments")

        # Convert to standard format
        result = []
        for seg in segments:
            speaker_id = seg.get("label", seg.get("speaker", "SPEAKER_00"))
            # Normalize speaker ID
            if not speaker_id.startswith("SPEAKER_"):
                try:
                    idx = int(speaker_id.replace("speaker", "").replace("_", ""))
                    speaker_id = f"SPEAKER_{idx:02d}"
                except (ValueError, AttributeError):
                    speaker_id = "SPEAKER_00"

            # Clamp to max speakers
            idx = int(speaker_id.split("_")[1])
            if idx >= max_speakers:
                speaker_id = f"SPEAKER_{(idx % max_speakers):02d}"

            result.append({
                "speaker": speaker_id,
                "start": round(seg.get("start", 0.0), 2),
                "end": round(seg.get("end", 0.0), 2),
                "color": self._get_color(speaker_id),
                "label": self._get_label(speaker_id),
            })

        return self._merge_adjacent(result)

    # ==================================================================
    # Backend 2: resemblyzer + KMeans (fallback)
    # ==================================================================
    def _diarize_resemblyzer(
        self, audio_path: str, min_speakers: int, max_speakers: int
    ) -> List[Dict]:
        """Diarize using resemblyzer embeddings + sklearn clustering."""
        import librosa

        if self._encoder is None:
            print("[Diarizer] Loading resemblyzer VoiceEncoder...")
            self._encoder = VoiceEncoder()
            print("[Diarizer] VoiceEncoder loaded")

        print(f"[Diarizer] resemblyzer: processing {audio_path}")
        start_time = time.time()

        # Load and preprocess audio
        wav = preprocess_wav(audio_path)
        duration = len(wav) / 16000.0

        # Create sliding windows (1.5s windows, 0.75s step)
        window_size = int(1.5 * 16000)
        step_size = int(0.75 * 16000)
        windows = []
        window_times = []

        for start_sample in range(0, len(wav) - window_size + 1, step_size):
            end_sample = start_sample + window_size
            segment = wav[start_sample:end_sample]
            # Skip silent segments
            if np.max(np.abs(segment)) < 0.01:
                continue
            windows.append(segment)
            window_times.append({
                "start": start_sample / 16000.0,
                "end": end_sample / 16000.0,
            })

        if len(windows) == 0:
            return [{
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": duration,
                "color": self._get_color("SPEAKER_00"),
                "label": self._get_label("SPEAKER_00"),
            }]

        # Extract embeddings
        embeddings = np.array([self._encoder.embed_utterance(w) for w in windows])

        # Determine optimal number of speakers using silhouette score
        best_n = min_speakers
        if min_speakers < max_speakers and len(embeddings) >= max_speakers:
            from sklearn.metrics import silhouette_score
            best_score = -1
            for n in range(min_speakers, max_speakers + 1):
                if n >= len(embeddings):
                    break
                try:
                    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embeddings)
                    if len(set(labels)) < 2:
                        continue
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_n = n
                except Exception:
                    continue

        # Final clustering
        kmeans = KMeans(n_clusters=best_n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Build segments
        segments = []
        for i, (time_info, label) in enumerate(zip(window_times, labels)):
            speaker_id = f"SPEAKER_{label:02d}"
            segments.append({
                "speaker": speaker_id,
                "start": round(time_info["start"], 2),
                "end": round(time_info["end"], 2),
                "color": self._get_color(speaker_id),
                "label": self._get_label(speaker_id),
            })

        elapsed = time.time() - start_time
        print(f"[Diarizer] Done in {elapsed:.1f}s, {len(segments)} raw segments, {best_n} speakers")

        return self._merge_adjacent(segments)

    # ==================================================================
    # Utilities
    # ==================================================================
    @staticmethod
    def _merge_adjacent(segments: List[Dict], gap_threshold: float = 0.5) -> List[Dict]:
        """Merge adjacent segments of the same speaker."""
        if not segments:
            return segments

        merged = [segments[0].copy()]
        for seg in segments[1:]:
            prev = merged[-1]
            if (
                seg["speaker"] == prev["speaker"]
                and seg["start"] - prev["end"] <= gap_threshold
            ):
                prev["end"] = max(prev["end"], seg["end"])
            else:
                merged.append(seg.copy())

        return merged

    def assign_transcript_to_speakers(
        self,
        segments: List[Dict],
        transcript_chunks: List[Dict],
    ) -> List[Dict]:
        """
        Align transcript chunks (with timestamps) to speaker segments.

        Args:
            segments: Diarization segments [{speaker, start, end, color, label}]
            transcript_chunks: [{text, start, end}] from Wav2Vec2 transcription

        Returns:
            List of dialogue lines:
            [{speaker, label, color, text, start, end}]
        """
        if not segments or not transcript_chunks:
            # If no diarization, assign all to SPEAKER_00
            return [{
                "speaker": "SPEAKER_00",
                "label": self._get_label("SPEAKER_00"),
                "color": self._get_color("SPEAKER_00"),
                "text": " ".join(c.get("text", "") for c in transcript_chunks),
                "start": transcript_chunks[0].get("start", 0.0) if transcript_chunks else 0.0,
                "end": transcript_chunks[-1].get("end", 0.0) if transcript_chunks else 0.0,
            }]

        dialogue = []
        for chunk in transcript_chunks:
            chunk_start = chunk.get("start", 0.0)
            chunk_end = chunk.get("end", chunk_start)
            chunk_mid = (chunk_start + chunk_end) / 2.0
            text = chunk.get("text", "").strip()

            if not text:
                continue

            # Find the speaker segment that best overlaps with this chunk
            best_speaker = "SPEAKER_00"
            best_overlap = 0.0

            for seg in segments:
                # Calculate overlap
                overlap_start = max(chunk_start, seg["start"])
                overlap_end = min(chunk_end, seg["end"])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = seg["speaker"]

            # Fallback: if no overlap found, use midpoint
            if best_overlap == 0:
                for seg in segments:
                    if seg["start"] <= chunk_mid <= seg["end"]:
                        best_speaker = seg["speaker"]
                        break

            dialogue.append({
                "speaker": best_speaker,
                "label": self._get_label(best_speaker),
                "color": self._get_color(best_speaker),
                "text": text,
                "start": round(chunk_start, 2),
                "end": round(chunk_end, 2),
            })

        # Merge consecutive lines from the same speaker
        return self._merge_dialogue_lines(dialogue)

    @staticmethod
    def _merge_dialogue_lines(lines: List[Dict]) -> List[Dict]:
        """Merge consecutive dialogue lines from the same speaker."""
        if not lines:
            return lines

        merged = [lines[0].copy()]
        for line in lines[1:]:
            prev = merged[-1]
            if line["speaker"] == prev["speaker"]:
                prev["text"] += " " + line["text"]
                prev["end"] = line["end"]
            else:
                merged.append(line.copy())

        return merged

    def format_as_dialogue(self, dialogue_lines: List[Dict]) -> str:
        """
        Format dialogue lines as human-readable text.

        Output:
            [Nguoi noi 1 - 0:00]: Xin chao...
            [Nguoi noi 2 - 0:05]: Chao ban...
        """
        lines = []
        for dl in dialogue_lines:
            time_str = _format_time(dl.get("start", 0))
            label = dl.get("label", "?")
            text = dl.get("text", "")
            lines.append(f"[{label} - {time_str}]: {text}")
        return "\n".join(lines)

    def get_speaker_stats(self, dialogue_lines: List[Dict]) -> Dict:
        """
        Calculate per-speaker statistics.

        Returns:
            {
                "SPEAKER_00": {"label": "...", "color": "...",
                               "words": 45, "duration": 12.3, "turns": 5,
                               "percentage": 55.2},
                ...
            }
        """
        stats = {}
        total_duration = 0.0

        for dl in dialogue_lines:
            speaker = dl.get("speaker", "SPEAKER_00")
            text = dl.get("text", "")
            duration = dl.get("end", 0) - dl.get("start", 0)
            word_count = len(text.split()) if text.strip() else 0

            if speaker not in stats:
                stats[speaker] = {
                    "label": dl.get("label", speaker),
                    "color": dl.get("color", "#666"),
                    "words": 0,
                    "duration": 0.0,
                    "turns": 0,
                }

            stats[speaker]["words"] += word_count
            stats[speaker]["duration"] += duration
            stats[speaker]["turns"] += 1
            total_duration += duration

        # Calculate percentages
        for speaker in stats:
            if total_duration > 0:
                stats[speaker]["percentage"] = round(
                    (stats[speaker]["duration"] / total_duration) * 100, 1
                )
            else:
                stats[speaker]["percentage"] = 0.0
            stats[speaker]["duration"] = round(stats[speaker]["duration"], 1)

        return stats

    def format_srt(self, dialogue_lines: List[Dict]) -> str:
        """Format dialogue as SRT subtitle file content."""
        srt_lines = []
        for i, dl in enumerate(dialogue_lines, 1):
            start = dl.get("start", 0)
            end = dl.get("end", 0)
            label = dl.get("label", "?")
            text = dl.get("text", "")

            start_h = int(start // 3600)
            start_m = int((start % 3600) // 60)
            start_s = int(start % 60)
            start_ms = int((start % 1) * 1000)

            end_h = int(end // 3600)
            end_m = int((end % 3600) // 60)
            end_s = int(end % 60)
            end_ms = int((end % 1) * 1000)

            srt_lines.append(str(i))
            srt_lines.append(
                f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d} --> "
                f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}"
            )
            srt_lines.append(f"[{label}] {text}")
            srt_lines.append("")

        return "\n".join(srt_lines)


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    d = SpeakerDiarizer()
    print(f"Backend: {d.backend_name}")
    print(f"Available: {d.available}")

    if d.available:
        import sys
        if len(sys.argv) > 1:
            audio_file = sys.argv[1]
            print(f"\nDiarizing: {audio_file}")
            segments = d.diarize(audio_file, min_speakers=2, max_speakers=4)
            for seg in segments:
                print(f"  {_format_time(seg['start'])} - {_format_time(seg['end'])}: "
                      f"{seg['label']} ({seg['speaker']})")
        else:
            print("Usage: python diarizer.py <audio_file>")
    else:
        print("No diarization backend installed.")
        print("Install: pip install simple-diarizer  OR  pip install resemblyzer scikit-learn")
