import numpy as np

import app.app as app_module


def test_realtime_asr_uses_local_wav2vec2_only(monkeypatch):
    calls = {"local": 0}

    def fake_local_asr(*args, **kwargs):
        calls["local"] += 1
        return {
            "success": True,
            "transcription": "xin chao",
            "original_transcription": "xin chao",
            "asr_metrics": {},
        }

    def forbidden_cloud_asr(*args, **kwargs):
        raise AssertionError("Realtime ASR must not call cloud ASR providers")

    monkeypatch.setattr(app_module, "transcribe_audio_array", fake_local_asr)
    monkeypatch.setattr(app_module, "transcribe_with_google_api_direct", forbidden_cloud_asr)
    monkeypatch.setattr(app_module, "transcribe_with_gemini_api", forbidden_cloud_asr)
    monkeypatch.setattr(app_module, "REALTIME_ASR_ENGINE", "gemini")

    audio = np.zeros(16000, dtype=np.float32)
    result = app_module.transcribe_realtime_array(audio, sampling_rate=16000)

    assert result["success"] is True
    assert result["transcription"] == "xin chao"
    assert calls["local"] == 1
