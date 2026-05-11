from io import BytesIO
from pathlib import Path

import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def main() -> None:
    dataset = load_dataset("linhtran92/viet_bud500", data_dir="data", split="train[:1]")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    sample = dataset[0]
    audio = sample["audio"]
    transcript = sample["transcription"]

    print("GROUND_TRUTH:", transcript)
    print("AUDIO_KEYS:", list(audio.keys()))
    print("AUDIO_PATH:", audio.get("path"))
    print("BYTES_LEN:", None if audio.get("bytes") is None else len(audio["bytes"]))

    if audio.get("bytes") is None:
        raise RuntimeError("Dataset sample does not contain raw audio bytes")

    data, sampling_rate = sf.read(BytesIO(audio["bytes"]))
    print("RAW_SR:", sampling_rate)
    print("RAW_SHAPE:", getattr(data, "shape", None))

    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype("float32")

    if sampling_rate != 16000:
        import librosa

        data = librosa.resample(data, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000

    temp_audio = Path("d:/Projects/do_an_tri_tue_nhan_tao/tmp_viet_bud500_sample.wav")
    sf.write(temp_audio, data, sampling_rate)

    model_path = Path("d:/Projects/do_an_tri_tue_nhan_tao/final_model")
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    inputs = processor(data, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    prediction = processor.batch_decode(pred_ids)[0]

    print("PREDICTION:", prediction)
    print("TEMP_WAV:", temp_audio)


if __name__ == "__main__":
    main()
