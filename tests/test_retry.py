"""Retry failed audio uploads"""
import requests, os

audio_dir = r'D:\Projects\do_an_tri_tue_nhan_tao\Data\Data\vlsp2020_train_p1'
files_to_retry = [
    'database_sa1_Jan08_Mar19_cleaned_utt_0000001354-1.wav',
    'database_sa1_Jan08_Mar19_cleaned_utt_0000000241-1.wav'
]

for fname in files_to_retry:
    fpath = os.path.join(audio_dir, fname)
    print(f'Retrying: {fname}')
    with open(fpath, 'rb') as f:
        resp = requests.post('http://localhost:5000/api/upload',
            files={'file': (fname, f, 'audio/wav')}, timeout=120)
    data = resp.json()
    if data.get('success'):
        t = data.get('transcription', '')
        pt = data.get('processing_time_seconds', 0)
        print(f'  SUCCESS: {t}')
        print(f'  Time: {pt}s')
    else:
        print(f'  FAILED: {data.get("error", "")}')
    print()
