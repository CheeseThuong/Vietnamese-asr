"""Test audio file upload via API"""
import requests, os, random

audio_dir = r'D:\Projects\do_an_tri_tue_nhan_tao\Data\Data\vlsp2020_train_p1'

wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
random.seed(42)
selected = random.sample(wav_files, min(5, len(wav_files)))

print('=' * 60)
print('AUDIO FILE UPLOAD TEST')
print('=' * 60)
print(f'Found {len(wav_files)} wav files, testing {len(selected)}')
print()

results = []
for i, fname in enumerate(selected, 1):
    fpath = os.path.join(audio_dir, fname)
    fsize = os.path.getsize(fpath) / 1024
    print(f'[{i}/{len(selected)}] {fname} ({fsize:.1f} KB)')

    try:
        with open(fpath, 'rb') as f:
            resp = requests.post(
                'http://localhost:5000/api/upload',
                files={'file': (fname, f, 'audio/wav')},
                timeout=120
            )
        data = resp.json()

        if data.get('success'):
            transcript = data.get('transcription', '(empty)')
            proc_time = data.get('processing_time_seconds', 0)
            print(f'  Status: SUCCESS')
            print(f'  Transcript: {transcript}')
            print(f'  Time: {proc_time}s')
            results.append({
                'file': fname, 'status': 'SUCCESS',
                'transcript': transcript, 'time': proc_time
            })
        else:
            error = data.get('error', 'Unknown error')
            print(f'  Status: FAILED - {error}')
            results.append({'file': fname, 'status': 'FAILED', 'error': error})
    except Exception as e:
        print(f'  Status: ERROR - {e}')
        results.append({'file': fname, 'status': 'ERROR', 'error': str(e)})
    print()

print('=' * 60)
print('SUMMARY')
print('=' * 60)
success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
print(f'Success: {success_count}/{len(results)}')
print(f'Failed:  {len(results) - success_count}/{len(results)}')
print()
for r in results:
    icon = 'OK' if r['status'] == 'SUCCESS' else 'FAIL'
    detail = r.get('transcript', r.get('error', ''))
    if len(detail) > 60:
        detail = detail[:60] + '...'
    print(f'  [{icon}] {r["file"]}')
    print(f'         -> {detail}')
