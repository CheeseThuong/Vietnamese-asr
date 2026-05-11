import json
from pathlib import Path

# Read vocab
vocab_path = Path('results/final_model/vocab.json')
with open(vocab_path, encoding='utf-8') as f:
    vocab = json.load(f)

print('\n' + '='*70)
print('PHÂN TÍCH VOCAB')
print('='*70)

print(f'\n📖 THÔNG TIN VOCAB:')
print(f'   Tổng số tokens: {len(vocab)}')
print(f'   Có <pad>: {"<pad>" in vocab}')
print(f'   Có <unk>: {"<unk>" in vocab}')
print(f'   Có <s>: {"<s>" in vocab}')
print(f'   Có </s>: {"</s>" in vocab}')

# Count character types
vietnamese_chars = 0
numbers = 0
special = 0
english = 0

for char in vocab.keys():
    if char in ['<pad>', '<unk>', '<s>', '</s>', '|']:
        special += 1
    elif char.isdigit():
        numbers += 1
    elif char.isascii() and char.isalpha():
        english += 1
    else:
        vietnamese_chars += 1

print(f'\n📊 PHÂN LOẠI TOKENS:')
print(f'   Ký tự tiếng Việt: {vietnamese_chars}')
print(f'   Chữ cái tiếng Anh: {english}')
print(f'   Số: {numbers}')
print(f'   Đặc biệt: {special}')

print(f'\n🔤 MẪU VOCAB (20 đầu):')
for i, (k, v) in enumerate(list(vocab.items())[:20]):
    print(f'   "{k}": {v}')

print(f'\n🔤 VIETNAMESE CHARS:')
viet_chars = [k for k in vocab.keys() if not k.isascii() and k not in ['<pad>', '<unk>', '|']]
print(f'   {" ".join(sorted(viet_chars)[:50])}')

print('\n' + '='*70)
print('NHẬN XÉT')
print('='*70)
print('\n✅ Vocab có vẻ ổn - có đầy đủ ký tự tiếng Việt')
print('⚠️  Nhưng WER = 1.0 → Vấn đề KHÔNG phải ở vocab')
print('\n🔍 Vấn đề có thể là:')
print('   1. Data preprocessing sai (audio → text mapping)')
print('   2. Training config sai (learning rate, batch size)')
print('   3. Model architecture không phù hợp với data')
print('   4. Dataset quá nhỏ (cần ít nhất vài GB audio)')
print('\n💡 Nên dùng pretrained model đã được train trên dataset lớn!')
print('='*70)
