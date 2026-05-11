import pandas as pd

# Read training history
df = pd.read_csv(r'results\training_history.csv')

print('\n' + '='*70)
print('PHÂN TÍCH TIẾN ĐỘ TRAINING')
print('='*70)

# Basic stats
print(f'\n📊 THỐNG KÊ CHUNG:')
print(f'   Tổng steps: {int(df["step"].max())}')
print(f'   Tổng epochs: {df["epoch"].max():.2f}')

# Training loss
print(f'\n📉 TRAINING LOSS:')
print(f'   Loss ban đầu: {df["loss"].iloc[0]:.4f}')
print(f'   Loss cuối cùng: {df["loss"].dropna().iloc[-1]:.4f}')
print(f'   Giảm được: {df["loss"].iloc[0] - df["loss"].dropna().iloc[-1]:.4f}')

# Evaluation metrics
eval_df = df[df["eval_wer"].notna()].copy()
print(f'\n📈 EVALUATION METRICS (tất cả {len(eval_df)} checkpoints):')
print()
print(eval_df[["step", "epoch", "eval_loss", "eval_wer", "eval_cer"]].to_string(index=False))

# Latest metrics
if len(eval_df) > 0:
    last = eval_df.iloc[-1]
    print(f'\n🎯 METRICS CUỐI CÙNG (Step {int(last["step"])}):')
    print(f'   Eval Loss: {last["eval_loss"]:.4f}')
    print(f'   WER: {last["eval_wer"]:.4f} (0.0 = hoàn hảo, 1.0 = lỗi 100%)')
    print(f'   CER: {last["eval_cer"]:.4f}')

print('\n' + '='*70)
print('CHẨN ĐOÁN')
print('='*70)

if eval_df["eval_wer"].iloc[-1] >= 0.99:
    print('\n❌ VẤN ĐỀ NGHIÊM TRỌNG: WER = 1.0')
    print('   → Model dự đoán SAI 100% từ')
    print('   → Hoàn toàn KHÔNG HỌC được gì')
    print()
    print('🔍 Nguyên nhân có thể:')
    print('   1. Vocab không khớp giữa training và inference')
    print('   2. Dữ liệu preprocessing sai')
    print('   3. Learning rate không phù hợp')
    print('   4. Model architecture sai')
    print('   5. Dataset quá nhỏ hoặc không đúng format')
    print()
    print('✅ GIẢI PHÁP KHUYẾN NGHỊ:')
    print('   → Dùng pretrained model: nguyenvulebinh/wav2vec2-large-vi-vlsp2020')
    print('   → Hoặc kiểm tra lại toàn bộ training pipeline')
elif eval_df["eval_wer"].iloc[-1] < 0.3:
    print('\n✅ Model training TỐT!')
    print(f'   WER: {eval_df["eval_wer"].iloc[-1]:.2%} (chấp nhận được)')
else:
    print('\n⚠️ Model training TRUNG BÌNH')
    print(f'   WER: {eval_df["eval_wer"].iloc[-1]:.2%}')
    print('   Cần train thêm hoặc điều chỉnh hyperparameters')

print('='*70)
