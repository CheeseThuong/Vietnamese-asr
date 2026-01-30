"""
Language Model integration để cải thiện kết quả decoding
Sử dụng n-gram LM với KenLM và pyctcdecode
"""
import os
from pathlib import Path
from typing import List, Optional
import json

def build_ngram_lm(text_corpus_file: str, output_arpa: str, output_binary: str, ngram_order: int = 5):
    """
    Build n-gram language model từ text corpus sử dụng KenLM
    
    Args:
        text_corpus_file: File chứa corpus text (mỗi dòng là một câu)
        output_arpa: Output ARPA format file
        output_binary: Output binary KenLM file
        ngram_order: Order của n-gram (mặc định 5-gram)
    """
    import subprocess
    
    print(f"Building {ngram_order}-gram language model...")
    
    # Build ARPA file
    print("Step 1: Building ARPA format...")
    lmplz_cmd = f"lmplz -o {ngram_order} --text {text_corpus_file} --arpa {output_arpa} --discount_fallback"
    
    try:
        subprocess.run(lmplz_cmd, shell=True, check=True)
        print(f"✓ ARPA file created: {output_arpa}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to build ARPA file: {e}")
        return False
    
    # Convert to binary format
    print("Step 2: Converting to binary format...")
    binary_cmd = f"build_binary {output_arpa} {output_binary}"
    
    try:
        subprocess.run(binary_cmd, shell=True, check=True)
        print(f"✓ Binary LM created: {output_binary}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to build binary LM: {e}")
        return False

def prepare_lm_corpus(jsonl_files: List[str], output_file: str):
    """
    Chuẩn bị corpus text từ các file JSONL để train language model
    """
    import json
    from data_preprocessing import VietnameseASRDataset
    
    print("Preparing LM corpus...")
    all_texts = []
    
    for jsonl_file in jsonl_files:
        dataset = VietnameseASRDataset(jsonl_file)
        for item in dataset.data:
            text = dataset.normalize_text(item['transcript'])
            if text.strip():  # Chỉ lấy các text không rỗng
                all_texts.append(text)
    
    # Ghi ra file
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in all_texts:
            f.write(text + '\n')
    
    print(f"✓ Corpus prepared with {len(all_texts)} sentences")
    print(f"  Saved to: {output_file}")
    
    return output_file

class LanguageModelDecoder:
    """
    Decoder với language model support sử dụng pyctcdecode
    """
    def __init__(
        self,
        processor,
        kenlm_model_path: Optional[str] = None,
        alpha: float = 0.5,
        beta: float = 1.0
    ):
        """
        Args:
            processor: Wav2Vec2Processor
            kenlm_model_path: Path to KenLM binary model
            alpha: Language model weight
            beta: Word insertion bonus
        """
        self.processor = processor
        self.alpha = alpha
        self.beta = beta
        
        if kenlm_model_path and os.path.exists(kenlm_model_path):
            self._init_lm_decoder(kenlm_model_path)
        else:
            print("⚠ No LM provided, using greedy decoding")
            self.decoder = None
    
    def _init_lm_decoder(self, kenlm_model_path: str):
        """
        Initialize pyctcdecode decoder với KenLM
        """
        try:
            from pyctcdecode import build_ctcdecoder
            
            # Get vocabulary
            vocab_dict = self.processor.tokenizer.get_vocab()
            sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
            vocab_list = [token for token, _ in sorted_vocab]
            
            # Build decoder
            self.decoder = build_ctcdecoder(
                labels=vocab_list,
                kenlm_model_path=kenlm_model_path,
                alpha=self.alpha,
                beta=self.beta
            )
            
            print(f"✓ LM decoder initialized with alpha={self.alpha}, beta={self.beta}")
            
        except ImportError:
            print("⚠ pyctcdecode not available, falling back to greedy decoding")
            self.decoder = None
        except Exception as e:
            print(f"⚠ Failed to initialize LM decoder: {e}")
            self.decoder = None
    
    def decode(self, logits, beam_width: int = 100):
        """
        Decode logits với hoặc không có language model
        
        Args:
            logits: Model output logits [batch, time, vocab]
            beam_width: Beam width for beam search
        """
        import numpy as np
        
        if self.decoder is not None:
            # Decode with LM using beam search
            if len(logits.shape) == 3:
                # Batch decoding
                texts = []
                for logit in logits:
                    text = self.decoder.decode(
                        logit.cpu().numpy(),
                        beam_width=beam_width
                    )
                    texts.append(text)
                return texts
            else:
                # Single decoding
                return self.decoder.decode(
                    logits.cpu().numpy(),
                    beam_width=beam_width
                )
        else:
            # Greedy decoding without LM
            import torch
            pred_ids = torch.argmax(logits, dim=-1)
            return self.processor.batch_decode(pred_ids)

def train_transformer_lm(corpus_file: str, output_dir: str, model_name: str = "vinai/phobert-base"):
    """
    Fine-tune một transformer model (như PhoBERT) làm language model
    để post-process kết quả ASR
    
    Args:
        corpus_file: Text corpus file
        output_dir: Directory to save trained model
        model_name: Pre-trained model name
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments
    )
    from datasets import load_dataset
    
    print(f"Training transformer LM based on {model_name}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # Load and tokenize corpus
    dataset = load_dataset('text', data_files={'train': corpus_file})
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=256)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=100,
        fp16=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train'],
    )
    
    # Train
    trainer.train()
    
    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Transformer LM trained and saved to: {output_dir}")

def main():
    """
    Main function để build language model
    """
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'processed_data'
    lm_dir = base_dir / 'language_models'
    lm_dir.mkdir(exist_ok=True)
    
    # Prepare corpus
    print("="*50)
    print("PREPARING LANGUAGE MODEL CORPUS")
    print("="*50)
    
    corpus_file = lm_dir / 'lm_corpus.txt'
    prepare_lm_corpus(
        [
            str(data_dir / 'train.jsonl'),
            str(data_dir / 'validation.jsonl')
        ],
        str(corpus_file)
    )
    
    # Build n-gram LM
    print("\n" + "="*50)
    print("BUILDING N-GRAM LANGUAGE MODEL")
    print("="*50)
    
    arpa_file = lm_dir / 'vietnamese_5gram.arpa'
    binary_file = lm_dir / 'vietnamese_5gram.bin'
    
    success = build_ngram_lm(
        str(corpus_file),
        str(arpa_file),
        str(binary_file),
        ngram_order=5
    )
    
    if success:
        print("\n✓ N-gram LM build completed!")
    else:
        print("\n✗ N-gram LM build failed. Please install KenLM:")
        print("  pip install https://github.com/kpu/kenlm/archive/master.zip")
    
    # Optionally: Train transformer LM
    print("\n" + "="*50)
    print("TRAINING TRANSFORMER LANGUAGE MODEL (OPTIONAL)")
    print("="*50)
    print("This step is optional and takes longer...")
    
    # Uncomment to train transformer LM
    # train_transformer_lm(
    #     str(corpus_file),
    #     str(lm_dir / 'phobert_lm'),
    #     model_name="vinai/phobert-base"
    # )

if __name__ == "__main__":
    main()
