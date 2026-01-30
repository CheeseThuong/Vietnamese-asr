"""
FastAPI backend cho ASR web application
Hỗ trợ upload file audio và real-time transcription
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import torch
import torchaudio
import tempfile
import os
import time
from typing import Optional, Dict
import logging

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from src.training.language_model import LanguageModelDecoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vietnamese ASR API",
    description="Automatic Speech Recognition API cho tiếng Việt",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
processor = None
decoder = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class TranscriptionResponse(BaseModel):
    """Response model for transcription"""
    text: str
    processing_time: float
    language_model_used: bool
    audio_duration: Optional[float] = None

class ModelConfig(BaseModel):
    """Configuration for model loading"""
    model_path: str
    lm_path: Optional[str] = None
    use_lm: bool = True

def load_model(model_path: str, lm_path: Optional[str] = None):
    """
    Load Wav2Vec2 model và language model
    """
    global model, processor, decoder
    
    try:
        logger.info(f"Loading model from: {model_path}")
        model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
        model.eval()
        
        logger.info(f"Loading processor from: {model_path}")
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        
        # Load language model decoder
        decoder = LanguageModelDecoder(
            processor,
            kenlm_model_path=lm_path
        )
        
        logger.info("✓ Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def transcribe_audio(audio_path: str, use_lm: bool = True) -> Dict:
    """
    Transcribe audio file
    """
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Load audio
        audio_array, sampling_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            audio_array = resampler(audio_array)
        
        # Convert to mono if stereo
        if audio_array.shape[0] > 1:
            audio_array = torch.mean(audio_array, dim=0, keepdim=True)
        
        audio_array = audio_array.squeeze().numpy()
        audio_duration = len(audio_array) / 16000
        
        # Process audio
        input_values = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(input_values).logits
        
        # Decode
        if use_lm and decoder.decoder is not None:
            transcription = decoder.decode(logits, beam_width=100)
            if isinstance(transcription, list):
                transcription = transcription[0]
            lm_used = True
        else:
            pred_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(pred_ids)[0]
            lm_used = False
        
        processing_time = time.time() - start_time
        
        return {
            "text": transcription,
            "processing_time": processing_time,
            "language_model_used": lm_used,
            "audio_duration": audio_duration
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """
    Load model khi khởi động server
    """
    base_dir = Path(__file__).parent
    model_dir = base_dir / 'models' / 'wav2vec2-vietnamese-asr' / 'final_model'
    lm_dir = base_dir / 'language_models' / 'vietnamese_5gram.bin'
    
    if not model_dir.exists():
        logger.warning(f"Model not found at: {model_dir}")
        logger.warning("Please train the model first or update model path")
        return
    
    lm_path = str(lm_dir) if lm_dir.exists() else None
    success = load_model(str(model_dir), lm_path)
    
    if not success:
        logger.error("Failed to load model at startup")

@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Vietnamese ASR API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "language_model_available": decoder.decoder is not None if decoder else False
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    use_lm: bool = True
):
    """
    Transcribe uploaded audio file
    
    Args:
        file: Audio file (wav, mp3, flac, etc.)
        use_lm: Whether to use language model for decoding
    
    Returns:
        Transcription result
    """
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Supported: wav, mp3, flac, ogg, m4a"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Transcribe
        result = transcribe_audio(temp_path, use_lm=use_lm)
        return TranscriptionResponse(**result)
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/transcribe-stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    use_lm: bool = True
):
    """
    Transcribe audio stream (for real-time recording)
    """
    # Similar to transcribe_file but optimized for streaming
    return await transcribe_file(file, use_lm)

@app.post("/load-model")
async def load_model_endpoint(config: ModelConfig):
    """
    Load or reload model
    """
    lm_path = config.lm_path if config.use_lm else None
    success = load_model(config.model_path, lm_path)
    
    if success:
        return {"message": "Model loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")

@app.get("/model-info")
async def model_info():
    """
    Get information about loaded model
    """
    if model is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": model.__class__.__name__,
        "device": device,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "vocab_size": len(processor.tokenizer) if processor else None,
        "language_model_loaded": decoder.decoder is not None if decoder else False
    }

# Mount static files (for web UI)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    @app.get("/app")
    async def serve_app():
        """Serve web application"""
        return FileResponse(str(static_dir / "index.html"))

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
