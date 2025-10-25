# Chatterbox Project Setup Guide

This guide provides all necessary information to replicate the Chatterbox project environment for generating chess coaching TTS audio clips.

## Prerequisites

- **Python Version**: 3.11.9 (compatible >=3.10)
- **Operating System**: Linux (current setup), though supporting Windows/macOS
- **Hardware Requirements**: 
  - GPU with CUDA support (recommended for faster model inference)
  - CUDA 12.x compatible (based on torch 2.6.0)

## Main Dependencies

The project uses the Chatterbox TTS library (v0.1.4), which requires the following core packages:

- **chatterbox-tts==0.1.4**
- **torch==2.6.0**
- **torchaudio==2.6.0**
- **transformers==4.46.3**
- **diffusers==0.29.0**
- **numpy>=1.24.0,<1.26.0**
- **librosa==0.11.0**
- **safetensors==0.5.3**

Additional dependencies installed via chatterbox-tts:
- s3tokenizer
- resemble-perth==1.0.1
- conformer==0.3.2
- pkuseg==0.0.25
- pykakasi==2.3.0
- gradio==5.44.1

## Main Model

- **ChatterboxTTS**: Open source multilingual TTS model by Resemble AI
- **Size**: ~1B parameters
- **Capabilities**: 
  - 23 language support including English, French, Spanish, German, etc.
  - Zero-shot voice cloning
  - Emotion exaggeration control
  - Voice conversion
- **Download**: Automatically downloaded on first run via `ChatterboxTTS.from_pretrained(device="cuda")`

## Project Files

### Python Scripts (Main Execution Files)
- `GPU_TEST.py`: Test GPU availability and model loading
- `5.py`, `10.py`, `15.py`, `20.py`, `30.py`, `40.py`: Generate TTS clips of different durations
- `batch5.py`, `batch10(5-10).py`, `batch10(10-15).py`: Generate batches of short clips
- `chatterbox.bat`: Windows batch file wrapper

### Configuration/Output Files
- `ref.wav`: Reference audio file for voice cloning (required for all scripts)
- `obsv.md`: Observations/notes file
- `OUTPUT/`, `OUTPUT10_5-10/`, `OUTPUT10_10-15/`: Generated audio file directories
- `venv/`: Virtual environment (contains all dependencies)

### Documentation
- `README.md`: Project README
- `setup.md`: This setup guide

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/DarshanS26/chatterbox.git
cd chatterbox-project
```

### 2. Set Up Virtual Environment (Same as Original)
Since the venv was created with Python 3.11.9 on Windows, for Linux:

```bash
python3.11 -m venv venv  # Use Python 3.11.9 if available, or 3.12.3
```

If exact Python 3.11.9 not available:
```bash
python3 -m venv venv
```

### 3. Install Chatterbox TTS
```bash
# Activate venv
source venv/bin/activate  # Linux/macOS
# or
./venv/bin/activate.ps1   # PowerShell

# Install chatterbox-tts
pip install chatterbox-tts==0.1.4
```

### 4. Verify Instalation
Run GPU test:
```python
python GPU_TEST.py
```

## Usage

### Basic Text-to-Speech
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
text = "Great knight move! You control the center splendidly."
wav = model.generate(text)
ta.save("output.wav", wav, model.sr)
```

### Voice Cloning (Used in Project Scripts)
```python
wav = model.generate(
    text,
    audio_prompt_path="ref.wav",
    exaggeration=0.85,
    cfg_weight=0.45,
    temperature=0.8
)
```

### Key Parameters
- `audio_prompt_path`: Path to reference wav for voice cloning
- `exaggeration`: Emotion intensity (0.0-1.0)
- `cfg_weight`: Guidance weight (0.0-1.0, lower for less strict voice following)
- `temperature`: Sampling temperature (affects variability)

## Hardware Acceleration

- **GPU**: CUDA-compatible, tested with CUDA 12.x
- **CPU**: Fallback available but much slower
- Set `device="cpu"` for CPU inference (not recommended for batch processing)

## Generated Content

The project generates chess coaching commentary with varying durations:
- 5-40 second clips of motivational chess advice
- Short 5-10 second encouraging phrases
- All using voice cloning from `ref.wav` for consistent voice

## Troubleshooting

- **CUDA Issues**: Ensure CUDA toolkit matches PyTorch version
- **Memory Issues**: Large models require ~8-16GB VRAM for fast inference
- **Slow Inference**: Use CPU device or reduce batch sizes
- **Watermarked**: All outputs include perceptual watermarks (built-in feature)

## Requirements File (Generated from venv)
If you want to install all dependencies manually:

```
chatterbox-tts==0.1.4
# Other dependencies are pulled automatically
```

## Notes

- Models are downloaded on first use (~4GB download)
- Voice cloning requires a clean reference.wav sample
- Project generates motivational chess coaching TTS in various lengths
- Built for AI chess coach application integration
