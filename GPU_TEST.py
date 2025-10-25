import torch
from chatterbox.tts import ChatterboxTTS

print('CUDA available:', torch.cuda.is_available())
print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')

print('Loading model...')
model = ChatterboxTTS.from_pretrained(device='cuda')
print('✓ Model loaded on GPU!')
