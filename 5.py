import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
text = "Amazing! Your knight move instantly controls the center and sets you up for victory."
wav = model.generate(
    text,
    audio_prompt_path="ref.wav",
    exaggeration=0.85,
    cfg_weight=0.45,
    temperature=0.8
)
ta.save("clip_5s.wav", wav, model.sr)
print("✓ Saved 5-second clip: clip_5s.wav")
