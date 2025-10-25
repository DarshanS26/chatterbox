import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
text = ("Careful now—your queen is hanging with no protection, and your opponent is planning a discovered attack. "
        "Spot the danger and find a safe square before it's too late.")
wav = model.generate(
    text,
    audio_prompt_path="ref.wav",
    exaggeration=0.70,
    cfg_weight=0.40,
    temperature=0.7
)
ta.save("clip_10s.wav", wav, model.sr)
print("✓ Saved 10-second clip: clip_10s.wav")
