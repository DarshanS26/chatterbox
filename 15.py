import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
text = ("A well-timed pawn break can open the center and unleash your bishops. Wait for the right moment, "
        "and use your pawns to create attacking chances. This kind of patient buildup is what separates strong players from beginners.")
wav = model.generate(
    text,
    audio_prompt_path="ref.wav",
    exaggeration=0.75,
    cfg_weight=0.37,
    temperature=0.65
)
ta.save("clip_15s.wav", wav, model.sr)
print("✓ Saved 15-second clip: clip_15s.wav")
