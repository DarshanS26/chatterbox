import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
text = (
    "Let me give you a pro tip—sometimes, the best move isn't obvious at all. Try developing quietly, improving your pieces without revealing your true plan. "
    "Here's a secret: if you lift your rook to the third rank, you can quickly switch it for either an attack or defense depending on how your opponent reacts. "
    "Watch carefully and you'll see how top players use this trick in their own games."
)
wav = model.generate(
    text,
    audio_prompt_path="ref.wav",
    exaggeration=0.8,
    cfg_weight=0.35,
    temperature=0.7
)
ta.save("clip_20s.wav", wav, model.sr)
print("✓ Saved 20-second clip: clip_20s.wav")
