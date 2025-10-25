import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Load the model once ("hot" engine for fast batch generation)
model = ChatterboxTTS.from_pretrained(device="cuda")
reference_voice = "ref.wav"

# List of (filename, text) pairs for five ~10 sec clips
clips = [
    ("clip1", "Nice play! Your knight controls key squares and sets up for a strong center attack."),
    ("clip2", "Clever defense—moving your rook keeps your back rank protected and ready for counterplay."),
    ("clip3", "This position is tense. Consider how your bishop can pressure your opponent's queen next."),
    ("clip4", "Your pawn structure is solid; now it's the perfect moment to develop your queen-side pieces."),
    ("clip5", "Remember, sometimes waiting and preparing is just as important as launching the first attack.")
]

for name, text in clips:
    wav = model.generate(
        text,
        audio_prompt_path=reference_voice,
        exaggeration=0.75,    # Expressive, natural for advice
        cfg_weight=0.45,      # Balanced pacing
        temperature=0.7       # Conversational variety
    )
    ta.save(f"{name}.wav", wav, model.sr)
    audio_duration = len(wav[0]) / model.sr
    print(f"✓ Saved {name}.wav - Duration: {audio_duration:.2f} seconds")

print("All batch clips generated!")
