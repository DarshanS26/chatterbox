import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
reference_voice = "ref.wav"

clips = [
    ("quick1", "Great move! Your queen just took control of the board."),
    ("quick2", "That pawn push was smart, opening up new possibilities."),
    ("quick3", "Keep your pieces active—don't let them get stuck."),
    ("quick4", "Nice! You found a tactic that gains material."),
    ("quick5", "Remember, always protect your king before attacking."),
    ("quick6", "Quick thinking! You defended just in time."),
    ("quick7", "Sometimes the best plan is a simple one."),
    ("quick8", "Watch your opponent’s threats and respond calmly."),
    ("quick9", "Setting up a discovered attack can win the game."),
    ("quick10", "Keep looking for new ideas every move you play.")
]

for name, text in clips:
    wav = model.generate(
        text,
        audio_prompt_path=reference_voice,
        exaggeration=0.8,    # Upbeat, energetic
        cfg_weight=0.5,      # Fast, clear delivery
        temperature=0.75     # Lively, less monotone
    )
    ta.save(f"{name}.wav", wav, model.sr)
    audio_duration = len(wav[0]) / model.sr
    print(f"✓ Saved {name}.wav - Duration: {audio_duration:.2f} seconds")

print("All short batch clips generated!")
