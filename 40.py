import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

# Script that fits ~30 seconds, with clear sections
text = (
    "Alright! Let's break down this tricky chess position together. "
    "You know, not every move needs to be flashy—sometimes a simple move is the best move. "
    "Remember how earlier you backed your rook into the corner? Haha, that always reminds me how even masters slip up sometimes... "
    "Now, watch closely: your knight is covering some key squares on the king side. If you shift your queen to e5, "
    "you’ll control the center and even set up a possible check. "
    "The secret, and I'm whispering this to you—never tell your opponent!—is that sometimes the quiet moves are the most dangerous ones. "
    "Give it a try, and let’s see if you can spot the hidden tactic. Good luck!"
)

# Main section: expressive coaching and slight laugh
# Whisper section: secret (delivered more softly and with "I'm whispering this to you" cues)

wav = model.generate(
    text,
    audio_prompt_path="ref.wav",    # Clone your voice
    exaggeration=0.8,               # More expressive (friendly, engaging)
    cfg_weight=0.3,                 # Deliberate pacing for clarity
    temperature=0.7                 # Some unpredictability for naturalness
)

ta.save("coaching_secret_laugh_30s.wav", wav, model.sr)

audio_duration = len(wav[0]) / model.sr
print(f"✓ Audio saved: coaching_secret_laugh_30s.wav")
print(f"Duration: {audio_duration:.2f} seconds (target ~30s)")
