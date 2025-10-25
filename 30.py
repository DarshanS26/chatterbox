import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
reference_voice = "ref.wav"

text = (
    "Let's look at this position together. Your knight is strong on f3, covering crucial squares, while your pawns in the center give you stable control and lots of possibilities. Remember, sometimes the best moves are not flashy but quietly build up your position. If you can bring your queen towards e5 soon, you'll apply pressure and control the board. By the way, here's something I've learned from top players and I'm whispering this just for you, don't let anyone else hear—often, the simple rook lift can catch an opponent off guard and turn the whole game around. Try to spot those quiet tactics, because that's how champions plan their attacks. So stay focused, trust your intuition, and let's make the next move your best move yet."
)

wav = model.generate(
    text,
    audio_prompt_path=reference_voice,  # Clone the voice
    exaggeration=0.8,                   # Expressive, friendly
    cfg_weight=0.4,                     # Natural pacing
    temperature=0.7                     # Conversational variety
)

ta.save("continuous_natural_coaching.wav", wav, model.sr)
audio_duration = len(wav[0]) / model.sr
print(f"✓ Audio saved: continuous_natural_coaching.wav")
print(f"Duration: {audio_duration:.2f} seconds")
