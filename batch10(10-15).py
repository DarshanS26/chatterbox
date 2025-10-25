import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
reference_voice = "ref.wav"

clips = [
    ("clip1", "Your control over the center is impressive. By keeping your pawns advanced and your minor pieces active, you set up potential attacks and limit your opponent’s options. Well done focusing on space early in the game."),
    ("clip2", "That rook move was clever—it not only defends your back rank, but it also prepares your pieces to launch a counterattack if your opponent gets careless. Always think ahead with your defenses."),
    ("clip3", "Consider how your bishop could pressure your opponent’s queen in the next few moves. By putting their strongest piece on the defensive, you create more opportunities for surprise tactics."),
    ("clip4", "You’ve built a solid position, but now is the perfect time to develop your queen-side pieces. This will help you control both flanks and create threats that force your opponent to respond."),
    ("clip5", "Remember, in chess, waiting patiently and preparing your plan can be just as powerful as a direct attack. Build up your forces, and look for moments when you can open up the board."),
    ("clip6", "When your knight jumps into the opponent’s territory, always check if there’s a way to fork two valuable pieces. These opportunities can win you material and create lasting pressure."),
    ("clip7", "Be careful with your pawn pushes—sometimes advancing too quickly can leave weaknesses behind. Plan carefully and make sure every pawn move supports your overall strategy."),
    ("clip8", "Swapping queens early isn’t always the best choice. Before trading, make sure your endgame pieces are well positioned so you can keep pressure once the queens are gone."),
    ("clip9", "If your king is exposed, consider castling for safety, or defending with pawns and minor pieces. A well-protected king allows you to focus on your attack without worrying about sudden threats."),
    ("clip10", "Check for new tactics after every move. Sometimes, a quiet repositioning of your rook or bishop can set up a powerful attack—even if your opponent isn’t expecting it.")
]

for name, text in clips:
    wav = model.generate(
        text,
        audio_prompt_path=reference_voice,
        exaggeration=0.75,    # Expressive but not exaggerated
        cfg_weight=0.45,      # Clear delivery, natural pace
        temperature=0.7       # Conversational variety
    )
    ta.save(f"{name}.wav", wav, model.sr)
    audio_duration = len(wav[0]) / model.sr
    print(f"✓ Saved {name}.wav - Duration: {audio_duration:.2f} seconds")

print("All 10 batch clips generated!")
