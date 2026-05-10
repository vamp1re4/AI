"""
Voice interface for assistant output.

Provides a simple text-to-speech fallback for environments without
an advanced TTS engine installed.
"""

import os
import shutil
import subprocess
from typing import Optional

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class VoiceSystem:
    """Voice system wrapper."""

    def __init__(self, default_output: str = 'voice_output.wav'):
        self.default_output = default_output

    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """Synthesize speech to a file or simulate speech output."""
        output_path = output_path or self.default_output

        if pyttsx3 is not None:
            engine = pyttsx3.init()
            if output_path.endswith('.txt'):
                engine.save_to_file(text, output_path.replace('.txt', '.wav'))
                engine.runAndWait()
                return output_path.replace('.txt', '.wav')
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            return output_path

        if shutil.which('espeak') is not None:
            command = ['espeak', '-w', output_path, text]
            subprocess.run(command, check=False)
            return output_path

        if output_path.endswith('.wav'):
            output_path = output_path.replace('.wav', '.txt')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        return output_path

    def speak(self, text: str) -> None:
        """Emit speech to the console or play it if available."""
        if pyttsx3 is not None:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        elif shutil.which('espeak') is not None:
            subprocess.run(['espeak', text], check=False)
        else:
            print(f"[voice] {text}")
