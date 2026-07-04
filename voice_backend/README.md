# Rack Guardian Voice Backend

Install and run:

```powershell
python -m pip install -r requirements-voice.txt
python voice_main.py
```

The service listens on `http://127.0.0.1:8000`.

Reusable Python API:

```python
from voice_backend import GradiumVoice

voice = GradiumVoice()
wav_bytes = await voice.text_to_speech("Rack R-04 requires attention.")
transcript = await voice.speech_to_text(wav_audio, content_type="audio/wav")
```

HTTP endpoints:

- `GET /health`
- `POST /tts` with `{"text": "..."}`
- `POST /stt` with raw WAV, PCM, Ogg, or Opus audio bytes
