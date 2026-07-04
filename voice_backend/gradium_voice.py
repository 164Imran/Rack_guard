import json
import os

import httpx


class GradiumVoice:
    """Small async Gradium client designed to be imported by any Python main."""

    BASE_URL = "https://api.gradium.ai/api/post/speech"

    def __init__(self, api_key: str | None = None, voice_id: str | None = None):
        self.api_key = api_key or os.environ["GRADIUM_API_KEY"]
        self.voice_id = voice_id or os.getenv("GRADIUM_VOICE_ID", "YTpq7expH9539ERJ")

    async def text_to_speech(self, text: str) -> bytes:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.BASE_URL}/tts",
                headers={"x-api-key": self.api_key},
                json={
                    "text": text,
                    "voice_id": self.voice_id,
                    "output_format": "wav",
                    "only_audio": True,
                },
            )
            response.raise_for_status()
            return response.content

    async def speech_to_text(
        self,
        audio: bytes,
        content_type: str = "audio/wav",
        language: str = "en",
    ) -> str:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{self.BASE_URL}/asr",
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": content_type,
                },
                params={
                    "model": "default",
                    "json_config": json.dumps({"language": language}),
                },
                content=audio,
            )
            response.raise_for_status()

        segments: list[str] = []
        for line in response.text.splitlines():
            if not line.strip():
                continue
            message = json.loads(line)
            if message.get("type") == "error":
                raise RuntimeError(message.get("message", "Gradium STT failed"))
            if message.get("type") == "text" and message.get("text"):
                segments.append(message["text"].strip())
        return " ".join(segments).strip()
