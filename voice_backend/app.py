import json
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .gradium_voice import GradiumVoice

load_dotenv(".env.local")


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=800)


def create_app() -> FastAPI:
    app = FastAPI(title="Rack Guardian Voice API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def voice_client() -> GradiumVoice:
        try:
            return GradiumVoice()
        except KeyError as error:
            raise HTTPException(503, "GRADIUM_API_KEY is not configured") from error

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "provider": "gradium"}

    @app.post("/tts")
    async def tts(payload: TTSRequest) -> Response:
        try:
            audio = await voice_client().text_to_speech(payload.text)
            return Response(audio, media_type="audio/wav")
        except httpx.HTTPError as error:
            raise HTTPException(502, "Gradium TTS request failed") from error

    @app.post("/stt")
    async def stt(request: Request) -> dict[str, str]:
        audio = await request.body()
        if not audio or len(audio) > 12_000_000:
            raise HTTPException(400, "Audio must contain between 1 byte and 12 MB")
        content_type = request.headers.get("content-type", "audio/wav").split(";")[0]
        if content_type not in {"audio/wav", "audio/pcm", "audio/ogg", "audio/opus"}:
            raise HTTPException(415, "Unsupported audio format")
        try:
            text = await voice_client().speech_to_text(
                audio,
                content_type=content_type,
                language=os.getenv("GRADIUM_STT_LANGUAGE", "en"),
            )
            return {"text": text}
        except (httpx.HTTPError, json.JSONDecodeError, RuntimeError) as error:
            raise HTTPException(502, "Gradium STT request failed") from error

    return app
