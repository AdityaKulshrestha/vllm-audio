"""OpenAI-compatible /v1/audio/speech endpoint."""

from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Optional

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.protocol import SpeechRequest  # TODO: Implement this
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.logger import init_logger
from vllm.outputs import SpeechRequestOutput, TTSOutput  # TODO: Implement this 
from vllm.sampling_params import SamplingParams


logger = init_logger(__name__)


class OpenAIServingSpeech(OpenAIServing):
    """
    Serving class for /v1/audio/speech endpoint

    Key difference from other endpoint: returns raw audio bytes,
    not JSON. This matches OpenAI's TTS API.
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: list[str] | None,
        *,
        request_logger: Optional[object] = None,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
        )

    async def create_speech(
        self,
        request: SpeechRequest,
        raw_request: Request,
    ) -> Response | StreamingResponse:
        """
        Handle /v1/audio/speech requests

        Returns raw audio bytes (not JSON), compatible with OpenAI's API.
        """

        # Validate model
        error = self._validate_model(request)

        if error:
            return self.create_error_response(error)
        
        # Draft prompt for the TTS model
        # Format depends on the model
        prompt = self._format_tts_prompt(request.input, request.voice)

        # Get sampling params
        sampling_params = request.to_sampling_params()

        # Generate
        request_id = f"req-{self._generate_request_id()}"


        try:
            if request.stream:
                return await self._stream_speech(
                    request, prompt, sampling_params, request_id
                )
            else:
                return await self._generate_speech(
                    request, prompt, sampling_params, request_id
                )
        except Exception as e:
            logger.exception(f"Error processing speech request {request_id}: {e}")
            return self.create_error_response(str(e))    
        
    async def _generate_speech(
        self,
        request: SpeechRequest,
        prompt: str,
        sampling_params: dict,
        request_id: str,
    ) -> Response:
        """Non-streaming speech generation"""
    
        # Submit to engine with TTS flag
        result_generator = self.engine_client.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            # is_tts_request=True,  TODO: Check for this flag
            # tts_response_format=request.response_format,
            # tts_voice=request.voice,
        )

        # Collect result
        final_output: SpeechRequestOutput | None = None

        async for output in result_generator:
            if isinstance(output, SpeechRequestOutput):
                final_output = output

        if final_output is None:
            return self.create_error_response("No audio generated")
        
        # Return raw audio bytes
        audio_bytes = final_output.to_response_bytes()
        content_type = final_output.content_type 

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format}"',           
            },
        )
    
    async def _stream_speech(
        self,
        request: SpeechRequest,
        prompt: str,
        sampling_params,
        request_id: str
    ) -> StreamingResponse:
        """Streaming speech generation (audio chunks)"""


        async def audio_stream() -> AsyncGenerator[bytes, None]:
            async for output in self.engine_client.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                # is_tts_request=True,
                # tts_response_format=request.response_format
            ):
                if isinstance(output, SpeechRequestOutput):
                    for chunk in output.audio_chunks:
                        yield chunk

        content_type = TTSOutput.get_content_type(request.response_format)

        return StreamingResponse(
            audio_stream(),
            media_type=content_type,
            headers={"X-Request-ID": request_id},
        )
        
    def _format_tts_prompt(self, text: str, voice: str) -> str:
        """
        Format input text into model-specific prompt.

        """
        return f"<speak><voice name='{voice}'>{text}</voice></speak>"

    # TODO: Add sampling params in the request
    def _build_speech_sampling_params(self, request: SpeechRequest) -> dict:
        """Build sampling params for TTS requests"""
        params = request.to_sampling_params()

        return params
    
    def _validate_model(self, request: SpeechRequest) -> Optional[str]:
        """Validate if the requested model supports TTS"""
        if request.model not in self.models:
            return f"Model '{request.model}' not found."