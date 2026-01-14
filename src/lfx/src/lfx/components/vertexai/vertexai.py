"""Patched Vertex AI component using google-genai SDK for Gemini 3 support.

This replaces the default ChatVertexAI component to use the google-genai SDK directly,
enabling Gemini 3 models with thinking capabilities without requiring langchain-google-genai 4.x.
"""

from typing import cast

from lfx.base.models.model import LCModelComponent
from lfx.field_typing import LanguageModel
from lfx.inputs.inputs import MessageTextInput
from lfx.io import DropdownInput, FileInput, FloatInput, IntInput, StrInput


class ChatVertexAIComponent(LCModelComponent):
    display_name = "Vertex AI"
    description = "Generate text using Vertex AI LLMs (Gemini 3 supported via google-genai SDK)."
    icon = "VertexAI"
    name = "VertexAiModel"

    inputs = [
        *LCModelComponent.get_base_inputs(),
        FileInput(
            name="credentials",
            display_name="Credentials",
            info="JSON credentials file. Leave empty to use Application Default Credentials (ADC).",
            file_types=["json"],
        ),
        MessageTextInput(
            name="model_name",
            display_name="Model Name",
            value="gemini-3-flash-preview",
        ),
        StrInput(
            name="project",
            display_name="Project",
            info="The GCP project ID.",
            advanced=True,
        ),
        StrInput(
            name="location",
            display_name="Location",
            value="global",
            advanced=True,
        ),
        IntInput(
            name="max_output_tokens",
            display_name="Max Output Tokens",
            advanced=True,
        ),
        FloatInput(
            name="temperature",
            value=1.0,
            display_name="Temperature",
        ),
        IntInput(
            name="top_k",
            display_name="Top K",
            advanced=True,
        ),
        FloatInput(
            name="top_p",
            display_name="Top P",
            value=0.95,
            advanced=True,
        ),
        DropdownInput(
            name="thinking_level",
            display_name="Thinking Level",
            options=["off", "low", "medium", "high"],
            value="off",
            info="Reasoning depth for Gemini 3 models. Higher = deeper reasoning.",
        ),
    ]

    def build_model(self) -> LanguageModel:
        from lfx.base.models.google_genai_model import ChatGoogleGenAI

        return cast(
            "LanguageModel",
            ChatGoogleGenAI(
                model=self.model_name,
                vertexai=True,
                project=self.project or None,
                location=self.location or None,
                credentials_path=self.credentials or None,
                temperature=self.temperature,
                top_k=self.top_k or None,
                top_p=self.top_p,
                max_output_tokens=self.max_output_tokens or None,
                thinking_level=self.thinking_level if self.thinking_level != "off" else None,
            ),
        )
