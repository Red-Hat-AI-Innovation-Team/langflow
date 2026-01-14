"""Wrapper for Google GenAI SDK to work with LangChain's BaseChatModel interface.

This module provides a bridge between the google-genai SDK and langchain,
enabling Gemini 3 support without requiring langchain-google-genai 4.x.

Implements feature parity with langchain-google-vertexai's ChatVertexAI.
"""

from __future__ import annotations

import ast
import base64
import json
import logging
import uuid
from operator import itemgetter

# Key for storing thought signatures in additional_kwargs
# Must match langchain-google-vertexai for compatibility
_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY = "__gemini_function_call_thought_signatures__"


def _bytes_to_base64(data: bytes) -> str:
    """Convert bytes to base64 string for JSON-safe storage."""
    return base64.b64encode(data).decode("utf-8")


def _base64_to_bytes(data: str) -> bytes:
    """Convert base64 string back to bytes."""
    return base64.b64decode(data.encode("utf-8"))


from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

from typing_extensions import TypedDict

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.messages.tool import invalid_tool_call, tool_call_chunk
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    parse_tool_calls,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, Field, SecretStr
from typing_extensions import is_typeddict

if TYPE_CHECKING:
    from google.genai import Client
    from google.genai.types import GenerateContentResponse

logger = logging.getLogger(__name__)


def _get_content_role(content: Any) -> str | None:
    """Get role from a content object (dict or types.Content)."""
    if hasattr(content, "role"):
        return content.role
    elif isinstance(content, dict):
        return content.get("role")
    return None


def _get_content_parts(content: Any) -> list:
    """Get parts from a content object (dict or types.Content)."""
    if hasattr(content, "parts"):
        return list(content.parts) if content.parts else []
    elif isinstance(content, dict):
        return content.get("parts", [])
    return []


def _extend_content_parts(content: Any, new_parts: list) -> None:
    """Extend parts of a content object (dict or types.Content)."""
    if hasattr(content, "parts"):
        if content.parts is None:
            content.parts = []
        content.parts.extend(new_parts)
    elif isinstance(content, dict):
        if "parts" not in content:
            content["parts"] = []
        content["parts"].extend(new_parts)


class _ToolConfigDict(TypedDict, total=False):
    """Tool configuration dictionary."""

    function_calling_config: Dict[str, Any]


class ChatGoogleGenAI(BaseChatModel):
    """Chat model that uses the google-genai SDK directly for Gemini 3 support.

    This wrapper allows using Gemini 3 models with the existing langchain 0.3.x stack
    by directly interfacing with the google-genai SDK instead of langchain-google-genai.

    Implements feature parity with ChatVertexAI including:
    - Tool/function calling with tool_choice and tool_config
    - Structured output (with_structured_output)
    - Multimodal support (images, media)
    - Safety settings
    - Async streaming
    - Retry logic
    """

    model: str = Field(default="gemini-3-flash-preview", description="The model name to use")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    top_p: float | None = Field(default=None, description="Top-p sampling parameter")
    top_k: int | None = Field(default=None, description="Top-k sampling parameter")
    max_output_tokens: int | None = Field(default=None, description="Maximum output tokens")
    stop_sequences: list[str] | None = Field(default=None, description="Stop sequences")
    candidate_count: int = Field(default=1, description="Number of candidates to generate")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")

    # Vertex AI settings
    vertexai: bool = Field(default=False, description="Use Vertex AI backend")
    project: str | None = Field(default=None, description="Google Cloud project ID")
    location: str | None = Field(default=None, description="Google Cloud location")
    credentials_path: str | None = Field(default=None, description="Path to service account JSON")

    # API key (for non-Vertex AI)
    api_key: SecretStr | None = Field(default=None, description="Google API key")

    # Thinking config (Gemini 3 specific)
    thinking_level: Literal["off", "low", "medium", "high"] | None = Field(
        default=None, description="Thinking level for Gemini 3 models"
    )
    thinking_budget: int | None = Field(
        default=None, description="Thinking budget tokens (deprecated, use thinking_level)"
    )

    # Response format
    response_mime_type: str | None = Field(default=None, description="Response MIME type")
    response_schema: dict | None = Field(
        default=None, description="Response schema for structured output"
    )

    # Safety settings
    safety_settings: list[dict] | None = Field(
        default=None,
        description="Safety settings for content filtering. List of dicts with 'category' and 'threshold' keys.",
    )

    # Retry settings
    max_retries: int = Field(default=6, description="Maximum number of retries for API calls")

    # Internal client cache
    _client: Client | None = None

    # Tools for function calling
    tools: list[Any] | None = Field(default=None, description="Bound tools for function calling")
    tool_config: _ToolConfigDict | None = Field(default=None, description="Tool configuration")

    class Config:
        arbitrary_types_allowed = True

    def bind_tools(
        self,
        tools: Sequence[Any],
        tool_config: Optional[_ToolConfigDict] = None,
        *,
        tool_choice: Optional[Union[str, bool]] = None,
        **kwargs: Any,
    ) -> "ChatGoogleGenAI":
        """Bind tools to this model for function calling.

        Args:
            tools: List of tools to bind (can be functions, pydantic models, or tool dicts)
            tool_config: Optional tool configuration dict with function_calling_config
            tool_choice: Optional tool choice - can be:
                - None: Model decides whether to use tools
                - True: Model must use at least one tool
                - False: Model cannot use tools
                - str: Name of specific tool to use
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            A new ChatGoogleGenAI instance with tools bound
        """
        # Handle tool_choice
        final_tool_config = tool_config
        if tool_choice is not None and tool_config is None:
            final_tool_config = self._tool_choice_to_tool_config(tool_choice, tools)

        return ChatGoogleGenAI(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
            stop_sequences=self.stop_sequences,
            candidate_count=self.candidate_count,
            seed=self.seed,
            vertexai=self.vertexai,
            project=self.project,
            location=self.location,
            credentials_path=self.credentials_path,
            api_key=self.api_key,
            thinking_level=self.thinking_level,
            thinking_budget=self.thinking_budget,
            response_mime_type=self.response_mime_type,
            response_schema=self.response_schema,
            safety_settings=self.safety_settings,
            max_retries=self.max_retries,
            tools=list(tools),
            tool_config=final_tool_config,
        )

    def _tool_choice_to_tool_config(
        self, tool_choice: Union[str, bool], tools: Sequence[Any]
    ) -> _ToolConfigDict:
        """Convert tool_choice to tool_config.

        Args:
            tool_choice: The tool choice setting
            tools: The tools being bound (used for validation in future)
        """
        _ = tools  # Reserved for future validation
        if tool_choice is True:
            # Must use at least one tool
            return {"function_calling_config": {"mode": "ANY"}}
        if tool_choice is False:
            # Cannot use tools
            return {"function_calling_config": {"mode": "NONE"}}
        if isinstance(tool_choice, str):
            # Must use specific tool
            return {
                "function_calling_config": {
                    "mode": "ANY",
                    "allowed_function_names": [tool_choice],
                }
            }
        return {"function_calling_config": {"mode": "AUTO"}}

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel], Type],
        *,
        include_raw: bool = False,
        method: Optional[Literal["json_mode"]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class.
            include_raw: If True, return both raw and parsed output.
            method: If "json_mode", use controlled generation instead of function calling.

        Returns:
            A Runnable that outputs structured data.
        """
        _ = kwargs.pop("strict", None)
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")

        parser: OutputParserLike

        if method == "json_mode":
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                schema_json = schema.model_json_schema()
                parser = PydanticOutputParser(pydantic_object=schema)
            else:
                if is_typeddict(schema):
                    from langchain_core.utils.function_calling import convert_to_json_schema

                    schema_json = convert_to_json_schema(schema)
                elif isinstance(schema, dict):
                    schema_json = schema
                else:
                    raise ValueError(f"Unsupported schema type {type(schema)}")
                parser = JsonOutputParser()

            llm = self.bind(
                response_mime_type="application/json",
                response_schema=schema_json,
            )
        else:
            # Function calling mode
            tool_name = self._get_tool_name(schema)
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                parser = PydanticToolsParser(tools=[schema], first_tool_only=True)
            elif is_typeddict(schema) or isinstance(schema, dict):
                parser = JsonOutputKeyToolsParser(key_name=tool_name, first_tool_only=True)
            else:
                raise ValueError(f"Unsupported schema type {type(schema)}")

            llm = self.bind_tools([schema], tool_choice=tool_name)

        if include_raw:
            parser_with_fallback = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | parser, parsing_error=lambda _: None
            ).with_fallbacks(
                [RunnablePassthrough.assign(parsed=lambda _: None)],
                exception_key="parsing_error",
            )
            return {"raw": llm} | parser_with_fallback
        else:
            return llm | parser

    def _get_tool_name(self, schema: Union[Dict, Type]) -> str:
        """Extract tool name from schema."""
        if isinstance(schema, dict):
            if "name" in schema:
                return schema["name"]
            if "function" in schema and "name" in schema["function"]:
                return schema["function"]["name"]
            if "title" in schema:
                return schema["title"]
            return "schema"
        if isinstance(schema, type):
            return schema.__name__
        return "schema"

    @property
    def _llm_type(self) -> str:
        return "google-genai"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
            "vertexai": self.vertexai,
            "thinking_level": self.thinking_level,
        }

    def _get_client(self) -> Client:
        """Get or create the google-genai client."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                msg = "The 'google-genai' package is required. Install it with: pip install google-genai"
                raise ImportError(msg) from e

            client_kwargs: dict[str, Any] = {}

            if self.vertexai:
                client_kwargs["vertexai"] = True
                if self.project:
                    client_kwargs["project"] = self.project
                if self.location:
                    client_kwargs["location"] = self.location
                if self.credentials_path:
                    from google.oauth2 import service_account

                    try:
                        credentials = service_account.Credentials.from_service_account_file(
                            self.credentials_path
                        )
                    except FileNotFoundError as e:
                        msg = f"Service account file not found: {self.credentials_path}"
                        raise ValueError(msg) from e
                    except Exception as e:
                        msg = f"Failed to load service account credentials: {e}"
                        raise ValueError(msg) from e
                    client_kwargs["credentials"] = credentials
            elif self.api_key:
                client_kwargs["api_key"] = self.api_key.get_secret_value()

            self._client = genai.Client(**client_kwargs)

        return self._client

    def _convert_tools_to_genai(self, tools: list[Any]) -> list[Any]:
        """Convert LangChain tools to google-genai Tool format."""
        from google.genai import types

        function_declarations = []
        for tool in tools:
            try:
                openai_tool = convert_to_openai_tool(tool)
                func = openai_tool["function"]
                genai_func = types.FunctionDeclaration(
                    name=func["name"],
                    description=func.get("description", ""),
                    parameters_json_schema=func.get("parameters", {}),
                )
                function_declarations.append(genai_func)
            except Exception:
                if isinstance(tool, dict):
                    genai_func = types.FunctionDeclaration(
                        name=tool.get("name", ""),
                        description=tool.get("description", ""),
                        parameters_json_schema=tool.get("parameters", {}),
                    )
                    function_declarations.append(genai_func)
                else:
                    function_declarations.append(tool)

        if function_declarations:
            return [types.Tool(function_declarations=function_declarations)]
        return []

    def _build_config(self) -> dict[str, Any]:
        """Build the GenerateContentConfig dictionary."""
        from google.genai import types

        config_kwargs: dict[str, Any] = {}

        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            config_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            config_kwargs["top_k"] = self.top_k
        if self.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = self.max_output_tokens
        if self.stop_sequences:
            config_kwargs["stop_sequences"] = self.stop_sequences
        if self.candidate_count and self.candidate_count > 1:
            config_kwargs["candidate_count"] = self.candidate_count
        if self.seed is not None:
            config_kwargs["seed"] = self.seed
        if self.response_mime_type:
            config_kwargs["response_mime_type"] = self.response_mime_type
        if self.response_schema:
            config_kwargs["response_schema"] = self.response_schema

        # Safety settings
        if self.safety_settings:
            config_kwargs["safety_settings"] = [
                types.SafetySetting(
                    category=s.get("category"),
                    threshold=s.get("threshold"),
                )
                for s in self.safety_settings
            ]

        # Thinking config for Gemini 3
        if self.thinking_level and self.thinking_level != "off":
            thinking_level_map = {
                "low": types.ThinkingLevel.LOW,
                "medium": types.ThinkingLevel.MEDIUM,
                "high": types.ThinkingLevel.HIGH,
            }
            if self.thinking_level in thinking_level_map:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=thinking_level_map[self.thinking_level]
                )
        elif self.thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )

        # Add tools if bound
        if self.tools:
            genai_tools = self._convert_tools_to_genai(self.tools)
            config_kwargs["tools"] = genai_tools
            config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True
            )

            # Add tool config if specified
            if self.tool_config and "function_calling_config" in self.tool_config:
                fc_config = self.tool_config["function_calling_config"]
                mode = fc_config.get("mode", "AUTO")
                mode_enum = getattr(types.FunctionCallingConfigMode, mode, None)
                if mode_enum:
                    tool_config_kwargs: dict[str, Any] = {"mode": mode_enum}
                    if "allowed_function_names" in fc_config:
                        tool_config_kwargs["allowed_function_names"] = fc_config[
                            "allowed_function_names"
                        ]
                    config_kwargs["tool_config"] = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(**tool_config_kwargs)
                    )

        return config_kwargs

    def _convert_part_to_genai(self, part: Union[str, Dict]) -> Optional[Dict]:
        """Convert a single message part to google-genai format."""
        if isinstance(part, str):
            return {"text": part}

        if not isinstance(part, dict):
            return {"text": str(part)}

        part_type = str(part.get("type", "text"))

        if part_type == "text":
            return {"text": part.get("text", "")}

        if part_type == "image_url":
            image_url = part.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else image_url

            # Handle base64 data URLs
            if url.startswith("data:"):
                # Parse data URL: data:image/png;base64,<data>
                try:
                    header, data = url.split(",", 1)
                    mime_type = header.split(":")[1].split(";")[0]
                    return {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": data,
                        }
                    }
                except (ValueError, IndexError):
                    logger.warning(f"Failed to parse data URL: {url[:50]}...")
                    return None

            # Handle http/https URLs
            if url.startswith(("http://", "https://")):
                return {"file_data": {"file_uri": url}}

            # Handle GCS URLs
            if url.startswith("gs://"):
                mime_type = image_url.get("mime_type", "image/jpeg")
                return {"file_data": {"file_uri": url, "mime_type": mime_type}}

            return None

        if part_type == "media":
            mime_type = part.get("mime_type", "application/octet-stream")
            if "data" in part:
                return {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": part["data"],
                    }
                }
            if "file_uri" in part:
                return {"file_data": {"file_uri": part["file_uri"], "mime_type": mime_type}}

        return {"text": str(part)}

    def _convert_messages_to_contents(
        self, messages: list[BaseMessage]
    ) -> tuple[list[dict], str | None]:
        """Convert langchain messages to google-genai contents format."""
        contents: list[dict] = []
        system_instruction = None
        prev_ai_message: Optional[AIMessage] = None

        for message in messages:
            if isinstance(message, SystemMessage):
                prev_ai_message = None
                content = message.content
                if isinstance(content, str):
                    system_instruction = content
                elif isinstance(content, list):
                    texts = [
                        p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content
                    ]
                    system_instruction = "\n".join(texts)

            elif isinstance(message, HumanMessage):
                prev_ai_message = None
                parts = self._convert_content_to_parts(message.content)
                if parts:
                    # Merge with previous user message if exists
                    if contents and _get_content_role(contents[-1]) == "user":
                        _extend_content_parts(contents[-1], parts)
                    else:
                        contents.append({"role": "user", "parts": parts})

            elif isinstance(message, AIMessage):
                from google.genai import types

                prev_ai_message = message
                parts = []

                # Add text/multimodal content
                if message.content:
                    parts.extend(self._convert_content_to_parts(message.content))

                # Add function calls with thought signatures if present
                if hasattr(message, "tool_calls") and message.tool_calls:

                    thought_sigs = message.additional_kwargs.get(
                        _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY, {}
                    )
                    for tc in message.tool_calls:
                        # Create proper FunctionCall object
                        function_call = types.FunctionCall(
                            name=tc["name"],
                            args=tc["args"],
                        )

                        # Get thought_signature if present
                        thought_signature: bytes | None = None
                        tc_id = tc.get("id", "")
                        if tc_id and tc_id in thought_sigs:
                            sig = thought_sigs[tc_id]
                            # Decode from base64 string to bytes
                            if isinstance(sig, str):
                                thought_signature = _base64_to_bytes(sig)
                            elif isinstance(sig, bytes):
                                thought_signature = sig

                        # Create Part with function_call and optional thought_signature
                        part_kwargs: Dict[str, Any] = {"function_call": function_call}
                        if thought_signature:
                            part_kwargs["thought_signature"] = thought_signature

                        parts.append(types.Part(**part_kwargs))

                if parts:
                    # Merge with previous model message if exists
                    if contents and _get_content_role(contents[-1]) == "model":
                        _extend_content_parts(contents[-1], parts)
                    else:
                        contents.append(types.Content(role="model", parts=parts))

            elif isinstance(message, FunctionMessage):
                # Legacy FunctionMessage support
                prev_ai_message = None
                part = {
                    "function_response": {
                        "name": message.name or "unknown",
                        "response": {"content": message.content},
                    }
                }
                # Merge with previous function response if exists
                if contents and _get_content_role(contents[-1]) == "user":
                    # Check if last part is a function_response
                    last_parts = _get_content_parts(contents[-1])
                    if (
                        last_parts
                        and isinstance(last_parts[-1], dict)
                        and "function_response" in last_parts[-1]
                    ):
                        _extend_content_parts(contents[-1], [part])
                    else:
                        contents.append({"role": "user", "parts": [part]})
                else:
                    contents.append({"role": "user", "parts": [part]})

            elif isinstance(message, ToolMessage):
                # Resolve tool name from previous AI message if not provided
                name = message.name
                if name is None and prev_ai_message:
                    tool_call_id = message.tool_call_id
                    for tc in prev_ai_message.tool_calls:
                        if tc.get("id") == tool_call_id:
                            name = tc["name"]
                            break

                # Parse content
                content = message.content
                if isinstance(content, str):
                    try:
                        response_content = json.loads(content)
                    except json.JSONDecodeError:
                        response_content = {"result": content}
                elif isinstance(content, dict):
                    response_content = content
                else:
                    response_content = {"result": str(content)}

                part = {
                    "function_response": {
                        "name": name or "unknown",
                        "response": response_content,
                    }
                }

                # Merge with previous function response
                if contents and _get_content_role(contents[-1]) == "user":
                    last_parts = _get_content_parts(contents[-1])
                    if (
                        last_parts
                        and isinstance(last_parts[-1], dict)
                        and "function_response" in last_parts[-1]
                    ):
                        _extend_content_parts(contents[-1], [part])
                    else:
                        contents.append({"role": "user", "parts": [part]})
                else:
                    contents.append({"role": "user", "parts": [part]})

            else:
                # Handle other message types as user messages
                prev_ai_message = None
                parts = self._convert_content_to_parts(message.content)
                if parts:
                    contents.append({"role": "user", "parts": parts})

        return contents, system_instruction

    def _convert_content_to_parts(self, content: Any) -> list[dict]:
        """Convert message content to parts list."""
        if isinstance(content, str):
            # Try to parse as literal (for multimodal agent strings)
            try:
                parsed = ast.literal_eval(content)
                if isinstance(parsed, list):
                    content = parsed
            except (SyntaxError, ValueError):
                pass

        if isinstance(content, str):
            return [{"text": content}]

        if isinstance(content, int):
            return [{"text": str(content)}]

        if isinstance(content, list):
            parts = []
            for item in content:
                part = self._convert_part_to_genai(item)
                if part:
                    parts.append(part)
            return parts

        return [{"text": str(content)}]

    def _parse_response(self, response: GenerateContentResponse) -> ChatResult:
        """Parse the google-genai response into a ChatResult."""
        generations = []

        for candidate in response.candidates:
            content: str | list[str] | None = None
            tool_calls = []
            invalid_tool_calls_list = []
            additional_kwargs: dict[str, Any] = {}

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # Extract text content
                    try:
                        text = part.text if hasattr(part, "text") else None
                    except AttributeError:
                        text = None

                    if text:
                        if content is None:
                            content = text
                        elif isinstance(content, str):
                            content = [content, text]
                        elif isinstance(content, list):
                            content.append(text)

                    # Extract function calls
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        func_name = fc.name if hasattr(fc, "name") else ""
                        func_args = fc.args if hasattr(fc, "args") else {}
                        # Use the function call's ID if available, otherwise generate one
                        tool_call_id = getattr(fc, "id", None) or str(uuid.uuid4())
                        # Capture thought_signature from Part (not FunctionCall)
                        thought_sig = getattr(part, "thought_signature", None)

                        if hasattr(func_args, "items"):
                            args_dict = dict(func_args.items())
                        elif isinstance(func_args, dict):
                            args_dict = func_args
                        else:
                            args_dict = {}

                        function_call = {
                            "name": func_name,
                            "arguments": json.dumps(args_dict),
                        }
                        additional_kwargs["function_call"] = function_call

                        # Store thought_signature keyed by tool call ID (base64 encoded)
                        if thought_sig:
                            if _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY not in additional_kwargs:
                                additional_kwargs[_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY] = {}
                            # Encode bytes to base64 string for JSON-safe storage
                            if isinstance(thought_sig, bytes):
                                thought_sig = _bytes_to_base64(thought_sig)
                            additional_kwargs[_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY][
                                tool_call_id
                            ] = thought_sig

                        try:
                            tool_calls_dicts = parse_tool_calls(
                                [{"function": function_call}],
                                return_id=False,
                            )
                            tool_calls.extend(
                                [
                                    create_tool_call(
                                        name=tc["name"],
                                        args=tc["args"],
                                        id=tool_call_id,
                                    )
                                    for tc in tool_calls_dicts
                                ]
                            )
                        except Exception as e:
                            invalid_tool_calls_list.append(
                                invalid_tool_call(
                                    name=function_call.get("name"),
                                    args=function_call.get("arguments"),
                                    id=tool_call_id,
                                    error=str(e),
                                )
                            )

            if content is None:
                content = ""

            generation_info: dict[str, Any] = {}
            if candidate.finish_reason:
                generation_info["finish_reason"] = str(candidate.finish_reason)
            if candidate.safety_ratings:
                generation_info["safety_ratings"] = [
                    {"category": str(r.category), "probability": str(r.probability)}
                    for r in candidate.safety_ratings
                ]

            message = AIMessage(
                content=content,
                tool_calls=tool_calls,
                additional_kwargs=additional_kwargs,
                invalid_tool_calls=invalid_tool_calls_list,
            )
            generations.append(ChatGeneration(message=message, generation_info=generation_info))

        llm_output: dict[str, Any] = {"model": self.model}
        if response.usage_metadata:
            llm_output["token_usage"] = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }
            if response.usage_metadata.thoughts_token_count:
                llm_output["token_usage"][
                    "thoughts_tokens"
                ] = response.usage_metadata.thoughts_token_count

        return ChatResult(generations=generations, llm_output=llm_output)

    def _parse_streaming_response(
        self, part: Any, seen_function_calls: Optional[set[str]] = None
    ) -> Optional[AIMessageChunk]:
        """Parse a streaming response part into an AIMessageChunk.

        Args:
            part: The response part to parse
            seen_function_calls: Optional set to track already-emitted function calls
                                 to avoid duplicates that cause name concatenation
        """
        content = ""
        tool_call_chunks = []
        additional_kwargs: Dict[str, Any] = {}

        # Extract text
        if hasattr(part, "text") and part.text:
            content = part.text

        # Extract function calls for streaming
        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            func_name = fc.name if hasattr(fc, "name") else ""
            func_args = fc.args if hasattr(fc, "args") else {}
            # Use the function call's ID if available for deduplication
            fc_id = getattr(fc, "id", None)
            # Capture thought_signature from Part (not FunctionCall)
            thought_sig = getattr(part, "thought_signature", None)

            if hasattr(func_args, "items"):
                args_dict = dict(func_args.items())
            elif isinstance(func_args, dict):
                args_dict = func_args
            else:
                args_dict = {}

            # Use function call ID if available, otherwise use name+args hash for deduplication
            if fc_id:
                fc_key = fc_id
            else:
                fc_key = f"{func_name}:{json.dumps(args_dict, sort_keys=True)}"

            # Only emit each unique function call once
            if seen_function_calls is None or fc_key not in seen_function_calls:
                if seen_function_calls is not None:
                    seen_function_calls.add(fc_key)

                # Generate a tool call ID if not provided by API
                tool_call_id = fc_id or str(uuid.uuid4())

                tool_call_chunks.append(
                    tool_call_chunk(
                        name=func_name,
                        args=json.dumps(args_dict),
                        id=tool_call_id,
                        index=0,
                    )
                )

                # Store thought_signature in additional_kwargs (base64 encoded)
                if thought_sig:
                    # Encode bytes to base64 string for JSON-safe storage
                    if isinstance(thought_sig, bytes):
                        thought_sig = _bytes_to_base64(thought_sig)
                    additional_kwargs[_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY] = {
                        tool_call_id: thought_sig
                    }

        if content or tool_call_chunks:
            return AIMessageChunk(
                content=content,
                tool_call_chunks=tool_call_chunks,
                additional_kwargs=additional_kwargs,
            )
        return None

    def _with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        import time

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                # Retry on rate limit and server errors
                if any(
                    x in error_str
                    for x in ["rate limit", "quota", "503", "500", "unavailable", "overloaded"]
                ):
                    wait_time = min(2**attempt, 60)  # Exponential backoff, max 60s
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} after {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    raise
        raise last_exception  # type: ignore

    async def _with_retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        import asyncio

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                if any(
                    x in error_str
                    for x in ["rate limit", "quota", "503", "500", "unavailable", "overloaded"]
                ):
                    wait_time = min(2**attempt, 60)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} after {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise
        raise last_exception  # type: ignore

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the model."""
        client = self._get_client()
        contents, system_instruction = self._convert_messages_to_contents(messages)

        config_kwargs = self._build_config()
        if stop:
            config_kwargs["stop_sequences"] = stop
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        from google.genai import types

        config = types.GenerateContentConfig(**config_kwargs)

        def _call():
            return client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        response = self._with_retry(_call)
        return self._parse_response(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate a response from the model."""
        client = self._get_client()
        contents, system_instruction = self._convert_messages_to_contents(messages)

        config_kwargs = self._build_config()
        if stop:
            config_kwargs["stop_sequences"] = stop
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        from google.genai import types

        config = types.GenerateContentConfig(**config_kwargs)

        async def _call():
            return await client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        response = await self._with_retry_async(_call)
        return self._parse_response(response)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream responses from the model."""
        # CRITICAL: When tools are bound, use non-streaming to avoid
        # tool_call_chunk name concatenation and thought_signature loss
        if self.tools:
            result = self._generate(messages, stop, run_manager, **kwargs)
            for generation in result.generations:
                msg = generation.message
                if isinstance(msg, AIMessage):
                    chunk_msg = AIMessageChunk(
                        content=msg.content,
                        additional_kwargs=msg.additional_kwargs,
                        tool_calls=msg.tool_calls,
                        invalid_tool_calls=msg.invalid_tool_calls,
                    )
                    yield ChatGenerationChunk(
                        message=chunk_msg,
                        generation_info=generation.generation_info,
                    )
            return

        client = self._get_client()
        contents, system_instruction = self._convert_messages_to_contents(messages)

        config_kwargs = self._build_config()
        if stop:
            config_kwargs["stop_sequences"] = stop
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        from google.genai import types

        config = types.GenerateContentConfig(**config_kwargs)

        # Track seen function calls to avoid duplicate emission
        seen_function_calls: set[str] = set()

        for chunk in client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.candidates:
                for candidate in chunk.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            msg_chunk = self._parse_streaming_response(part, seen_function_calls)
                            if msg_chunk:
                                yield ChatGenerationChunk(message=msg_chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream responses from the model."""
        # CRITICAL: When tools are bound, use non-streaming to avoid
        # tool_call_chunk name concatenation and thought_signature loss
        if self.tools:
            result = await self._agenerate(messages, stop, run_manager, **kwargs)
            for generation in result.generations:
                msg = generation.message
                if isinstance(msg, AIMessage):
                    chunk_msg = AIMessageChunk(
                        content=msg.content,
                        additional_kwargs=msg.additional_kwargs,
                        tool_calls=msg.tool_calls,
                        invalid_tool_calls=msg.invalid_tool_calls,
                    )
                    yield ChatGenerationChunk(
                        message=chunk_msg,
                        generation_info=generation.generation_info,
                    )
            return

        client = self._get_client()
        contents, system_instruction = self._convert_messages_to_contents(messages)

        config_kwargs = self._build_config()
        if stop:
            config_kwargs["stop_sequences"] = stop
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        from google.genai import types

        config = types.GenerateContentConfig(**config_kwargs)

        # Track seen function calls to avoid duplicate emission
        seen_function_calls: set[str] = set()

        async for chunk in await client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.candidates:
                for candidate in chunk.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            msg_chunk = self._parse_streaming_response(part, seen_function_calls)
                            if msg_chunk:
                                if run_manager and msg_chunk.content:
                                    await run_manager.on_llm_new_token(msg_chunk.content)
                                yield ChatGenerationChunk(message=msg_chunk)
