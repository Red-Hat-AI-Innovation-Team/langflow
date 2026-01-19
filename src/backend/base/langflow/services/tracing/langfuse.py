from __future__ import annotations

import os
from collections import OrderedDict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from lfx.log.logger import logger
from typing_extensions import override

from langflow.serialization.serialization import serialize
from langflow.services.tracing.base import BaseTracer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from langchain.callbacks.base import BaseCallbackHandler
    from lfx.graph.vertex.base import Vertex

    from langflow.services.tracing.schema import Log


class LangFuseTracer(BaseTracer):
    flow_id: str

    def __init__(
        self,
        trace_name: str,
        trace_type: str,
        project_name: str,
        trace_id: UUID,
        user_id: str | None = None,
        session_id: str | None = None,
        parent_observation_id: str | None = None,
    ) -> None:
        self.project_name = project_name
        self.trace_name = trace_name
        self.trace_type = trace_type
        self.trace_id = trace_id
        self.user_id = user_id
        self.session_id = session_id
        self.parent_observation_id = parent_observation_id
        self.flow_id = trace_name.split(" - ")[-1]
        self.spans: dict = OrderedDict()  # spans that are not ended

        config = self._get_config()
        self._ready: bool = self.setup_langfuse(config) if config else False

    @property
    def ready(self):
        return self._ready

    def setup_langfuse(self, config) -> bool:
        try:
            from langfuse import Langfuse

            self._client = Langfuse(**config)
            try:
                from langfuse.api.core.request_options import RequestOptions

                self._client.client.health.health(request_options=RequestOptions(timeout_in_seconds=1))
            except Exception as e:  # noqa: BLE001
                logger.debug(f"can not connect to Langfuse: {e}")
                return False

            # CRITICAL: Use hex format (no dashes) for trace_id to match LangFuse's internal format
            # LangFuse stores trace IDs as 32-char hex strings, not UUID format with dashes
            # The experiment creates traces with hex IDs like "744bc58d0e442e8aefe5c30d41008bb3"
            # If we use UUID format "744bc58d-0e44-2e8a-efe5-c30d41008bb3", LangFuse won't find the trace
            if hasattr(self.trace_id, 'hex'):
                trace_id_str = self.trace_id.hex  # UUID object -> 32 char hex string
            else:
                trace_id_str = str(self.trace_id).replace('-', '')  # String with dashes -> hex string

            # If parent_observation_id is provided, we're nesting under an existing experiment trace
            # In this case, we should NOT create a new trace - just verify parent exists and create spans
            if self.parent_observation_id:
                logger.info(f"[LANGFLOW-TRACING] Parent observation mode - trace_id: {trace_id_str}")
                logger.info(f"[LANGFLOW-TRACING] Attempting to nest under parent: {self.parent_observation_id}")

                # Verify parent observation exists by fetching it
                parent_exists = self._verify_parent_observation_exists(trace_id_str)
                if parent_exists:
                    logger.info(f"[LANGFLOW-TRACING] SUCCESS: Parent observation verified in LangFuse")
                else:
                    logger.warning(f"[LANGFLOW-TRACING] WARNING: Could not verify parent observation exists")

                # Get reference to existing trace (don't create new one with different metadata)
                # Using trace() with just the ID connects to existing trace
                self.trace = self._client.trace(id=trace_id_str)

                # Create root span as child of the parent observation
                # This is the key: parent_observation_id links this span to the experiment's observation
                self._root_span = self.trace.span(
                    name=f"Langflow: {self.flow_id}",
                    parent_observation_id=self.parent_observation_id,
                )
                logger.info(f"[LANGFLOW-TRACING] Created root span '{self._root_span.id}' under parent observation")
            else:
                # Original behavior - create our own trace
                logger.info(f"[LANGFLOW-TRACING] Standalone mode - creating new trace: {trace_id_str}")
                self.trace = self._client.trace(
                    id=trace_id_str,
                    name=self.flow_id,
                    user_id=self.user_id,
                    session_id=self.session_id,
                )
                self._root_span = None

        except ImportError:
            logger.exception("Could not import langfuse. Please install it with `pip install langfuse`.")
            return False

        except Exception as e:  # noqa: BLE001
            logger.debug(f"Error setting up LangFuse tracer: {e}")
            return False

        return True

    def _verify_parent_observation_exists(self, trace_id: str) -> bool:
        """Verify that the parent observation exists in LangFuse.

        This helps debug timing issues where Langflow tries to nest under
        an observation that hasn't been flushed to LangFuse yet.
        """
        try:
            # Try to fetch the trace to verify it exists
            trace_data = self._client.fetch_trace(trace_id)
            if trace_data:
                logger.info(f"[LANGFLOW-TRACING] Trace exists with {len(trace_data.observations or [])} observations")
                # Check if parent_observation_id is in the observations
                if trace_data.observations:
                    obs_ids = [obs.id for obs in trace_data.observations]
                    if self.parent_observation_id in obs_ids:
                        logger.info(f"[LANGFLOW-TRACING] Parent observation {self.parent_observation_id} found in trace")
                        return True
                    else:
                        logger.warning(f"[LANGFLOW-TRACING] Parent observation NOT in trace. Available: {obs_ids[:5]}")
                        return False
                else:
                    logger.warning(f"[LANGFLOW-TRACING] Trace exists but has no observations yet")
                    return False
            return False
        except Exception as e:
            logger.warning(f"[LANGFLOW-TRACING] Could not verify parent observation: {e}")
            return False

    @override
    def add_trace(
        self,
        trace_id: str,  # actualy component id
        trace_name: str,
        trace_type: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        vertex: Vertex | None = None,
    ) -> None:
        start_time = datetime.now(tz=timezone.utc)
        if not self._ready:
            return

        metadata_: dict = {"from_langflow_component": True, "component_id": trace_id}
        metadata_ |= {"trace_type": trace_type} if trace_type else {}
        metadata_ |= metadata or {}

        name = trace_name.removesuffix(f" ({trace_id})")
        content_span = {
            "name": name,
            "input": inputs,
            "metadata": metadata_,
            "start_time": start_time,
        }

        # If we have a root span (from parent_observation_id), create spans under it
        # Otherwise create spans directly under the trace
        parent = self._root_span if self._root_span else self.trace
        span = parent.span(**serialize(content_span))

        self.spans[trace_id] = span

    @override
    def end_trace(
        self,
        trace_id: str,
        trace_name: str,
        outputs: dict[str, Any] | None = None,
        error: Exception | None = None,
        logs: Sequence[Log | dict] = (),
    ) -> None:
        end_time = datetime.now(tz=timezone.utc)
        if not self._ready:
            return

        span = self.spans.pop(trace_id, None)
        if span:
            output: dict = {}
            output |= outputs or {}
            output |= {"error": str(error)} if error else {}
            output |= {"logs": list(logs)} if logs else {}
            content = serialize({"output": output, "end_time": end_time})
            span.update(**content)

    @override
    def end(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        error: Exception | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._ready:
            return
        content_update = {
            "input": inputs,
            "output": outputs,
            "metadata": metadata,
        }
        # End the root span if it exists
        if self._root_span:
            self._root_span.update(**serialize(content_update))
            self._root_span.end()
        self.trace.update(**serialize(content_update))

    def get_langchain_callback(self) -> BaseCallbackHandler | None:
        if not self._ready:
            return None

        # get callback from parent span, preferring root_span if it exists
        if len(self.spans) > 0:
            stateful_client = self.spans[next(reversed(self.spans))]
        elif self._root_span:
            stateful_client = self._root_span
        else:
            stateful_client = self.trace
        return stateful_client.get_langchain_handler()

    @staticmethod
    def _get_config() -> dict:
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", None)
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", None)
        host = os.getenv("LANGFUSE_HOST", None)
        if secret_key and public_key and host:
            return {"secret_key": secret_key, "public_key": public_key, "host": host}
        return {}
