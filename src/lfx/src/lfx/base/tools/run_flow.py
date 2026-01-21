from collections import Counter
from datetime import datetime
from types import MethodType  # near the imports
from typing import TYPE_CHECKING, Any

from langflow.helpers.flow import get_flow_by_id_or_name
from langflow.processing.process import process_tweaks

from lfx.base.tools.constants import TOOL_OUTPUT_NAME
from lfx.custom.custom_component.component import Component, get_component_toolkit
from lfx.field_typing import Tool
from lfx.graph.graph.base import Graph
from lfx.graph.vertex.base import Vertex

# TODO: switch to lfx
from lfx.helpers import get_flow_inputs, run_flow
from lfx.inputs.inputs import BoolInput, DropdownInput, InputTypes, MessageTextInput, StrInput
from lfx.log.logger import logger
from lfx.schema.data import Data
from lfx.schema.dotdict import dotdict
from lfx.services.cache.utils import CacheMiss
from lfx.services.deps import get_shared_component_cache_service
from lfx.template.field.base import Output

if TYPE_CHECKING:
    from collections.abc import Callable

    from lfx.base.tools.component_tool import ComponentToolkit


class RunFlowBaseComponent(Component):
    def __init__(self, *args, **kwargs):
        self._flow_output_methods: set[str] = set()
        super().__init__(*args, **kwargs)
        self.add_tool_output = True
        ################################################################
        # cache the selected flow's graph in the shared component cache
        # if cache_flow is enabled.
        ################################################################
        self._shared_component_cache = get_shared_component_cache_service()
        # add all the flow cache related methods to the dispatcher.
        # these are used internally among the cache related methods.
        # the _flow_cache_call method is meant to be user-facing
        # for cache operations as it handles validation.
        self._cache_flow_dispatcher: dict[str, Callable[..., Any]] = {
            "get": self._get_cached_flow,
            "set": self._set_cached_flow,
            "delete": self._delete_cached_flow,
            "_build_key": self._build_flow_cache_key,
            "_build_graph": self._build_graph_from_dict,
        }
        # save the run's outputs to avoid re-executing
        # the flow if it has multiple outputs.
        self._last_run_outputs: list[Data] | None = None
        # save the updated_at of the user's selected flow
        self._cached_flow_updated_at: str | None = None

    _base_inputs: list[InputTypes] = [
        DropdownInput(
            name="flow_name_selected",
            display_name="Flow Name",
            info="The name of the flow to run.",
            options=[],
            options_metadata=[],
            real_time_refresh=True,
            refresh_button=True,
            value=None,
        ),
        StrInput(
            name="flow_id_selected",
            display_name="Flow ID",
            info="The ID of the flow to run.",
            value=None,
            show=False,
            override_skip=True,  # persist to runtime
        ),
        MessageTextInput(
            name="session_id",
            display_name="Session ID",
            info="The session ID to run the flow in.",
            advanced=True,
        ),
        # bool dropdown to select if the flow should be cached
        # Note: the user's selected flow is automatically updated when
        # when the flow_name_selected dropdown is refreshed.
        # TODO: find a more explicit way to update the cached flow.
        BoolInput(
            name="cache_flow",
            display_name="Cache Flow",
            info="Whether to cache the selected flow.",
            value=False,
            advanced=True,
        ),
    ]
    _base_outputs: list[Output] = []
    default_keys = [
        "code",
        "_type",
        "flow_name_selected",
        "flow_id_selected",
        "session_id",
        "cache_flow",
    ]
    FLOW_INPUTS: list[dotdict] = []
    flow_tweak_data: dict = {}
    IOPUT_SEP = "~"  # separator for joining a vertex id and input/output name to form a unique input/output name

    ################################################################
    # set and register the selected flow's output methods
    ################################################################
    def map_outputs(self) -> None:  # Note: overrides the base map_outputs method
        super().map_outputs()
        self._ensure_flow_output_methods()

    def _ensure_flow_output_methods(self) -> None:
        self._clear_dynamic_flow_output_methods()
        for output in self._outputs_map.values():
            if (
                not output
                or not output.name
                or output.name == TOOL_OUTPUT_NAME
                or self.IOPUT_SEP not in output.name
            ):
                continue
            vertex_id, output_name = output.name.split(self.IOPUT_SEP, 1)
            output.method = self._register_flow_output_method(
                vertex_id=vertex_id,
                output_name=output_name,
            )

    ################################################################
    # Flow retrieval
    ################################################################
    async def get_flow(
        self, flow_name_selected: str | None = None, flow_id_selected: str | None = None
    ) -> Data:
        """Get a flow's data by name or id."""
        flow = await get_flow_by_id_or_name(
            user_id=self.user_id,
            flow_id=flow_id_selected,
            flow_name=flow_name_selected,
        )
        return flow or Data(data={})

    async def get_graph(
        self,
        flow_name_selected: str | None = None,
        flow_id_selected: str | None = None,
        updated_at: str | None = None,
    ) -> Graph | None:
        """Get a flow's graph by name or id."""
        if not (flow_name_selected or flow_id_selected):
            msg = "Flow name or id is required"
            raise ValueError(msg)
        if flow_id_selected and (flow := self._flow_cache_call("get", flow_id=flow_id_selected)):
            if self._is_cached_flow_up_to_date(flow, updated_at):
                return flow
            self._flow_cache_call("delete", flow_id=flow_id_selected)  # stale, delete it

        # TODO: use flow id only
        flow = await self.get_flow(
            flow_name_selected=flow_name_selected, flow_id_selected=flow_id_selected
        )
        if not flow:
            msg = "Flow not found"
            raise ValueError(msg)

        graph = Graph.from_payload(
            payload=flow.data.get("data", {}),
            flow_id=flow_id_selected,
            flow_name=flow_name_selected,
        )
        graph.description = flow.data.get("description", None)
        graph.updated_at = flow.data.get("updated_at", None)

        self._flow_cache_call("set", flow=graph)

        return graph

    ################################################################
    # Flow inputs/config
    ################################################################
    def get_new_fields_from_graph(self, graph: Graph) -> list[dotdict]:
        inputs = get_flow_inputs(graph)
        return self.get_new_fields(inputs)

    def update_build_config_from_graph(self, build_config: dotdict, graph: Graph):
        try:
            new_fields = self.get_new_fields_from_graph(graph)
            # Ensure all fields can accept Message type connections
            new_fields = self.update_input_types(new_fields)
            keep_fields: set[str] = set(
                [new_field["name"] for new_field in new_fields] + self.default_keys
            )
            self.delete_fields(
                build_config, [key for key in build_config if key not in keep_fields]
            )
            build_config.update((field["name"], field) for field in new_fields)
        except Exception as e:
            msg = "Error updating build config from graph"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def get_new_fields(self, inputs_vertex: list[Vertex]) -> list[dotdict]:
        new_fields: list[dotdict] = []
        vdisp_cts = Counter(v.display_name for v in inputs_vertex)

        for vertex in inputs_vertex:
            field_template = vertex.data.get("node", {}).get("template", {})
            field_order = vertex.data.get("node", {}).get("field_order", [])
            if not (field_order and field_template):
                continue
            new_vertex_inputs = [
                dotdict(
                    {
                        **field_template[input_name],
                        "name": self._get_ioput_name(vertex.id, input_name),
                        "display_name": (
                            f"{field_template[input_name]['display_name']} ({vertex.display_name})"
                            if vdisp_cts[vertex.display_name] == 1
                            else (
                                f"{field_template[input_name]['display_name']}"
                                f"({vertex.display_name}-{vertex.id.split('-')[-1]})"
                            )
                        ),
                        # TODO: make this more robust?
                        "tool_mode": not (field_template[input_name].get("advanced", False)),
                    }
                )
                for input_name in field_order
                if input_name in field_template
            ]
            new_fields += new_vertex_inputs
        return new_fields

    def add_new_fields(self, build_config: dotdict, new_fields: list[dotdict]) -> dotdict:
        """Add new fields to the build_config."""
        for field in new_fields:
            build_config[field["name"]] = field
        return build_config

    def delete_fields(self, build_config: dotdict, fields: dict | list[str]) -> None:
        """Delete specified fields from build_config.

        Args:
            build_config: The build_config to delete the fields from.
            fields: The fields to delete from the build_config.
        """
        if isinstance(fields, dict):
            fields = list(fields.keys())
        for field in fields:
            build_config.pop(field, None)

    async def get_required_data(self) -> tuple[str, list[dotdict]] | None:
        """Retrieve flow description and tool-mode input fields for the selected flow.

        Fetches the graph for the given flow, extracts its input fields, and filters
        for only those inputs that are eligible for tool mode (non-advanced fields).

        Args:
            flow_name_selected: The name of the flow to retrieve data for. If None,
                returns None.

        Returns:
            A tuple of (flow_description, tool_mode_fields) where:
                - flow_description (str): The human-readable description of the flow
                - tool_mode_fields (list[dotdict]): Input fields marked for tool mode
            Returns None if the flow cannot be found or loaded.
        """
        graph = await self.get_graph(
            self.flow_name_selected, self.flow_id_selected, self._cached_flow_updated_at
        )
        formatted_outputs = self._format_flow_outputs(graph)
        self._sync_flow_outputs(formatted_outputs)
        new_fields = self.get_new_fields_from_graph(graph)
        new_fields = self.update_input_types(new_fields)

        return (
            graph.description,
            [field for field in new_fields if field.get("tool_mode") is True],
        )

    def update_input_types(self, fields: list[dotdict]) -> list[dotdict]:
        """Update the input_types of the fields.

            Ensures all fields can accept Message type connections, enabling
            both TextInput and ChatInput components to be wired to Run Flow
            input terminals.

        Args:
            fields: The fields to update the input_types for.

        Returns:
            The updated fields.
        """
        for field in fields:
            if isinstance(field, dict):
                input_types = field.get("input_types")
                # Ensure Message type is accepted to allow ChatInput/TextInput connections
                if input_types is None or len(input_types) == 0:
                    field["input_types"] = ["Message"]
                elif "Message" not in input_types:
                    field["input_types"] = list(input_types) + ["Message"]
            elif hasattr(field, "input_types"):
                if field.input_types is None or len(field.input_types) == 0:
                    field.input_types = ["Message"]
                elif "Message" not in field.input_types:
                    field.input_types = list(field.input_types) + ["Message"]
        return fields

    async def _get_tools(self) -> list[Tool]:
        """Expose flow as a tool."""
        component_toolkit: type[ComponentToolkit] = get_component_toolkit()
        flow_description, tool_mode_inputs = await self.get_required_data()
        if not tool_mode_inputs:
            return []
        # convert list of dicts to list of dotdicts
        tool_mode_inputs = [dotdict(field) for field in tool_mode_inputs]
        return component_toolkit(component=self).get_tools(
            tool_name=f"{self.flow_name_selected}_tool",
            tool_description=(
                f"Tool designed to execute the flow '{self.flow_name_selected}'. Flow details: {flow_description}."
            ),
            callbacks=self.get_langchain_callbacks(),
            flow_mode_inputs=tool_mode_inputs,
        )

    ################################################################
    # Flow output resolution
    ################################################################
    async def _get_cached_run_outputs(
        self,
        *,
        user_id: str | None = None,
        tweaks: dict | None,
        inputs: dict | list[dict] | None,
        output_type: str,
    ):
        if self._last_run_outputs is not None:
            return self._last_run_outputs
        resolved_tweaks = tweaks or self.flow_tweak_data or {}
        resolved_inputs = (
            inputs or self._flow_run_inputs or self._build_inputs_from_tweaks(resolved_tweaks)
        ) or None
        self._last_run_outputs = await self._run_flow_with_cached_graph(
            user_id=user_id,
            tweaks=resolved_tweaks,
            inputs=resolved_inputs,
            output_type=output_type,
        )
        return self._last_run_outputs

    async def _resolve_flow_output(self, *, vertex_id: str, output_name: str):
        """Resolve the value of a given vertex's output.

            Given a vertex_id and output_name, it will resolve the value of the output
            belonging to the vertex with the given vertex_id.

        Args:
            vertex_id: The ID of the vertex to resolve the output for.
            output_name: The name of the output to resolve.

        Returns:
            The resolved output.
        """
        run_outputs = await self._get_cached_run_outputs(
            user_id=self.user_id,
            tweaks=self.flow_tweak_data,
            inputs=None,
            output_type="any",
        )

        if not run_outputs:
            return None
        first_output = run_outputs[0]
        if not first_output.outputs:
            return None
        for result in first_output.outputs:
            if not (result and result.component_id == vertex_id):
                continue
            if isinstance(result.results, dict) and output_name in result.results:
                return result.results[output_name]
            if result.artifacts and output_name in result.artifacts:
                return result.artifacts[output_name]
            return result.results or result.artifacts or result.outputs

        return None

    def _clear_dynamic_flow_output_methods(self) -> None:
        for method_name in self._flow_output_methods:
            if hasattr(self, method_name):
                delattr(self, method_name)
        self._flow_output_methods.clear()

    def _register_flow_output_method(self, *, vertex_id: str, output_name: str) -> str:
        safe_vertex = vertex_id.replace("-", "_")
        safe_output = output_name.replace("-", "_").replace(self.IOPUT_SEP, "_")
        method_name = f"_resolve_flow_output__{safe_vertex}__{safe_output}"

        async def _dynamic_resolver(_self):
            return await _self._resolve_flow_output(  # noqa: SLF001
                vertex_id=vertex_id,
                output_name=output_name,
            )

        setattr(self, method_name, MethodType(_dynamic_resolver, self))
        self._flow_output_methods.add(method_name)
        return method_name

    ################################################################
    # Dynamic flow output synchronization
    ################################################################
    def _sync_flow_outputs(self, outputs: list[Output]) -> None:
        """Persist dynamic flow outputs in the component.

        Args:
            outputs: The list of Output objects to persist.

        Returns:
            None
        """
        tool_output = None
        if TOOL_OUTPUT_NAME in self._outputs_map:
            tool_output = self._outputs_map[TOOL_OUTPUT_NAME]
        else:
            tool_output = next(
                (out for out in outputs if out and out.name == TOOL_OUTPUT_NAME),
                None,
            )

        self.outputs = outputs
        self._outputs_map = {out.name: out for out in outputs if out}
        self._outputs_map[TOOL_OUTPUT_NAME] = tool_output

    async def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        """Update the outputs of the frontend node.

        This method is called when the flow_name_selected field is updated.
        It will generate the Output objects for the selected flow and update the outputs of the frontend node.

        Args:
            frontend_node: The frontend node to update the outputs for.
            field_name: The name of the field that was updated.
            field_value: The value of the field that was updated.

        Returns:
            The updated frontend node.
        """
        if field_name != "flow_name_selected" or not field_value:
            return frontend_node

        flow_selected_metadata = (
            frontend_node.get("template", {})
            .get("flow_name_selected", {})
            .get("selected_metadata", {})
        )
        graph = await self.get_graph(
            flow_name_selected=field_value,
            flow_id_selected=flow_selected_metadata.get("id"),
            updated_at=flow_selected_metadata.get("updated_at"),
        )
        outputs = self._format_flow_outputs(
            graph
        )  # generate Output objects from the flow's output nodes
        self._sync_flow_outputs(outputs)
        frontend_node["outputs"] = [output.model_dump() for output in outputs]
        return frontend_node

    ################################################################
    # Tool mode + formatting
    ################################################################
    def _format_flow_outputs(self, graph: Graph) -> list[Output]:
        """Generate Output objects from the graph's outputs.

        The Output objects modify the name and method of the graph's outputs.
        The name is modified by prepending the vertex_id and to the original name,
        which uniquely identifies the output.
        The method is set to a dynamically generated method which uses a unique name
        to resolve the output to its value generated during the flow execution.

        Args:
            graph: The graph to generate outputs for.

        Returns:
            A list of Output objects.
        """
        output_vertices: list[Vertex] = [v for v in graph.vertices if v.is_output]
        outputs: list[Output] = []
        vdisp_cts = Counter(v.display_name for v in output_vertices)
        for vertex in output_vertices:
            one_out = len(vertex.outputs) == 1
            for vertex_output in vertex.outputs:
                new_name = self._get_ioput_name(vertex.id, vertex_output.get("name"))
                output = Output(**vertex_output)
                output.name = new_name
                output.method = self._register_flow_output_method(
                    vertex_id=vertex.id,
                    output_name=vertex_output.get("name"),
                )
                vdn = vertex.display_name
                odn = output.display_name
                output.display_name = (
                    vdn
                    if one_out and vdisp_cts[vdn] == 1
                    else odn
                    + (
                        # output.display_name potentially collides w/ those of other vertices
                        f" ({vdn})"
                        if vdisp_cts[vdn] == 1
                        # output.display_name collides w/ those of duplicate vertices
                        else f"-{vertex.id}"
                    )
                )
                outputs.append(output)

        return outputs

    def _get_ioput_name(
        self,
        vertex_id: str,
        ioput_name: str,
    ) -> str:
        """Helper for joining a vertex id and input/output name to form a unique input/output name.

        Args:
            vertex_id: The ID of the vertex who's input/output name is being generated.
            ioput_name: The name of the input/output to get the name for.

        Returns:
            A unique output name for the given vertex's output.
        """
        if not vertex_id or not ioput_name:
            msg = "Vertex ID and input/output name are required"
            raise ValueError(msg)
        return f"{vertex_id}{self.IOPUT_SEP}{ioput_name}"

    ################################################################
    # Flow execution
    ################################################################
    async def _run_flow_with_cached_graph(
        self,
        *,
        user_id: str | None = None,
        tweaks: dict | None = None,
        inputs: dict | list[dict] | None = None,
        output_type: str = "any",  # "any" is used to return all outputs
    ):
        # When tweaks are provided, we must apply them to the FLOW DATA before
        # creating the Graph. The old approach using process_tweaks_on_graph()
        # doesn't work because apply_tweaks_on_vertex() only updates vertex.params
        # if the key already exists there.
        #
        # The correct approach (matching simple_run_flow in endpoints.py) is:
        # 1. Get the flow data
        # 2. Apply tweaks using process_tweaks() which modifies the JSON template data
        # 3. Create the Graph from the tweaked data
        # 4. Run the graph with the provided inputs

        if tweaks:
            # Save original vertex IDs before process_tweaks modifies the dict
            # (process_tweaks adds "stream" key to the dict in place)
            original_vertex_ids = list(tweaks.keys())

            # Get fresh flow data (don't use cache when tweaks are provided)
            flow = await self.get_flow(
                flow_name_selected=self.flow_name_selected,
                flow_id_selected=self.flow_id_selected,
            )
            if not flow:
                msg = "Flow not found"
                raise ValueError(msg)

            # Apply tweaks to the flow data BEFORE creating the Graph
            flow_data = flow.data.get("data", {})
            tweaked_flow_data = process_tweaks(flow_data, tweaks)

            # Create Graph from the tweaked data
            graph = Graph.from_payload(
                payload=tweaked_flow_data,
                flow_id=self.flow_id_selected,
                flow_name=self.flow_name_selected,
            )
            graph.description = flow.data.get("description", None)
            graph.updated_at = flow.data.get("updated_at", None)

            logger.debug(
                f"[RunFlow] Applied tweaks to {len(original_vertex_ids)} vertices "
                f"in flow {self.flow_name_selected}"
            )
        else:
            # No tweaks - use cached graph for performance
            graph = await self.get_graph(
                flow_name_selected=self.flow_name_selected,
                flow_id_selected=self.flow_id_selected,
                updated_at=self._cached_flow_updated_at,
            )

        # Run the graph with the provided inputs (which contain the same values as tweaks).
        # We need to pass inputs so the graph.arun loop executes at least once.
        run_inputs = inputs if inputs else []
        logger.debug(f"[RunFlow] Running graph with {len(run_inputs)} inputs")
        return await run_flow(
            inputs=run_inputs,
            flow_id=self.flow_id_selected,
            flow_name=self.flow_name_selected,
            user_id=user_id,
            session_id=self.session_id,
            output_type=output_type,
            graph=graph,
        )

    ################################################################
    # Flow cache utils
    ################################################################
    def _flow_cache_call(self, action: str, *args, **kwargs):
        """Call a flow cache related method."""
        if not self.cache_flow:
            msg = "Cache flow is disabled"
            logger.warning(msg)
            return None
        if self._shared_component_cache is None:
            logger.warning("Shared component cache is not available")
            return None

        handler = self._cache_flow_dispatcher.get(action)
        if handler is None:
            msg = f"Unknown cache action: {action}"
            raise ValueError(msg)
        try:
            return handler(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            key = (
                kwargs.get("cache_key")
                or kwargs.get("flow_name")
                or kwargs.get("flow_name_selected")
            )
            if not key and args:
                key = args[0]
            logger.warning("Cache %s failed for key %s: %s", action, key or "[missing key]", exc)
            return None

    def _get_cached_flow(self, *, flow_id: str | None = None) -> Graph | None:
        cache_key = self._build_flow_cache_key(flow_id=flow_id)
        cache_entry = self._shared_component_cache.get(cache_key)
        if isinstance(cache_entry, CacheMiss):
            logger.debug(f"{cache_entry} for key {cache_key}")
            return None
        if not cache_entry:
            logger.warning(f"None or empty cache entry ({cache_entry}) for key {cache_key}")
            return None
        return self._build_graph_from_dict(cache_entry=cache_entry)

    def _set_cached_flow(self, *, flow: Graph) -> None:
        graph_dump = flow.dump()
        payload = {
            "graph_dump": graph_dump,
            "flow_id": flow.flow_id,
            "user_id": self.user_id,
            "description": flow.description or graph_dump.get("description"),
            "updated_at": flow.updated_at or graph_dump.get("updated_at"),
        }
        cache_key = self._build_flow_cache_key(flow_id=flow.flow_id)
        self._shared_component_cache.set(cache_key, payload)

    def _build_flow_cache_key(self, *, flow_id: str | None = None) -> str | None:
        """Build a cache key for a flow.

        Raises a ValueError if the user or flow ID is not provided.

        Args:
            flow_id: The ID of the flow to build the cache key for.

        Returns:
            The cache key for the flow.
        """
        if not (self.user_id and flow_id):
            msg = "Failed to build cache key: Flow ID and user ID are required"
            raise ValueError(msg)
        return f"run_flow:{self.user_id}:{flow_id or 'missing_id'}"

    def _build_graph_from_dict(self, *, cache_entry: dict[str, Any]) -> Graph | None:
        if not (graph_dump := cache_entry.get("graph_dump")):
            return None
        graph = Graph.from_payload(
            payload=graph_dump.get("data", {}),
            flow_id=cache_entry.get("flow_id"),
            flow_name=cache_entry.get("flow_name"),
            user_id=cache_entry.get("user_id"),
        )
        graph.description = cache_entry.get("description") or graph_dump.get("description")
        graph.updated_at = cache_entry.get("updated_at") or graph_dump.get("updated_at")
        return graph

    def _is_cached_flow_up_to_date(self, cached_flow: Graph, updated_at: str | None) -> bool:
        if not updated_at or not (cached_ts := getattr(cached_flow, "updated_at", None)):
            return False  # both timetamps must be present
        return self._parse_timestamp(cached_ts) >= self._parse_timestamp(updated_at)

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        from datetime import timezone

        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.replace(tzinfo=timezone.utc, microsecond=0)
        except ValueError:
            logger.warning("Invalid updated_at value: %s", value)
            return None

    def _delete_cached_flow(self, flow_id: str | None) -> None:
        """Remove the flow with the given ID or name from cache.

        Args:
            flow_id: The ID of the flow to delete from cache.
            flow_name: The name of the flow to delete from cache.

        Returns:
            None
        """
        err_msg_prefix = "Failed to delete user flow from cache"
        if self._shared_component_cache is None:
            msg = f"{err_msg_prefix}: Shared component cache is not available"
            raise ValueError(msg)
        if not self.user_id:
            msg = f"{err_msg_prefix}: Please provide your user ID"
            raise ValueError(msg)
        if not flow_id or not flow_id.strip():
            msg = f"{err_msg_prefix}: Please provide a valid flow ID"
            raise ValueError(msg)

        self._shared_component_cache.delete(self._build_flow_cache_key(flow_id=flow_id))

    ################################################################
    # Build inputs and flow tweak data
    ################################################################
    def _normalize_input_value(self, value: Any) -> str | Any:
        """Normalize input value to string if it's a complex object.

        When a prompt template or other component is wired to the input terminal,
        it may pass a Message, Data, or dict object instead of a plain string.
        This method extracts the text content from such objects.

        Args:
            value: The input value to normalize (may be str, dict, Message, Data, etc.)

        Returns:
            The extracted string value, or the original value if no text content found.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        # Preserve lists as-is (e.g., files field should remain a list, not be stringified)
        if isinstance(value, list):
            return value
        # Handle Message, Data, or dict-like objects
        if isinstance(value, dict):
            for key in ("text", "content", "message", "input_value"):
                if key in value:
                    return self._normalize_input_value(value[key])
            return value  # Return as-is if no known text keys
        # Handle objects with text/content attributes (Message, Data classes)
        for attr in ("text", "content", "message"):
            if hasattr(value, attr):
                attr_value = getattr(value, attr)
                if attr_value is not None:
                    return self._normalize_input_value(attr_value)
        return str(value)

    def _extract_tweaks_from_keyed_values(
        self,
        values: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        tweaks: dict[str, dict[str, Any]] = {}
        if not values:
            return tweaks
        for field_name, field_value in values.items():
            if self.IOPUT_SEP not in field_name:
                continue
            node_id, param_name = field_name.split(self.IOPUT_SEP, 1)
            # Normalize input values from text input/chat input/prompt templates
            # to handle Message/Data objects that should be plain strings
            normalized_value = self._normalize_input_value(field_value)
            tweaks.setdefault(node_id, {})[param_name] = normalized_value
        return tweaks

    def _infer_input_type(self, vertex_id: str) -> str:
        """Infer the input type from the vertex ID.

        Args:
            vertex_id: The vertex ID (e.g., 'TextInput-4z0su', 'ChatInput-abc123')

        Returns:
            The inferred type: 'text' for TextInput, 'chat' for ChatInput, etc.
        """
        vertex_id_lower = vertex_id.lower()
        if vertex_id_lower.startswith("textinput"):
            return "text"
        if vertex_id_lower.startswith("chatinput"):
            return "chat"
        if vertex_id_lower.startswith("messagetextinput"):
            return "text"
        # Default to text for unknown types (safer than chat)
        return "text"

    def _get_parent_input_value_by_type(self, input_type: str) -> str | None:
        """Get the parent graph's input value for a matching input type.

        When a sub-flow has a ChatInput/TextInput that doesn't have an explicit value
        wired to it, this method attempts to find the parent graph's corresponding
        input value to forward automatically.

        Args:
            input_type: The type of input to look for ('chat' or 'text')

        Returns:
            The parent's input value if found, None otherwise.
        """
        if not hasattr(self, "graph") or self.graph is None:
            return None

        # Get the parent graph's input vertices
        parent_input_vertices = getattr(self.graph, "_is_input_vertices", [])
        if not parent_input_vertices:
            return None

        for vertex_id in parent_input_vertices:
            # Check if this vertex matches the requested input type
            vertex_id_lower = vertex_id.lower()
            if input_type == "chat" and "chatinput" not in vertex_id_lower:
                continue
            if input_type == "text" and "textinput" not in vertex_id_lower:
                continue

            vertex = self.graph.get_vertex(vertex_id)
            if vertex is None:
                continue

            # Try to get the input_value from the vertex's raw params
            raw_params = getattr(vertex, "_raw_params", {})
            if raw_params and "input_value" in raw_params:
                value = raw_params.get("input_value")
                if value:
                    return self._normalize_input_value(value)

            # Try to get from vertex data
            vertex_template = vertex.data.get("node", {}).get("template", {})
            input_value_field = vertex_template.get("input_value", {})
            if isinstance(input_value_field, dict):
                value = input_value_field.get("value")
                if value:
                    return self._normalize_input_value(value)

        return None

    def _build_inputs_from_tweaks(
        self,
        tweaks: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        inputs: list[dict[str, Any]] = []
        for vertex_id, params in tweaks.items():
            if "input_value" not in params:
                continue
            # Normalize input value to handle Message/Data objects from prompt templates
            # (defensive - values should already be normalized by _extract_tweaks_from_keyed_values)
            normalized_value = self._normalize_input_value(params["input_value"])

            # Infer input type from vertex ID if not explicitly provided
            input_type = params.get("type") or self._infer_input_type(vertex_id)

            # If the sub-flow input has an empty value, try to forward the parent's
            # matching input (e.g., parent ChatInput -> sub-flow ChatInput)
            if not normalized_value:
                parent_value = self._get_parent_input_value_by_type(input_type)
                if parent_value:
                    normalized_value = parent_value
                    display_value = (
                        f"{normalized_value[:50]}..."
                        if len(str(normalized_value)) > 50
                        else normalized_value
                    )
                    self.log(
                        f"[RunFlow] Auto-forwarding parent {input_type} input to {vertex_id}: {display_value}"
                    )

            payload: dict[str, Any] = {
                "components": [vertex_id],
                "input_value": normalized_value,
                "type": input_type,
            }
            inputs.append(payload)
        return inputs

    def _get_selected_flow_updated_at(self) -> str | None:
        updated_at = (
            getattr(self, "_vertex", {})
            .data.get("node", {})
            .get("template", {})
            .get("flow_name_selected", {})
            .get("selected_metadata", {})
            .get("updated_at", None)
        )
        if updated_at:
            return updated_at
        return self._attributes.get("flow_name_selected_updated_at")

    def _pre_run_setup(self) -> None:  # Note: overrides the base pre_run_setup method
        """Reset the last run's outputs upon new flow execution."""
        self._last_run_outputs = None
        self._cached_flow_updated_at = self._get_selected_flow_updated_at()
        if self._cached_flow_updated_at:
            self._attributes["flow_name_selected_updated_at"] = self._cached_flow_updated_at
        self._attributes["flow_tweak_data"] = {}
        self.flow_tweak_data = self._extract_tweaks_from_keyed_values(self._attributes)
        self._flow_run_inputs = self._build_inputs_from_tweaks(self.flow_tweak_data)
