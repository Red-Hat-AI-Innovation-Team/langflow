from collections import defaultdict
from unittest.mock import MagicMock

import pytest
from lfx.components.processing.data_operations import DataOperationsComponent
from lfx.schema import Data

from tests.base import ComponentTestBaseWithoutClient


class TestDataOperationsComponent(ComponentTestBaseWithoutClient):
    @pytest.fixture
    def component_class(self):
        """Return the component class to test."""
        return DataOperationsComponent

    @pytest.fixture
    def default_kwargs(self):
        """Return the default kwargs for the component."""
        return {
            "data": Data(data={"key1": "value1", "key2": "value2", "key3": "value3"}),
            "actions": [{"name": "Select Keys"}],
            "select_keys_input": ["key1", "key2"],
        }

    @pytest.fixture
    def file_names_mapping(self):
        """Return an empty list since this component doesn't have version-specific files."""
        return []

    def test_select_keys(self):
        """Test the Select Keys operation."""
        component = DataOperationsComponent(
            data=Data(data={"key1": "value1", "key2": "value2", "key3": "value3"}),
            operations=[{"name": "Select Keys"}],
            select_keys_input=["key1", "key2"],
        )

        result = component.as_data()
        assert isinstance(result, Data)
        assert "key1" in result.data
        assert "key2" in result.data
        assert "key3" not in result.data
        assert result.data["key1"] == "value1"
        assert result.data["key2"] == "value2"

    def test_remove_keys(self):
        """Test the Remove Keys operation."""
        component = DataOperationsComponent(
            data=Data(data={"key1": "value1", "key2": "value2", "key3": "value3"}),
            operations=[{"name": "Remove Keys"}],
            remove_keys_input=["key3"],
        )

        result = component.as_data()
        assert isinstance(result, Data)
        assert "key1" in result.data
        assert "key2" in result.data
        assert "key3" not in result.data

    def test_rename_keys(self):
        """Test the Rename Keys operation."""
        component = DataOperationsComponent(
            data=Data(data={"key1": "value1", "key2": "value2"}),
            operations=[{"name": "Rename Keys"}],
            rename_keys_input={"key1": "new_key1"},
        )

        result = component.as_data()
        assert isinstance(result, Data)
        assert "new_key1" in result.data
        assert "key1" not in result.data
        assert result.data["new_key1"] == "value1"

    def test_literal_eval(self):
        """Test the Literal Eval operation."""
        component = DataOperationsComponent(
            data=Data(data={"list_as_string": "[1, 2, 3]", "dict_as_string": "{'a': 1, 'b': 2}"}),
            operations=[{"name": "Literal Eval"}],
        )

        result = component.as_data()
        assert isinstance(result, Data)
        assert isinstance(result.data["list_as_string"], list)
        assert result.data["list_as_string"] == [1, 2, 3]
        assert isinstance(result.data["dict_as_string"], dict)
        assert result.data["dict_as_string"] == {"a": 1, "b": 2}

    def test_combine(self):
        """Test the Combine operation."""
        data1 = Data(data={"key1": "value1"})
        data2 = Data(data={"key2": "value2"})

        component = DataOperationsComponent(
            data=[data1, data2],
            operations=[{"name": "Combine"}],
        )

        result = component.as_data()
        assert isinstance(result, Data)
        assert "key1" in result.data
        assert "key2" in result.data
        assert result.data["key1"] == "value1"
        assert result.data["key2"] == "value2"

    def test_combine_with_overlapping_keys(self):
        """Test the Combine operation with overlapping keys."""
        data1 = Data(data={"common_key": "value1", "key1": "value1"})
        data2 = Data(data={"common_key": "value2", "key2": "value2"})

        component = DataOperationsComponent(
            data=[data1, data2],
            operations=[{"name": "Combine"}],
        )

        result = component.as_data()
        assert isinstance(result, Data)
        assert result.data["common_key"] == ["value1", "value2"]  # Combined string values
        assert result.data["key1"] == "value1"
        assert result.data["key2"] == "value2"

    def test_append_update(self):
        """Test the Append or Update Data operation."""
        component = DataOperationsComponent(
            data=Data(data={"existing_key": "existing_value"}),
            operations=[{"name": "Append or Update"}],
            append_update_data={"new_key": "new_value", "existing_key": "updated_value"},
        )

        result = component.as_data()
        assert isinstance(result, Data)
        assert result.data["existing_key"] == "updated_value"
        assert result.data["new_key"] == "new_value"

    def test_filter_values(self):
        """Test the Filter Values operation."""
        nested_data = {
            "items": [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2"},
                {"id": 3, "name": "Different Item"},
            ]
        }

        component = DataOperationsComponent(
            data=Data(data=nested_data),
            operations=[{"name": "Filter Values"}],
            filter_key=["items"],
            filter_values={"name": "Item"},
            operator="contains",
        )

        result = component.as_data()
        assert isinstance(result, Data)
        assert len(result.data["items"]) == 3
        assert result.data["items"][0]["id"] == 1
        assert result.data["items"][1]["id"] == 2

    def test_no_actions(self):
        """Test behavior when no actions are specified."""
        component = DataOperationsComponent(
            data=Data(data={"key1": "value1"}),
            operations=[],
        )

        result = component.as_data()
        assert isinstance(result, Data)
        assert result.data == {}

    def test_get_normalized_data(self):
        """Test the get_normalized_data helper method."""
        component = DataOperationsComponent(
            data=Data(data={"key1": "value1"}),
            operations=[],
        )

        # Add data under the "data" key
        component.data = Data(data={"test": {"key2": "value2"}})
        normalized = component.get_normalized_data()
        assert normalized == {"test": {"key2": "value2"}}

        # Test without the "data" key
        component.data = Data(data={"key3": "value3"})
        normalized = component.get_normalized_data()
        assert normalized == {"key3": "value3"}

    def test_validate_single_data_with_multiple_data(self):
        """Test that operations that don't support multiple data objects raise an error."""
        component = DataOperationsComponent(
            data=[Data(data={"key1": "value1"}), Data(data={"key2": "value2"})],
            operations=[{"name": "Select Keys"}],
            select_keys_input=["key1"],
        )

        with pytest.raises(ValueError, match="Select Keys operation is not supported for multiple data objects"):
            component.as_data()

    def test_combine_updates_dependencies_in_cycle(self):
        """Test that Combine re-adds data input dependencies for proper loop synchronization."""
        # Create component with mock vertex and graph
        data1 = Data(data={"key1": "value1"})
        data2 = Data(data={"key2": "value2"})

        component = DataOperationsComponent(
            data=[data1, data2],
            operations=[{"name": "Combine"}],
        )

        # Mock graph with run_manager
        mock_graph = MagicMock()
        mock_graph.run_manager.run_predecessors = defaultdict(list)
        mock_graph.run_manager.run_map = defaultdict(list)

        # Mock vertex with cycle edges and incoming data edges
        # The graph property returns self._vertex.graph, so we set it on the mock_vertex
        mock_vertex = MagicMock()
        mock_vertex.has_cycle_edges = True
        mock_vertex.incoming_edges = [
            MagicMock(target_param="data", source_id="source1"),
            MagicMock(target_param="data", source_id="source2"),
        ]
        mock_vertex.graph = mock_graph
        component._vertex = mock_vertex

        # Run combine
        result = component.as_data()

        # Verify result is correct
        assert isinstance(result, Data)
        assert "key1" in result.data
        assert "key2" in result.data

        # Verify dependencies were re-added
        assert "source1" in mock_graph.run_manager.run_predecessors[component._id]
        assert "source2" in mock_graph.run_manager.run_predecessors[component._id]

        # Verify run_map was updated
        assert component._id in mock_graph.run_manager.run_map["source1"]
        assert component._id in mock_graph.run_manager.run_map["source2"]

    def test_combine_does_not_update_dependencies_without_cycle(self):
        """Test that Combine does not update dependencies when not in a cycle context."""
        data1 = Data(data={"key1": "value1"})
        data2 = Data(data={"key2": "value2"})

        component = DataOperationsComponent(
            data=[data1, data2],
            operations=[{"name": "Combine"}],
        )

        # Mock graph with run_manager
        mock_graph = MagicMock()
        mock_graph.run_manager.run_predecessors = defaultdict(list)
        mock_graph.run_manager.run_map = defaultdict(list)

        # Mock vertex without cycle edges
        mock_vertex = MagicMock()
        mock_vertex.has_cycle_edges = False
        mock_vertex.incoming_edges = [
            MagicMock(target_param="data", source_id="source1"),
            MagicMock(target_param="data", source_id="source2"),
        ]
        mock_vertex.graph = mock_graph
        component._vertex = mock_vertex

        # Run combine
        result = component.as_data()

        # Verify result is correct
        assert isinstance(result, Data)
        assert "key1" in result.data
        assert "key2" in result.data

        # Verify dependencies were NOT added (no cycle context)
        assert "source1" not in mock_graph.run_manager.run_predecessors[component._id]
        assert "source2" not in mock_graph.run_manager.run_predecessors[component._id]
