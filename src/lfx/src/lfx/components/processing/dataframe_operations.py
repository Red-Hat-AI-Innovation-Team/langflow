import pandas as pd

from lfx.custom.custom_component.component import Component
from lfx.inputs import SortableListInput
from lfx.io import BoolInput, DataFrameInput, DropdownInput, IntInput, MessageTextInput, Output, StrInput
from lfx.log.logger import logger
from lfx.schema.dataframe import DataFrame


class DataFrameOperationsComponent(Component):
    display_name = "DataFrame Operations"
    description = "Perform various operations on a DataFrame."
    documentation: str = "https://docs.langflow.org/dataframe-operations"
    icon = "table"
    name = "DataFrameOperations"

    OPERATION_CHOICES = [
        "Add Column",
        "Drop Column",
        "Drop NA",
        "Fill NA",
        "Filter",
        "Head",
        "Rename Column",
        "Replace Value",
        "Reset Index",
        "Sample",
        "Select Columns",
        "Set Index",
        "Slice",
        "Sort",
        "Tail",
        "Transpose",
        "Unique Values",
        "Drop Duplicates",
    ]

    inputs = [
        DataFrameInput(
            name="df",
            display_name="DataFrame",
            info="The input DataFrame to operate on.",
            required=True,
        ),
        SortableListInput(
            name="operation",
            display_name="Operation",
            placeholder="Select Operation",
            info="Select the DataFrame operation to perform.",
            options=[
                {"name": "Add Column", "icon": "plus"},
                {"name": "Drop Column", "icon": "minus"},
                {"name": "Drop Duplicates", "icon": "copy-x"},
                {"name": "Drop NA", "icon": "eraser"},
                {"name": "Fill NA", "icon": "pencil-line"},
                {"name": "Filter", "icon": "filter"},
                {"name": "Head", "icon": "arrow-up"},
                {"name": "Rename Column", "icon": "pencil"},
                {"name": "Replace Value", "icon": "replace"},
                {"name": "Reset Index", "icon": "list-restart"},
                {"name": "Sample", "icon": "shuffle"},
                {"name": "Select Columns", "icon": "columns"},
                {"name": "Set Index", "icon": "list-ordered"},
                {"name": "Slice", "icon": "scissors"},
                {"name": "Sort", "icon": "arrow-up-down"},
                {"name": "Tail", "icon": "arrow-down"},
                {"name": "Transpose", "icon": "flip-horizontal"},
                {"name": "Unique Values", "icon": "fingerprint"},
            ],
            real_time_refresh=True,
            limit=1,
        ),
        StrInput(
            name="column_name",
            display_name="Column Name",
            info="The column name to use for the operation.",
            dynamic=True,
            show=False,
        ),
        MessageTextInput(
            name="filter_value",
            display_name="Filter Value",
            info="The value to filter rows by.",
            dynamic=True,
            show=False,
        ),
        DropdownInput(
            name="filter_operator",
            display_name="Filter Operator",
            options=[
                "equals",
                "not equals",
                "contains",
                "not contains",
                "starts with",
                "ends with",
                "greater than",
                "less than",
            ],
            value="equals",
            info="The operator to apply for filtering rows.",
            advanced=False,
            dynamic=True,
            show=False,
        ),
        BoolInput(
            name="ascending",
            display_name="Sort Ascending",
            info="Whether to sort in ascending order.",
            dynamic=True,
            show=False,
            value=True,
        ),
        StrInput(
            name="new_column_name",
            display_name="New Column Name",
            info="The new column name when renaming or adding a column.",
            dynamic=True,
            show=False,
        ),
        MessageTextInput(
            name="new_column_value",
            display_name="New Column Value",
            info="The value to populate the new column with.",
            dynamic=True,
            show=False,
        ),
        StrInput(
            name="columns_to_select",
            display_name="Columns to Select",
            dynamic=True,
            is_list=True,
            show=False,
        ),
        IntInput(
            name="num_rows",
            display_name="Number of Rows",
            info="Number of rows to return (for head/tail).",
            dynamic=True,
            show=False,
            value=5,
        ),
        MessageTextInput(
            name="replace_value",
            display_name="Value to Replace",
            info="The value to replace in the column.",
            dynamic=True,
            show=False,
        ),
        MessageTextInput(
            name="replacement_value",
            display_name="Replacement Value",
            info="The value to replace with.",
            dynamic=True,
            show=False,
        ),
        BoolInput(
            name="drop_index",
            display_name="Drop Index",
            info="Whether to drop the old index instead of adding it as a column.",
            dynamic=True,
            show=False,
            value=True,
        ),
        MessageTextInput(
            name="fill_value",
            display_name="Fill Value",
            info="The value to fill missing values with.",
            dynamic=True,
            show=False,
        ),
        IntInput(
            name="start_index",
            display_name="Start Index",
            info="The starting row index for slicing (0-based, inclusive).",
            dynamic=True,
            show=False,
            value=0,
        ),
        IntInput(
            name="end_index",
            display_name="End Index",
            info="The ending row index for slicing (0-based, exclusive).",
            dynamic=True,
            show=False,
            value=10,
        ),
        IntInput(
            name="random_seed",
            display_name="Random Seed",
            info="Optional seed for reproducible random sampling.",
            dynamic=True,
            show=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="DataFrame",
            name="output",
            method="perform_operation",
            info="The resulting DataFrame after the operation.",
        )
    ]

    def update_build_config(self, build_config, field_value, field_name=None):
        dynamic_fields = [
            "column_name",
            "filter_value",
            "filter_operator",
            "ascending",
            "new_column_name",
            "new_column_value",
            "columns_to_select",
            "num_rows",
            "replace_value",
            "replacement_value",
            "drop_index",
            "fill_value",
            "start_index",
            "end_index",
            "random_seed",
        ]
        for field in dynamic_fields:
            build_config[field]["show"] = False

        if field_name == "operation":
            # Handle SortableListInput format
            if isinstance(field_value, list):
                operation_name = field_value[0].get("name", "") if field_value else ""
            else:
                operation_name = field_value or ""

            # If no operation selected, all dynamic fields stay hidden (already set to False above)
            if not operation_name:
                return build_config

            if operation_name == "Filter":
                build_config["column_name"]["show"] = True
                build_config["filter_value"]["show"] = True
                build_config["filter_operator"]["show"] = True
            elif operation_name == "Sort":
                build_config["column_name"]["show"] = True
                build_config["ascending"]["show"] = True
            elif operation_name == "Drop Column":
                build_config["column_name"]["show"] = True
            elif operation_name == "Rename Column":
                build_config["column_name"]["show"] = True
                build_config["new_column_name"]["show"] = True
            elif operation_name == "Add Column":
                build_config["new_column_name"]["show"] = True
                build_config["new_column_value"]["show"] = True
            elif operation_name == "Select Columns":
                build_config["columns_to_select"]["show"] = True
            elif operation_name in {"Head", "Tail"}:
                build_config["num_rows"]["show"] = True
            elif operation_name == "Replace Value":
                build_config["column_name"]["show"] = True
                build_config["replace_value"]["show"] = True
                build_config["replacement_value"]["show"] = True
            elif operation_name == "Drop Duplicates":
                build_config["column_name"]["show"] = True
            elif operation_name == "Transpose":
                pass  # No additional inputs needed
            elif operation_name == "Reset Index":
                build_config["drop_index"]["show"] = True
            elif operation_name == "Set Index":
                build_config["column_name"]["show"] = True
            elif operation_name == "Fill NA":
                build_config["column_name"]["show"] = True
                build_config["fill_value"]["show"] = True
            elif operation_name == "Drop NA":
                build_config["column_name"]["show"] = True
            elif operation_name == "Sample":
                build_config["num_rows"]["show"] = True
                build_config["random_seed"]["show"] = True
            elif operation_name == "Slice":
                build_config["start_index"]["show"] = True
                build_config["end_index"]["show"] = True
            elif operation_name == "Unique Values":
                build_config["column_name"]["show"] = True

        return build_config

    def perform_operation(self) -> DataFrame:
        df_copy = self.df.copy()

        # Handle SortableListInput format for operation
        operation_input = getattr(self, "operation", [])
        if isinstance(operation_input, list) and len(operation_input) > 0:
            op = operation_input[0].get("name", "")
        else:
            op = ""

        # If no operation selected, return original DataFrame
        if not op:
            return df_copy

        # Log input
        input_shape = df_copy.shape if hasattr(df_copy, "shape") else "unknown"
        filter_val = getattr(self, "filter_value", "")
        col_name = getattr(self, "column_name", "")
        self.log(
            f"[DFOPS] INPUT: shape={input_shape}, operation={op}, column={col_name}, filter_value={filter_val}",
            name=f"[{self._id}]",
        )

        result = None
        if op == "Filter":
            result = self.filter_rows_by_value(df_copy)
        elif op == "Sort":
            result = self.sort_by_column(df_copy)
        elif op == "Drop Column":
            result = self.drop_column(df_copy)
        elif op == "Rename Column":
            result = self.rename_column(df_copy)
        elif op == "Add Column":
            result = self.add_column(df_copy)
        elif op == "Select Columns":
            result = self.select_columns(df_copy)
        elif op == "Head":
            result = self.head(df_copy)
        elif op == "Tail":
            result = self.tail(df_copy)
        elif op == "Replace Value":
            result = self.replace_values(df_copy)
        elif op == "Drop Duplicates":
            result = self.drop_duplicates(df_copy)
        elif op == "Transpose":
            result = self.transpose(df_copy)
        elif op == "Reset Index":
            result = self.reset_index(df_copy)
        elif op == "Set Index":
            result = self.set_index(df_copy)
        elif op == "Fill NA":
            result = self.fill_na(df_copy)
        elif op == "Drop NA":
            result = self.drop_na(df_copy)
        elif op == "Sample":
            result = self.sample_rows(df_copy)
        elif op == "Slice":
            result = self.slice_rows(df_copy)
        elif op == "Unique Values":
            result = self.unique_values(df_copy)
        else:
            msg = f"Unsupported operation: {op}"
            logger.error(msg)
            raise ValueError(msg)

        # Log output
        output_shape = result.shape if hasattr(result, "shape") else "unknown"
        self.log(f"[DFOPS] OUTPUT: shape={output_shape}", name=f"[{self._id}]")

        return result

    def filter_rows_by_value(self, df: DataFrame) -> DataFrame:
        column = df[self.column_name]
        filter_value = self.filter_value

        # Handle regular DropdownInput format (just a string value)
        operator = getattr(self, "filter_operator", "equals")  # Default to equals for backward compatibility

        if operator == "equals":
            mask = column == filter_value
        elif operator == "not equals":
            mask = column != filter_value
        elif operator == "contains":
            mask = column.astype(str).str.contains(str(filter_value), na=False)
        elif operator == "not contains":
            mask = ~column.astype(str).str.contains(str(filter_value), na=False)
        elif operator == "starts with":
            mask = column.astype(str).str.startswith(str(filter_value), na=False)
        elif operator == "ends with":
            mask = column.astype(str).str.endswith(str(filter_value), na=False)
        elif operator == "greater than":
            try:
                # Try to convert filter_value to numeric for comparison
                numeric_value = pd.to_numeric(filter_value)
                mask = column > numeric_value
            except (ValueError, TypeError):
                # If conversion fails, compare as strings
                mask = column.astype(str) > str(filter_value)
        elif operator == "less than":
            try:
                # Try to convert filter_value to numeric for comparison
                numeric_value = pd.to_numeric(filter_value)
                mask = column < numeric_value
            except (ValueError, TypeError):
                # If conversion fails, compare as strings
                mask = column.astype(str) < str(filter_value)
        else:
            mask = column == filter_value  # Fallback to equals

        return DataFrame(df[mask])

    def sort_by_column(self, df: DataFrame) -> DataFrame:
        return DataFrame(df.sort_values(by=self.column_name, ascending=self.ascending))

    def drop_column(self, df: DataFrame) -> DataFrame:
        return DataFrame(df.drop(columns=[self.column_name]))

    def rename_column(self, df: DataFrame) -> DataFrame:
        return DataFrame(df.rename(columns={self.column_name: self.new_column_name}))

    def add_column(self, df: DataFrame) -> DataFrame:
        df[self.new_column_name] = [self.new_column_value] * len(df)
        return DataFrame(df)

    def select_columns(self, df: DataFrame) -> DataFrame:
        columns = [col.strip() for col in self.columns_to_select]
        return DataFrame(df[columns])

    def head(self, df: DataFrame) -> DataFrame:
        return DataFrame(df.head(self.num_rows))

    def tail(self, df: DataFrame) -> DataFrame:
        return DataFrame(df.tail(self.num_rows))

    def replace_values(self, df: DataFrame) -> DataFrame:
        df[self.column_name] = df[self.column_name].replace(self.replace_value, self.replacement_value)
        return DataFrame(df)

    def drop_duplicates(self, df: DataFrame) -> DataFrame:
        return DataFrame(df.drop_duplicates(subset=self.column_name))

    def transpose(self, df: DataFrame) -> DataFrame:
        """Transpose the DataFrame (swap rows and columns)."""
        # Convert to pandas DataFrame first to avoid issues with custom DataFrame class
        pandas_df = pd.DataFrame(df)
        # Transpose
        transposed = pandas_df.T
        # Reset index to make original column names a regular column
        transposed = transposed.reset_index()
        # Rename columns: first column is the original column names, rest are row values
        new_columns = ["column"] + [f"row_{i}" for i in range(len(transposed.columns) - 1)]
        transposed.columns = new_columns
        # Convert to dict and back to DataFrame to ensure proper initialization
        return DataFrame(transposed.to_dict(orient="records"))

    def reset_index(self, df: DataFrame) -> DataFrame:
        """Reset the DataFrame index to default integer index."""
        pandas_df = pd.DataFrame(df)
        result = pandas_df.reset_index(drop=self.drop_index)
        return DataFrame(result.to_dict(orient="records"))

    def set_index(self, df: DataFrame) -> DataFrame:
        """Set a column as the DataFrame index.

        Note: The column is moved to become the index. When converting back to
        Langflow DataFrame, the index is preserved as a column named after the original.
        """
        pandas_df = pd.DataFrame(df)
        result = pandas_df.set_index(self.column_name)
        # Reset index to preserve the column data when converting to dict
        result = result.reset_index()
        return DataFrame(result.to_dict(orient="records"))

    def fill_na(self, df: DataFrame) -> DataFrame:
        """Fill missing values with a specified value."""
        column_name = getattr(self, "column_name", "")
        if column_name:
            df[column_name] = df[column_name].fillna(self.fill_value)
            return DataFrame(df)
        return DataFrame(df.fillna(self.fill_value))

    def drop_na(self, df: DataFrame) -> DataFrame:
        """Drop rows with missing values."""
        column_name = getattr(self, "column_name", "")
        if column_name:
            return DataFrame(df.dropna(subset=[column_name]))
        return DataFrame(df.dropna())

    def sample_rows(self, df: DataFrame) -> DataFrame:
        """Randomly sample rows from the DataFrame."""
        n = min(self.num_rows, len(df))
        seed = getattr(self, "random_seed", None)
        random_state = seed if seed else None
        return DataFrame(df.sample(n=n, random_state=random_state))

    def slice_rows(self, df: DataFrame) -> DataFrame:
        """Get rows by index range."""
        return DataFrame(df.iloc[self.start_index : self.end_index])

    def unique_values(self, df: DataFrame) -> DataFrame:
        """Get unique values from a column as a new DataFrame."""
        unique = df[self.column_name].unique()
        return DataFrame(pd.DataFrame({self.column_name: unique}))
