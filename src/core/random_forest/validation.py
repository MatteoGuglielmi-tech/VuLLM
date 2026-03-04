from pydantic import BaseModel, Field, ValidationError
from pathlib import Path
import json
from typing import Literal
from collections import defaultdict


class DatasetExample(BaseModel):
    """Dataset example schema with validation."""

    func: str = Field(..., min_length=1, description="Function code")
    target: Literal[0, 1] = Field(
        ..., description="Binary target (0=safe, 1=vulnerable)"
    )
    project: str = Field(..., min_length=1, description="Project name")
    reasoning: str = Field(..., min_length=1, description="Reasoning text")
    cwe: list[str] = Field(default_factory=list, description="CWE IDs")
    cwe_desc: list[str] = Field(default_factory=list, description="CWE descriptions")

    class Config:
        # Allow extra fields if present in JSONL
        extra = "forbid"  # Change to "allow" if you want to permit extra fields


class ValidationResult(BaseModel):
    """Results of JSONL validation."""

    total_lines: int
    valid_lines: int
    invalid_lines: int
    errors: list[dict[str, str | int]]

    @property
    def is_valid(self) -> bool:
        """Check if all lines are valid."""
        return self.invalid_lines == 0

    def print_summary(self) -> None:
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total lines: {self.total_lines}")
        print(f"Valid lines: {self.valid_lines} ✅")
        print(f"Invalid lines: {self.invalid_lines} ❌")

        if self.errors:
            print("\n" + "=" * 70)
            print("ERRORS:")
            print("=" * 70)
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"\nLine {error['line']}:")
                print(f"  Error: {error['error']}")
                if "data" in error:
                    print(f"  Data: {error['data']}")

            if len(self.errors) > 10:
                print(f"\n... and {len(self.errors) - 10} more errors")

        print("\n" + "=" * 70)


def validate_jsonl(file_path: Path, strict: bool = True) -> ValidationResult:
    """
    Validate JSONL file using Pydantic model.

    Parameters
    ----------
    file_path : Path
        Path to JSONL file
    strict : bool, default=True
        If True, raise exception on first error. If False, collect all errors.

    Returns
    -------
    ValidationResult
        Validation results with detailed error information

    Raises
    ------
    ValueError
        If strict=True and validation fails
    """
    errors = []
    valid_count = 0
    total_count = 0

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            total_count += 1

            try:
                # Parse & Validate against Pydantic model
                DatasetExample.model_validate_json(line)
                valid_count += 1

            except json.JSONDecodeError as e:
                error_info = {
                    "line": line_num,
                    "error": f"Invalid JSON: {str(e)}",
                    "data": line.strip()[:100],  # First 100 chars
                }
                errors.append(error_info)

                if strict:
                    raise ValueError(f"Line {line_num}: Invalid JSON - {e}") from e

            except ValidationError as e:
                # Collect all validation errors for this line
                error_messages = []
                for error in e.errors():
                    field = " -> ".join(str(loc) for loc in error["loc"])
                    msg = error["msg"]
                    error_messages.append(f"{field}: {msg}")

                error_info = {
                    "line": line_num,
                    "error": "; ".join(error_messages),
                    "data": line[:100],
                }
                errors.append(error_info)

                if strict:
                    raise ValueError(
                        f"Line {line_num}: Validation failed - {'; '.join(error_messages)}"
                    ) from e

    result = ValidationResult(
        total_lines=total_count,
        valid_lines=valid_count,
        invalid_lines=len(errors),
        errors=errors,
    )

    return result


def validate_and_load_jsonl(
    file_path: Path, strict: bool = False
) -> list[DatasetExample]:
    """
    Validate and load JSONL file, returning only valid examples.

    Parameters
    ----------
    file_path : Path
        Path to JSONL file
    strict : bool, default=False
        If True, raise exception on any invalid line

    Returns
    -------
    list[DatasetExample]
        List of validated examples
    """
    valid_examples = []

    validation_result = validate_jsonl(file_path, strict=strict)

    if not strict:
        with open(file=file_path, mode="r") as f:
            for _, line in enumerate(f, 1):
                try:
                    example = DatasetExample.model_validate_json(line)
                    valid_examples.append(example)
                except (json.JSONDecodeError, ValidationError):
                    continue
    else:
        # All lines should be valid if we got here
        with open(file=file_path, mode="r") as f:
            for line in f:
                data = json.loads(line)
                example = DatasetExample(**data)
                valid_examples.append(example)

    validation_result.print_summary()

    return valid_examples


def inspect_jsonl_schema_pydantic(file_path: Path) -> dict[str, set[str]]:
    """
    Inspect JSONL schema and report type inconsistencies.

    Uses Pydantic validation to identify issues.

    Purpose: Discover what types actually exist in the file (even if invalid)

    Parameters
    ----------
    file_path : Path
        Path to JSONL file

    Returns
    -------
    dict[str, set[str]]
        Field names mapped to observed types
    """
    field_types = defaultdict(set)
    field_errors = defaultdict(list)

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)

                # Try Pydantic validation
                try:
                    DatasetExample.model_validate(data)

                    # Track successful types
                    for field, value in data.items():
                        field_types[field].add(type(value).__name__)

                except ValidationError as e:
                    # Track validation errors
                    for error in e.errors():
                        field = " -> ".join(str(loc) for loc in error["loc"])
                        field_errors[field].append(
                            {
                                "line": line_num,
                                "error": error["msg"],
                                "type": error["type"],
                            }
                        )

                        # Also track the actual type
                        if error["loc"]:
                            field_name = str(error["loc"][0])
                            if field_name in data:
                                field_types[field_name].add(
                                    type(data[field_name]).__name__
                                )

            except json.JSONDecodeError as e:
                print(f"Line {line_num}: Invalid JSON - {e}")

    # Print results
    print("\n" + "=" * 70)
    print("SCHEMA ANALYSIS (Pydantic Validation)")
    print("=" * 70)

    # Print expected schema
    print("\n📋 Expected Schema:")
    for field_name, field_info in DatasetExample.model_fields.items():
        annotation = field_info.annotation
        required = field_info.is_required()
        print(
            f"  {field_name}: {annotation} {'(required)' if required else '(optional)'}"
        )

    # Print observed types
    print("\n📊 Observed Types:")
    for field, types in sorted(field_types.items()):
        type_str = ", ".join(sorted(types))
        if len(types) > 1:
            print(f"  ⚠️  {field}: {type_str} (INCONSISTENT)")
        else:
            print(f"  ✓  {field}: {type_str}")

    # Print validation errors
    if field_errors:
        print("\n❌ Validation Errors:")
        for field, errors in sorted(field_errors.items()):
            print(f"\n  {field}:")
            for error in errors[:5]:  # Show first 5 per field
                print(f"    Line {error['line']}: {error['error']} ({error['type']})")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more errors")
    else:
        print("\n✅ All fields validate successfully")

    print("\n" + "=" * 70)

    return dict(field_types)
