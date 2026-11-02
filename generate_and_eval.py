#!/usr/bin/env python3
"""Generate and evaluate retail-command JSON extraction with Ollama.

Install runtime dependencies before submitting the PBS job:
    python3 -m pip install --user ollama codecarbon pandas scikit-learn

The script checkpoints every completed request in responses/results.json. Re-run
the same command after an interrupted job to continue from the saved results.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "test_dataset_v3.csv"
DEFAULT_RESPONSES_DIR = BASE_DIR / "responses" / "ollama"
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "ollama"

# Hugging Face counterparts are configured in train_peft.py.
DEFAULT_MODELS = (
    "llama3.2:3b",
    "llama3.1:8b",
)

TIER_COLUMNS = {
    "tier_0": "tier_0_clean",
    "tier_1": "tier_1_lexical",
    "tier_2": "tier_2_conversational",
    "tier_3": "tier_3_cognitive",
}
SCHEMA_KEYS = frozenset({"action", "product", "quantity"})
INVALID_PREDICTION_LABEL = "__invalid_prediction__"

OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "minLength": 1},
        "product": {"type": "string", "minLength": 1},
        "quantity": {"type": "integer"},
    },
    "required": ["action", "product", "quantity"],
    "additionalProperties": False,
}

PROMPT_VARIANTS = {
    "minimal": """Extract action, product, and quantity from the retail command.
Return only JSON: {"action":"","product":"","quantity":0}.""",
    "extended": """You are a retail command parser. Extract the final intended cart action,
product, and quantity from the user's command. When the user corrects an earlier
statement, use the final correction. Respond with only one JSON object that exactly
matches this schema: {"action": "", "product": "", "quantity": 0}.
Use one of add, remove, or update for action. Do not include markdown or explanation.""",
    "few_shot": """You are a retail command parser. Extract the final intended cart action,
product, and quantity from the user's command. When the user corrects an earlier
statement, use the final correction. Respond with only one JSON object that exactly
matches this schema: {"action": "", "product": "", "quantity": 0}.
Use one of add, remove, or update for action. Do not include markdown or explanation.

Examples:
User: Add two cartons of soy milk.
Output: {"action":"add","product":"soy milk","quantity":2}

User: Remove 3 apples from my cart.
Output: {"action":"remove","product":"apples","quantity":3}

User: Put 4 coffee bags in the cart, wait, make that 2.
Output: {"action":"add","product":"coffee bags","quantity":2}""",
}
DEFAULT_PROMPT_VARIANTS = ("minimal", "extended", "few_shot")

FENCED_JSON_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL
)
LOGGER = logging.getLogger("generate_and_eval")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def resolve_prompt_variants(values: Sequence[str]) -> list[str]:
    prompt_variants: list[str] = []
    for value in values:
        prompt_variants.extend(
            part.strip().replace("-", "_") for part in value.split(",") if part.strip()
        )
    prompt_variants = list(dict.fromkeys(prompt_variants))
    unknown = sorted(set(prompt_variants).difference(PROMPT_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown prompt variants: {', '.join(unknown)}.")
    if not prompt_variants:
        raise ValueError("At least one prompt variant is required.")
    return prompt_variants


def build_messages(prompt_variant: str, command: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PROMPT_VARIANTS[prompt_variant]},
        {"role": "user", "content": command},
    ]


def load_json_list(path: Path, description: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as source_file:
            value = json.load(source_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {description} at {path}: {error}") from error

    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RuntimeError(f"{description.capitalize()} at {path} must be a JSON list of objects.")
    return value


def parse_dataset_quantity(value: Any, location: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{location} must be an integer, not a boolean.")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    raise ValueError(f"{location} must be an integer; received {value!r}.")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as source_file:
                raw_rows = json.load(source_file)
        elif path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as source_file:
                raw_rows = list(csv.DictReader(source_file))
        else:
            raise ValueError("Dataset must use a .json or .csv extension.")
    except (OSError, json.JSONDecodeError, csv.Error) as error:
        raise RuntimeError(f"Could not load dataset {path}: {error}") from error

    if not isinstance(raw_rows, list):
        raise ValueError("Dataset root must be a list of command objects.")

    required_columns = {
        "id",
        "ground_truth_action",
        "ground_truth_product",
        "ground_truth_quantity",
        *TIER_COLUMNS.values(),
    }
    dataset: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row_index, raw_row in enumerate(raw_rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"Dataset row {row_index} is not an object.")

        missing_columns = required_columns.difference(raw_row)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Dataset row {row_index} is missing: {missing}.")

        command_id = raw_row["id"]
        if not isinstance(command_id, str) or not command_id.strip():
            raise ValueError(f"Dataset row {row_index} has an invalid id.")
        if command_id in seen_ids:
            raise ValueError(f"Dataset contains duplicate id {command_id!r}.")
        seen_ids.add(command_id)

        action = raw_row["ground_truth_action"]
        product = raw_row["ground_truth_product"]
        if not isinstance(action, str) or not action.strip():
            raise ValueError(f"Dataset row {row_index} has an invalid ground-truth action.")
        if not isinstance(product, str) or not product.strip():
            raise ValueError(f"Dataset row {row_index} has an invalid ground-truth product.")

        commands: dict[str, str] = {}
        for tier, column in TIER_COLUMNS.items():
            command = raw_row[column]
            if not isinstance(command, str) or not command.strip():
                raise ValueError(f"Dataset row {row_index} has an invalid {column} command.")
            commands[tier] = command

        dataset.append(
            {
                "id": command_id,
                "ground_truth": {
                    "action": action,
                    "product": product,
                    "quantity": parse_dataset_quantity(
                        raw_row["ground_truth_quantity"],
                        f"Dataset row {row_index} ground_truth_quantity",
                    ),
                },
                "commands": commands,
            }
        )

    if not dataset:
        raise ValueError("Dataset contains no command rows.")
    return dataset


def validate_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("JSON value must be an object.")

    actual_keys = set(value)
    if actual_keys != SCHEMA_KEYS:
        missing = sorted(SCHEMA_KEYS.difference(actual_keys))
        unexpected = sorted(actual_keys.difference(SCHEMA_KEYS))
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected keys: {', '.join(unexpected)}")
        raise ValueError("Schema key mismatch (" + "; ".join(details) + ").")

    action = value["action"]
    product = value["product"]
    quantity = value["quantity"]
    if not isinstance(action, str) or not action.strip():
        raise ValueError("'action' must be a non-empty string.")
    if not isinstance(product, str) or not product.strip():
        raise ValueError("'product' must be a non-empty string.")
    if not isinstance(quantity, int) or isinstance(quantity, bool):
        raise ValueError("'quantity' must be an integer.")

    return {"action": action, "product": product, "quantity": quantity}


def extract_json(text: str) -> dict[str, Any]:
    """Return a schema-valid JSON object from a plain or fenced model response.

    Markdown code fences are accepted, but prose surrounding the JSON object and
    non-schema keys are rejected so schema adherence remains meaningful.
    """

    if not isinstance(text, str):
        raise ValueError("Model response must be text.")

    candidate = text.strip()
    fenced_match = FENCED_JSON_PATTERN.match(candidate)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    else:
        candidate = re.sub(r"^\s*```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```\s*$", "", candidate).strip()

    if not candidate:
        raise ValueError("Model response is empty.")

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {error.msg}.") from error
    return validate_payload(payload)


def normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def canonical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action": normalize_text(payload["action"]),
        "product": normalize_text(payload["product"]),
        "quantity": payload["quantity"],
    }


def matches_ground_truth(prediction: Mapping[str, Any], ground_truth: Mapping[str, Any]) -> bool:
    return canonical_payload(prediction) == canonical_payload(ground_truth)


def get_response_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    return converted if converted >= 0 else None


def optional_float(value: Any, attribute_names: Sequence[str] = ()) -> float | None:
    if isinstance(value, bool) or value is None:
        return None

    for attribute_name in attribute_names:
        nested_value = getattr(value, attribute_name, None)
        if nested_value is not None and nested_value is not value:
            converted = optional_float(nested_value)
            if converted is not None:
                return converted

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def calculate_tps(eval_count: int | None, eval_duration_ns: int | None) -> float | None:
    if eval_count is None or eval_duration_ns is None or eval_duration_ns <= 0:
        return None
    return eval_count / (eval_duration_ns / 1_000_000_000)


def query_model(
    ollama_module: Any,
    model: str,
    command: str,
    retries: int,
    retry_delay: float,
    temperature: float,
    use_json_schema: bool,
    prompt_variant: str = "extended",
) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(1, retries + 2):
        try:
            request: dict[str, Any] = {
                "model": model,
                "messages": build_messages(prompt_variant, command),
                "options": {"temperature": temperature},
            }
            if use_json_schema:
                request["format"] = OUTPUT_JSON_SCHEMA

            response = ollama_module.chat(**request)
            message = get_response_value(response, "message")
            content = get_response_value(message, "content")
            if not isinstance(content, str):
                raise RuntimeError("Ollama response did not contain message.content text.")

            prompt_eval_count = optional_int(get_response_value(response, "prompt_eval_count"))
            eval_count = optional_int(get_response_value(response, "eval_count"))
            eval_duration_ns = optional_int(get_response_value(response, "eval_duration"))
            return {
                "status": "ok",
                "response": content,
                "error": None,
                "attempts": attempt,
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
                "eval_duration_ns": eval_duration_ns,
                "tokens_per_second": calculate_tps(eval_count, eval_duration_ns),
            }
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt <= retries:
                time.sleep(retry_delay * (2 ** (attempt - 1)))

    return {
        "status": "error",
        "response": None,
        "error": last_error,
        "attempts": retries + 1,
        "prompt_eval_count": None,
        "eval_count": None,
        "eval_duration_ns": None,
        "tokens_per_second": None,
    }


def make_result_record(
    model: str,
    tier: str,
    item: Mapping[str, Any],
    query_result: Mapping[str, Any],
    prompt_variant: str = "extended",
) -> dict[str, Any]:
    raw_response = query_result["response"]
    parsed_response: dict[str, Any] | None = None
    parse_error: str | None = None
    if isinstance(raw_response, str):
        try:
            parsed_response = extract_json(raw_response)
        except ValueError as error:
            parse_error = str(error)
    else:
        parse_error = query_result["error"] or "No model response was returned."

    schema_valid = parsed_response is not None
    exact_match = schema_valid and matches_ground_truth(parsed_response, item["ground_truth"])
    return {
        "id": item["id"],
        "model": model,
        "tier": tier,
        "prompt_variant": prompt_variant,
        "command": item["commands"][tier],
        "ground_truth": item["ground_truth"],
        "response": raw_response,
        "parsed_response": parsed_response,
        "schema_valid": schema_valid,
        "exact_match": exact_match,
        "updated_intent_match": exact_match if tier == "tier_3" else None,
        "status": query_result["status"],
        "error": query_result["error"],
        "parse_error": parse_error,
        "attempts": query_result["attempts"],
        "prompt_eval_count": query_result["prompt_eval_count"],
        "eval_count": query_result["eval_count"],
        "eval_duration_ns": query_result["eval_duration_ns"],
        "tokens_per_second": query_result["tokens_per_second"],
        "timestamp": utc_timestamp(),
    }


def completed_keys(results: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    for result in results:
        model = result.get("model")
        command_id = result.get("id")
        tier = result.get("tier")
        status = result.get("status", "ok")
        if (
            status == "ok"
            and isinstance(model, str)
            and isinstance(command_id, str)
            and isinstance(tier, str)
            and tier in TIER_COLUMNS
        ):
            done.add((model, command_id, tier))
    return done


def require_module(module_name: str, package_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"Missing dependency '{package_name}'. Install it with: "
            f"python3 -m pip install --user {package_name}"
        ) from error


def start_energy_tracker(results_dir: Path, model: str, tier: str) -> Any:
    codecarbon_module = require_module("codecarbon", "codecarbon")
    tracker_class = getattr(codecarbon_module, "EmissionsTracker", None)
    if tracker_class is None:
        raise RuntimeError("The installed codecarbon package does not expose EmissionsTracker.")

    results_dir.mkdir(parents=True, exist_ok=True)
    project_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"noisy_json_{model}_{tier}")
    tracker = tracker_class(
        project_name=project_name,
        output_dir=str(results_dir),
        output_file="codecarbon_emissions.csv",
        save_to_file=True,
        log_level="error",
    )
    tracker.start()
    return tracker


def stop_energy_tracker(tracker: Any) -> dict[str, float | None]:
    emissions_kg = optional_float(tracker.stop())
    emissions_data = getattr(tracker, "final_emissions_data", None)

    energy_kwh = optional_float(
        get_response_value(emissions_data, "energy_consumed"),
        attribute_names=("kWh", "value"),
    )
    if energy_kwh is None:
        energy_kwh = optional_float(
            getattr(tracker, "_total_energy", None), attribute_names=("kWh", "value")
        )

    gpu_power_w = optional_float(
        get_response_value(emissions_data, "gpu_power"),
        attribute_names=("W", "watts", "value"),
    )
    return {
        "kwh": energy_kwh,
        "gpu_power_w": gpu_power_w,
        "emissions_kg": emissions_kg,
    }


def resolve_names(values: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    for value in values:
        resolved.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(resolved))


def validate_parsed_response(value: Any) -> dict[str, Any] | None:
    try:
        return validate_payload(value)
    except ValueError:
        return None


def evaluation_records(
    results: Sequence[Mapping[str, Any]], dataset: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    items_by_id = {item["id"]: item for item in dataset}
    latest_results: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for result in results:
        model = result.get("model")
        command_id = result.get("id")
        tier = result.get("tier")
        if (
            isinstance(model, str)
            and isinstance(command_id, str)
            and isinstance(tier, str)
            and command_id in items_by_id
            and tier in TIER_COLUMNS
        ):
            latest_results[(model, command_id, tier)] = result

    evaluated: list[dict[str, Any]] = []
    for (model, command_id, tier), result in latest_results.items():
        item = items_by_id[command_id]
        parsed = validate_parsed_response(result.get("parsed_response"))
        if parsed is None and isinstance(result.get("response"), str):
            try:
                parsed = extract_json(result["response"])
            except ValueError:
                parsed = None

        schema_valid = parsed is not None
        exact_match = schema_valid and matches_ground_truth(parsed, item["ground_truth"])
        evaluated.append(
            {
                "model": model,
                "id": command_id,
                "tier": tier,
                "ground_truth": item["ground_truth"],
                "parsed_response": parsed,
                "schema_valid": schema_valid,
                "exact_match": exact_match,
                "status": result.get("status", "ok"),
                "prompt_eval_count": optional_int(result.get("prompt_eval_count")),
                "eval_count": optional_int(result.get("eval_count")),
                "eval_duration_ns": optional_int(result.get("eval_duration_ns")),
                "tokens_per_second": optional_float(result.get("tokens_per_second")),
            }
        )
    return evaluated


def aggregate_energy_runs(
    energy_runs: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, float | None]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for run in energy_runs:
        model = run.get("model")
        tier = run.get("tier")
        if isinstance(model, str) and isinstance(tier, str) and tier in TIER_COLUMNS:
            grouped[(model, tier)].append(run)

    aggregates: dict[tuple[str, str], dict[str, float | None]] = {}
    for key, runs in grouped.items():
        kwh_values = [optional_float(run.get("kwh")) for run in runs]
        emission_values = [optional_float(run.get("emissions_kg")) for run in runs]
        power_and_duration = [
            (
                optional_float(run.get("gpu_power_w")),
                optional_float(run.get("duration_seconds")),
            )
            for run in runs
        ]

        valid_kwh = [value for value in kwh_values if value is not None]
        valid_emissions = [value for value in emission_values if value is not None]
        valid_power = [
            (power, duration if duration is not None and duration > 0 else 1.0)
            for power, duration in power_and_duration
            if power is not None
        ]
        weighted_gpu_power = None
        if valid_power:
            total_duration = sum(duration for _, duration in valid_power)
            weighted_gpu_power = sum(power * duration for power, duration in valid_power) / total_duration

        aggregates[key] = {
            "total_kwh": sum(valid_kwh) if valid_kwh else None,
            "total_emissions_kg": sum(valid_emissions) if valid_emissions else None,
            "average_gpu_power_w": weighted_gpu_power,
        }
    return aggregates


def percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def field_scores(
    records: Sequence[Mapping[str, Any]], field: str, metrics_module: Any
) -> tuple[float, float, float]:
    true_values: list[str] = []
    predicted_values: list[str] = []
    for record in records:
        ground_truth = record["ground_truth"]
        true_values.append(str(canonical_payload(ground_truth)[field]))
        prediction = record["parsed_response"]
        if prediction is None:
            predicted_values.append(INVALID_PREDICTION_LABEL)
        else:
            predicted_values.append(str(canonical_payload(prediction)[field]))

    precision, recall, f1_score, _ = metrics_module.precision_recall_fscore_support(
        true_values,
        predicted_values,
        average="micro",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1_score)


def build_metric_rows(
    evaluated: Sequence[Mapping[str, Any]],
    energy_by_scope: Mapping[tuple[str, str], Mapping[str, float | None]],
    dataset_size: int,
) -> list[dict[str, Any]]:
    metrics_module = require_module("sklearn.metrics", "scikit-learn")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in evaluated:
        grouped[(record["model"], record["tier"])].append(record)

    rows: list[dict[str, Any]] = []
    tier_order = {tier: index for index, tier in enumerate(TIER_COLUMNS)}
    for (model, tier), records in sorted(
        grouped.items(), key=lambda entry: (entry[0][0], tier_order[entry[0][1]])
    ):
        total_records = len(records)
        schema_valid_count = sum(record["schema_valid"] for record in records)
        exact_match_count = sum(record["exact_match"] for record in records)
        failed_request_count = sum(record["status"] != "ok" for record in records)

        action_precision, action_recall, action_f1 = field_scores(records, "action", metrics_module)
        product_precision, product_recall, product_f1 = field_scores(records, "product", metrics_module)
        quantity_precision, quantity_recall, quantity_f1 = field_scores(records, "quantity", metrics_module)

        input_tokens = [record["prompt_eval_count"] for record in records]
        output_tokens = [record["eval_count"] for record in records]
        tps_values = [record["tokens_per_second"] for record in records]
        valid_input_tokens = [value for value in input_tokens if value is not None]
        valid_output_tokens = [value for value in output_tokens if value is not None]
        valid_tps = [value for value in tps_values if value is not None]
        timed_token_pairs = [
            (record["eval_count"], record["eval_duration_ns"])
            for record in records
            if record["eval_count"] is not None
            and record["eval_duration_ns"] is not None
            and record["eval_duration_ns"] > 0
        ]

        aggregate_tps = None
        if timed_token_pairs:
            aggregate_tps = sum(tokens for tokens, _ in timed_token_pairs) / (
                sum(duration for _, duration in timed_token_pairs) / 1_000_000_000
            )

        energy = energy_by_scope.get((model, tier), {})
        rows.append(
            {
                "model": model,
                "tier": tier,
                "records": total_records,
                "coverage_pct": percentage(total_records, dataset_size),
                "failed_requests": failed_request_count,
                "schema_adherence_pct": percentage(schema_valid_count, total_records),
                "emr_pct": percentage(exact_match_count, total_records),
                "action_precision": action_precision,
                "action_recall": action_recall,
                "action_f1": action_f1,
                "product_precision": product_precision,
                "product_recall": product_recall,
                "product_f1": product_f1,
                "quantity_precision": quantity_precision,
                "quantity_recall": quantity_recall,
                "quantity_f1": quantity_f1,
                "tier3_correction_accuracy_pct": (
                    percentage(exact_match_count, total_records) if tier == "tier_3" else None
                ),
                "mean_input_tokens": (
                    sum(valid_input_tokens) / len(valid_input_tokens) if valid_input_tokens else None
                ),
                "mean_output_tokens": (
                    sum(valid_output_tokens) / len(valid_output_tokens) if valid_output_tokens else None
                ),
                "aggregate_tps": aggregate_tps,
                "mean_tps": sum(valid_tps) / len(valid_tps) if valid_tps else None,
                "total_kwh": energy.get("total_kwh"),
                "average_gpu_power_w": energy.get("average_gpu_power_w"),
                "total_emissions_kg": energy.get("total_emissions_kg"),
            }
        )
    return rows


def write_metrics_table(rows: Sequence[Mapping[str, Any]], results_dir: Path) -> None:
    pandas_module = require_module("pandas", "pandas")
    dataframe = pandas_module.DataFrame(rows)
    csv_path = results_dir / "mdpi_metrics_table.csv"
    latex_path = results_dir / "mdpi_metrics_table.tex"
    json_path = results_dir / "mdpi_metrics_table.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_content = dataframe.to_csv(index=False)
    latex_content = dataframe.to_latex(
        index=False,
        escape=True,
        na_rep="--",
        float_format=lambda value: f"{value:.4f}",
    )
    atomic_write_text(csv_path, csv_content)
    atomic_write_text(latex_path, latex_content)
    atomic_write_json(json_path, list(rows))


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate strict retail JSON with Ollama and export MDPI evaluation metrics."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model names, separated by spaces or commas.",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=list(TIER_COLUMNS),
        help="Any of: tier_0, tier_1, tier_2, tier_3. Accepts spaces or commas.",
    )
    parser.add_argument(
        "--prompt-variants",
        nargs="+",
        default=list(DEFAULT_PROMPT_VARIANTS),
        help="Any of: minimal, extended, few_shot. Accepts spaces or commas.",
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N dataset rows.")
    parser.add_argument("--retries", type=int, default=2, help="Retries after the initial request.")
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--request-delay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--no-json-schema-format",
        dest="use_json_schema",
        action="store_false",
        help="Do not send Ollama's JSON Schema format argument; parsing remains strict.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


def run_prompt_variant(
    args: argparse.Namespace,
    dataset: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    tiers: Sequence[str],
    prompt_variant: str,
    ollama_module: Any,
) -> bool:
    responses_file = args.responses_dir / prompt_variant / "results.json"
    results_dir = args.results_dir / prompt_variant
    energy_path = results_dir / "energy_runs.json"
    results = load_json_list(responses_file, f"{prompt_variant} response checkpoint")
    energy_runs = load_json_list(energy_path, f"{prompt_variant} energy checkpoint")
    done = completed_keys(results)
    LOGGER.info(
        "%s: loaded %d saved responses and %d completed checkpoints.",
        prompt_variant,
        len(results),
        len(done),
    )

    for model in models:
        for tier in tiers:
            pending_items = [item for item in dataset if (model, item["id"], tier) not in done]
            if not pending_items:
                LOGGER.info("%s %s %s: already complete.", prompt_variant, model, tier)
                continue

            LOGGER.info(
                "%s %s %s: processing %d commands.",
                prompt_variant,
                model,
                tier,
                len(pending_items),
            )
            tracker = start_energy_tracker(results_dir, model, tier)
            scope_started_at = utc_timestamp()
            scope_started = time.monotonic()
            completed_in_scope = 0
            try:
                for index, item in enumerate(pending_items, start=1):
                    query_result = query_model(
                        ollama_module=ollama_module,
                        model=model,
                        command=item["commands"][tier],
                        retries=args.retries,
                        retry_delay=args.retry_delay,
                        temperature=args.temperature,
                        use_json_schema=args.use_json_schema,
                        prompt_variant=prompt_variant,
                    )
                    record = make_result_record(
                        model,
                        tier,
                        item,
                        query_result,
                        prompt_variant,
                    )
                    results.append(record)
                    atomic_write_json(responses_file, results)
                    completed_in_scope += 1

                    if record["status"] == "ok":
                        done.add((model, item["id"], tier))
                        LOGGER.info(
                            "%s %s %s %s (%d/%d): schema=%s exact=%s tps=%s",
                            prompt_variant,
                            model,
                            tier,
                            item["id"],
                            index,
                            len(pending_items),
                            record["schema_valid"],
                            record["exact_match"],
                            f"{record['tokens_per_second']:.2f}"
                            if record["tokens_per_second"] is not None
                            else "n/a",
                        )
                    else:
                        LOGGER.error(
                            "%s %s %s %s (%d/%d): %s",
                            prompt_variant,
                            model,
                            tier,
                            item["id"],
                            index,
                            len(pending_items),
                            record["error"],
                        )

                    if args.request_delay and index < len(pending_items):
                        time.sleep(args.request_delay)
            finally:
                energy_record: dict[str, Any] = {
                    "model": model,
                    "tier": tier,
                    "started_at": scope_started_at,
                    "finished_at": utc_timestamp(),
                    "duration_seconds": time.monotonic() - scope_started,
                    "completed_requests": completed_in_scope,
                }
                try:
                    energy_record.update(stop_energy_tracker(tracker))
                except Exception as error:
                    energy_record.update(
                        {
                            "kwh": None,
                            "gpu_power_w": None,
                            "emissions_kg": None,
                            "tracker_error": f"{type(error).__name__}: {error}",
                        }
                    )
                    LOGGER.exception(
                        "Could not stop CodeCarbon tracking for %s %s %s.",
                        prompt_variant,
                        model,
                        tier,
                    )
                energy_runs.append(energy_record)
                atomic_write_json(energy_path, energy_runs)

    evaluated = evaluation_records(results, dataset)
    metric_rows = build_metric_rows(evaluated, aggregate_energy_runs(energy_runs), len(dataset))
    for row in metric_rows:
        row["prompt_variant"] = prompt_variant
    write_metrics_table(metric_rows, results_dir)
    LOGGER.info("%s: wrote %d metric rows to %s.", prompt_variant, len(metric_rows), results_dir)
    return any(record["status"] != "ok" for record in evaluated)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative.")
    if args.retry_delay < 0 or args.request_delay < 0:
        raise ValueError("Retry and request delays cannot be negative.")

    models = resolve_names(args.models)
    tiers = resolve_names(args.tiers)
    prompt_variants = resolve_prompt_variants(args.prompt_variants)
    if not models:
        raise ValueError("At least one model is required.")
    invalid_tiers = sorted(set(tiers).difference(TIER_COLUMNS))
    if invalid_tiers:
        raise ValueError(f"Unknown tiers: {', '.join(invalid_tiers)}.")
    if not tiers:
        raise ValueError("At least one tier is required.")

    dataset = load_dataset(args.dataset)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    ollama_module = require_module("ollama", "ollama")
    require_module("codecarbon", "codecarbon")
    require_module("pandas", "pandas")
    require_module("sklearn.metrics", "scikit-learn")

    any_failed_requests = False
    for prompt_variant in prompt_variants:
        any_failed_requests = run_prompt_variant(
            args,
            dataset,
            models,
            tiers,
            prompt_variant,
            ollama_module,
        ) or any_failed_requests
    return 1 if any_failed_requests else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted. Saved checkpoints can be resumed by rerunning this script.")
        raise SystemExit(130)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(2)