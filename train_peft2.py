#!/usr/bin/env python3
"""Train a 4-bit LoRA or IA3 adapter and evaluate noisy retail JSON parsing.

Configure MODEL_NAMES and ADAPTER_TYPES below, then run this script directly or
through train_peft_job.pbs. Training, validation, and test CSVs are configured
separately below. Each experiment combination is isolated in its own directory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import logging
import math
import os
import re
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
TRAIN_DATASET_PATH = BASE_DIR / "train_dataset_v3.csv"
VALIDATION_DATASET_PATH = BASE_DIR / "val_dataset_v3.csv"
TEST_DATASET_PATH = BASE_DIR / "test_dataset_v3.csv"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
# Hugging Face baseline counterparts are configured in generate_and_eval_hf.py.
MODEL_NAMES = (
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
)
ADAPTER_TYPES = (
    # "lora",
    "ia3",
)
ADAPTER_OUTPUT_ROOT = BASE_DIR / "adapters"
PEFT_RESULTS_ROOT = DEFAULT_RESULTS_DIR / "peft"
TIER_COLUMNS = {
    "tier_0": "tier_0_clean",
    "tier_1": "tier_1_lexical",
    "tier_2": "tier_2_conversational",
    "tier_3": "tier_3_cognitive",
}
SCHEMA_KEYS = frozenset({"action", "product", "quantity"})
INVALID_FIELD_VALUE = "__invalid_json__"
FENCED_JSON_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL
)
LOGGER = logging.getLogger("train_peft")

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


def parse_quantity(value: Any, location: str) -> int:
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
    seen_ids: set[str] = set()
    commands: list[dict[str, Any]] = []
    for row_number, row in enumerate(raw_rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"Dataset row {row_number} is not an object.")
        missing = sorted(required_columns.difference(row))
        if missing:
            raise ValueError(f"Dataset row {row_number} is missing: {', '.join(missing)}.")

        command_id = row["id"]
        if not isinstance(command_id, str) or not command_id.strip():
            raise ValueError(f"Dataset row {row_number} has an invalid id.")
        if command_id in seen_ids:
            raise ValueError(f"Dataset contains duplicate id {command_id!r}.")
        seen_ids.add(command_id)

        action = row["ground_truth_action"]
        product = row["ground_truth_product"]
        if not isinstance(action, str) or not action.strip():
            raise ValueError(f"Dataset row {row_number} has an invalid action.")
        if not isinstance(product, str) or not product.strip():
            raise ValueError(f"Dataset row {row_number} has an invalid product.")

        tier_commands: dict[str, str] = {}
        for tier, column in TIER_COLUMNS.items():
            command = row[column]
            if not isinstance(command, str) or not command.strip():
                raise ValueError(f"Dataset row {row_number} has an invalid {column} command.")
            tier_commands[tier] = command

        commands.append(
            {
                "id": command_id,
                "ground_truth": {
                    "action": action,
                    "product": product,
                    "quantity": parse_quantity(
                        row["ground_truth_quantity"],
                        f"Dataset row {row_number} ground_truth_quantity",
                    ),
                },
                "commands": tier_commands,
            }
        )

    if len(commands) < 2:
        raise ValueError("At least two command IDs are required for a train/test split.")
    return commands


def resolve_names(values: Sequence[str]) -> list[str]:
    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(names))


def validate_tiers(tiers: Sequence[str], argument_name: str) -> list[str]:
    resolved = resolve_names(tiers)
    unknown = sorted(set(resolved).difference(TIER_COLUMNS))
    if unknown:
        raise ValueError(f"{argument_name} contains unknown tiers: {', '.join(unknown)}.")
    if not resolved:
        raise ValueError(f"{argument_name} must include at least one tier.")
    return resolved


def validate_dataset_splits(
    train_commands: Sequence[Mapping[str, Any]],
    validation_commands: Sequence[Mapping[str, Any]],
    test_commands: Sequence[Mapping[str, Any]],
) -> None:
    split_ids = {
        "train": {command["id"] for command in train_commands},
        "validation": {command["id"] for command in validation_commands},
        "test": {command["id"] for command in test_commands},
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_ids[left].intersection(split_ids[right])
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"Dataset ID leakage between {left} and {right}: {examples}.")


def expand_examples(
    commands: Sequence[Mapping[str, Any]], tiers: Sequence[str]
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for command in commands:
        for tier in tiers:
            examples.append(
                {
                    "id": command["id"],
                    "tier": tier,
                    "input": command["commands"][tier],
                    "ground_truth": dict(command["ground_truth"]),
                }
            )
    return examples


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "action": payload["action"],
            "product": payload["product"],
            "quantity": payload["quantity"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def fallback_training_text(
    command: str,
    target: Mapping[str, Any],
    prompt_variant: str,
) -> str:
    return f"{PROMPT_VARIANTS[prompt_variant]}\n\nUser: {command}\n\nAssistant: {canonical_json(target)}"


def format_training_text(
    example: Mapping[str, Any],
    tokenizer: Any,
    prompt_variant: str,
) -> str:
    messages = [
        {"role": "system", "content": PROMPT_VARIANTS[prompt_variant]},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": canonical_json(example["ground_truth"])},
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except (AttributeError, TypeError, ValueError) as error:
            LOGGER.warning("Chat template failed during formatting; using fallback: %s", error)
    return fallback_training_text(example["input"], example["ground_truth"], prompt_variant)


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


def get_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


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


def load_dependencies() -> SimpleNamespace:
    missing: list[str] = []
    required_modules = {
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
        "codecarbon": "codecarbon",
        "datasets": "datasets",
        "peft": "peft",
        "sklearn.metrics": "scikit-learn",
        "torch": "torch",
        "transformers": "transformers",
        "trl": "trl",
        "pandas": "pandas",
    }
    for module_name, package_name in required_modules.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        packages = " ".join(sorted(set(missing)))
        raise RuntimeError(
            "Missing training dependencies: "
            + ", ".join(sorted(set(missing)))
            + f". Install them with: python3 -m pip install --user {packages}"
        )

    import pandas
    import torch
    from codecarbon import EmissionsTracker
    from datasets import Dataset
    from peft import IA3Config, LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from sklearn import metrics as sklearn_metrics
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    return SimpleNamespace(
        torch=torch,
        pandas=pandas,
        Dataset=Dataset,
        EmissionsTracker=EmissionsTracker,
        IA3Config=IA3Config,
        LoraConfig=LoraConfig,
        PeftModel=PeftModel,
        TaskType=TaskType,
        get_peft_model=get_peft_model,
        prepare_model_for_kbit_training=prepare_model_for_kbit_training,
        metrics=sklearn_metrics,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        BitsAndBytesConfig=BitsAndBytesConfig,
        get_last_checkpoint=get_last_checkpoint,
        SFTConfig=SFTConfig,
        SFTTrainer=SFTTrainer,
    )


def select_compute_dtype(torch_module: Any) -> Any:
    return torch_module.bfloat16 if torch_module.cuda.is_bf16_supported() else torch_module.float16


def configure_tokenizer(model_name: str, args: argparse.Namespace, dependencies: SimpleNamespace) -> Any:
    tokenizer = dependencies.AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise RuntimeError("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_peft_config(args: argparse.Namespace, dependencies: SimpleNamespace) -> Any:
    if args.adapter_type == "lora":
        return dependencies.LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=dependencies.TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
    return dependencies.IA3Config(
        task_type=dependencies.TaskType.CAUSAL_LM,
        target_modules=["k_proj", "v_proj", "down_proj"],
        feedforward_modules=["down_proj"],
        inference_mode=False,
    )


def resolve_resume_checkpoint(args: argparse.Namespace, dependencies: SimpleNamespace) -> str | None:
    if args.resume_from_checkpoint is None:
        return None
    if args.resume_from_checkpoint == "auto":
        checkpoint = dependencies.get_last_checkpoint(str(args.output_dir))
        if checkpoint is None:
            LOGGER.warning("No checkpoint found in %s; starting a new run.", args.output_dir)
        return checkpoint
    checkpoint = Path(args.resume_from_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return str(checkpoint)


def train_adapter(
    args: argparse.Namespace,
    dependencies: SimpleNamespace,
    train_examples: Sequence[Mapping[str, Any]],
    validation_examples: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, dict[str, Any]]:
    torch_module = dependencies.torch
    compute_dtype = select_compute_dtype(torch_module)
    tokenizer = configure_tokenizer(args.model_name, args, dependencies)
    training_records = [
        {"text": format_training_text(example, tokenizer, args.prompt_variant)}
        for example in train_examples
    ]
    train_dataset = dependencies.Dataset.from_list(training_records)
    validation_records = [
        {"text": format_training_text(example, tokenizer, args.prompt_variant)}
        for example in validation_examples
    ]
    validation_dataset = dependencies.Dataset.from_list(validation_records)

    quantization_config = dependencies.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = dependencies.AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map={"": 0},
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = dependencies.prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    model = dependencies.get_peft_model(model, build_peft_config(args, dependencies))

    if not isinstance(model, dependencies.PeftModel):
        raise RuntimeError("PEFT did not return a PeftModel instance.")
    model.print_trainable_parameters()

    use_bf16 = compute_dtype == torch_module.bfloat16
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 2e-4 if args.adapter_type == "lora" else 1e-3

    training_config = dependencies.SFTConfig(
        output_dir=str(args.output_dir),
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=args.num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=args.report_to,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
    )
    trainer = dependencies.SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        args=training_config,
    )

    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats()
    training_started_at = time.time()
    resume_checkpoint = resolve_resume_checkpoint(args, dependencies)
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    training_seconds = time.time() - training_started_at
    peak_training_vram_gb = torch_module.cuda.max_memory_allocated() / 1024**3

    adapter_dir = args.output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()

    training_summary = {
        "created_at": utc_timestamp(),
        "model_name": args.model_name,
        "adapter_type": args.adapter_type,
        "prompt_variant": args.prompt_variant,
        "adapter_path": str(adapter_dir),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "training_seconds": training_seconds,
        "peak_training_vram_gb": peak_training_vram_gb,
        "train_metrics": train_result.metrics,
        "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
        "compute_dtype": str(compute_dtype),
    }
    atomic_write_json(args.output_dir / "training_summary.json", training_summary)
    return trainer.model, tokenizer, training_summary


def build_generation_inputs(
    tokenizer: Any,
    command: str,
    max_seq_length: int,
    prompt_variant: str,
) -> Mapping[str, Any]:
    messages = [
        {"role": "system", "content": PROMPT_VARIANTS[prompt_variant]},
        {"role": "user", "content": command},
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=max_seq_length,
                return_dict=True,
                return_tensors="pt",
            )
        except (AttributeError, TypeError, ValueError) as error:
            LOGGER.warning("Chat template failed during generation; using fallback: %s", error)
    prompt = f"{PROMPT_VARIANTS[prompt_variant]}\n\nUser: {command}\n\nAssistant:"
    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )


def model_input_device(model: Any) -> Any:
    return next(model.parameters()).device


def generate_prediction(
    model: Any,
    tokenizer: Any,
    command: str,
    args: argparse.Namespace,
    torch_module: Any,
) -> dict[str, Any]:
    try:
        encoded = build_generation_inputs(
            tokenizer,
            command,
            args.max_seq_length,
            args.prompt_variant,
        )
        device = model_input_device(model)
        encoded = {name: value.to(device) for name, value in encoded.items()}
        input_token_count = int(encoded["input_ids"].shape[-1])

        torch_module.cuda.synchronize()
        started_at = time.time()
        with torch_module.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        torch_module.cuda.synchronize()
        latency_seconds = time.time() - started_at
        generated_tokens = generated[0, input_token_count:]
        output_token_count = int(generated_tokens.shape[-1])
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        tokens_per_second = output_token_count / latency_seconds if latency_seconds > 0 else None
        try:
            parsed_response = extract_json(response)
            parse_error = None
        except ValueError as error:
            parsed_response = None
            parse_error = str(error)
        return {
            "response": response,
            "parsed_response": parsed_response,
            "parse_error": parse_error,
            "generation_error": None,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "latency_seconds": latency_seconds,
            "tokens_per_second": tokens_per_second,
        }
    except Exception as error:
        torch_module.cuda.empty_cache()
        return {
            "response": None,
            "parsed_response": None,
            "parse_error": None,
            "generation_error": f"{type(error).__name__}: {error}",
            "input_token_count": None,
            "output_token_count": None,
            "latency_seconds": None,
            "tokens_per_second": None,
        }


def start_energy_tracker(
    dependencies: SimpleNamespace,
    results_dir: Path,
    model_name: str,
    adapter_type: str,
    tier: str,
) -> tuple[Any | None, str | None]:
    project_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"peft_{adapter_type}_{model_name}_{tier}",
    )
    try:
        tracker = dependencies.EmissionsTracker(
            project_name=project_name,
            output_dir=str(results_dir),
            output_file=f"codecarbon_{tier}.csv",
            save_to_file=True,
            log_level="error",
        )
        tracker.start()
        return tracker, None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def stop_energy_tracker(tracker: Any | None, tracker_error: str | None) -> dict[str, Any]:
    if tracker is None:
        return {
            "energy_kwh": None,
            "emissions_kg": None,
            "average_gpu_power_w": None,
            "tracker_error": tracker_error,
        }
    try:
        emissions_kg = optional_float(tracker.stop())
        emissions_data = getattr(tracker, "final_emissions_data", None)
        energy_kwh = optional_float(
            get_value(emissions_data, "energy_consumed"),
            attribute_names=("kWh", "value"),
        )
        gpu_power_w = optional_float(
            get_value(emissions_data, "gpu_power"),
            attribute_names=("W", "watts", "value"),
        )
        return {
            "energy_kwh": energy_kwh,
            "emissions_kg": emissions_kg,
            "average_gpu_power_w": gpu_power_w,
            "tracker_error": None,
        }
    except Exception as error:
        return {
            "energy_kwh": None,
            "emissions_kg": None,
            "average_gpu_power_w": None,
            "tracker_error": f"{type(error).__name__}: {error}",
        }


def evaluate_adapter(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    dependencies: SimpleNamespace,
    test_examples: Sequence[Mapping[str, Any]],
    eval_tiers: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], float]:
    torch_module = dependencies.torch
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats()

    predictions_path = args.results_dir / "peft_predictions.json"
    all_predictions: list[dict[str, Any]] = []
    records_by_tier: dict[str, list[dict[str, Any]]] = {}
    energy_by_tier: dict[str, dict[str, Any]] = {}

    for tier in eval_tiers:
        tier_examples = [example for example in test_examples if example["tier"] == tier]
        tracker, tracker_error = start_energy_tracker(
            dependencies,
            args.results_dir,
            args.model_name,
            args.adapter_type,
            tier,
        )
        tier_started_at = time.time()
        tier_records: list[dict[str, Any]] = []
        try:
            for index, example in enumerate(tier_examples, start=1):
                prediction = generate_prediction(
                    model,
                    tokenizer,
                    example["input"],
                    args,
                    torch_module,
                )
                parsed_response = prediction["parsed_response"]
                schema_valid = parsed_response is not None
                exact_match = schema_valid and matches_ground_truth(
                    parsed_response, example["ground_truth"]
                )
                record = {
                    "id": example["id"],
                    "tier": tier,
                    "command": example["input"],
                    "ground_truth": example["ground_truth"],
                    "response": prediction["response"],
                    "parsed_response": parsed_response,
                    "schema_valid": schema_valid,
                    "exact_match": exact_match,
                    "updated_intent_match": exact_match if tier == "tier_3" else None,
                    "parse_error": prediction["parse_error"],
                    "generation_error": prediction["generation_error"],
                    "input_token_count": prediction["input_token_count"],
                    "output_token_count": prediction["output_token_count"],
                    "latency_seconds": prediction["latency_seconds"],
                    "tokens_per_second": prediction["tokens_per_second"],
                    "timestamp": utc_timestamp(),
                }
                tier_records.append(record)
                all_predictions.append(record)
                atomic_write_json(predictions_path, all_predictions)
                if index % args.progress_every == 0 or index == len(tier_examples):
                    LOGGER.info(
                        "%s %d/%d schema=%s exact=%s",
                        tier,
                        index,
                        len(tier_examples),
                        schema_valid,
                        exact_match,
                    )
        finally:
            energy = stop_energy_tracker(tracker, tracker_error)
            energy["inference_wall_seconds"] = time.time() - tier_started_at
            energy_by_tier[tier] = energy
            records_by_tier[tier] = tier_records

    peak_inference_vram_gb = torch_module.cuda.max_memory_allocated() / 1024**3
    return records_by_tier, energy_by_tier, peak_inference_vram_gb


def field_scores(
    records: Sequence[Mapping[str, Any]],
    field: str,
    metrics_module: Any,
) -> tuple[float, float, float]:
    truth: list[str] = []
    predictions: list[str] = []
    for record in records:
        truth.append(str(canonical_payload(record["ground_truth"])[field]))
        prediction = record["parsed_response"]
        predictions.append(
            str(canonical_payload(prediction)[field])
            if prediction is not None
            else INVALID_FIELD_VALUE
        )
    precision, recall, f1_score, _ = metrics_module.precision_recall_fscore_support(
        truth,
        predictions,
        average="micro",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1_score)


def percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def aggregate_energy(energy_by_tier: Mapping[str, Mapping[str, Any]]) -> dict[str, float | None]:
    kwh_values = [optional_float(energy.get("energy_kwh")) for energy in energy_by_tier.values()]
    emissions_values = [
        optional_float(energy.get("emissions_kg")) for energy in energy_by_tier.values()
    ]
    power_duration_pairs = [
        (
            optional_float(energy.get("average_gpu_power_w")),
            optional_float(energy.get("inference_wall_seconds")),
        )
        for energy in energy_by_tier.values()
    ]
    valid_kwh = [value for value in kwh_values if value is not None]
    valid_emissions = [value for value in emissions_values if value is not None]
    valid_power = [
        (power, duration if duration is not None and duration > 0 else 1.0)
        for power, duration in power_duration_pairs
        if power is not None
    ]
    average_power = None
    if valid_power:
        total_duration = sum(duration for _, duration in valid_power)
        average_power = sum(power * duration for power, duration in valid_power) / total_duration
    return {
        "energy_kwh": sum(valid_kwh) if valid_kwh else None,
        "emissions_kg": sum(valid_emissions) if valid_emissions else None,
        "average_gpu_power_w": average_power,
    }


def metric_row(
    records: Sequence[Mapping[str, Any]],
    tier: str,
    args: argparse.Namespace,
    metrics_module: Any,
    energy: Mapping[str, Any],
    peak_training_vram_gb: float,
    peak_inference_vram_gb: float,
) -> dict[str, Any]:
    total = len(records)
    schema_valid = sum(record["schema_valid"] for record in records)
    exact_match = sum(record["exact_match"] for record in records)
    generation_errors = sum(record["generation_error"] is not None for record in records)
    action_precision, action_recall, action_f1 = field_scores(records, "action", metrics_module)
    product_precision, product_recall, product_f1 = field_scores(records, "product", metrics_module)
    quantity_precision, quantity_recall, quantity_f1 = field_scores(records, "quantity", metrics_module)

    input_tokens = [record["input_token_count"] for record in records]
    output_tokens = [record["output_token_count"] for record in records]
    latencies = [record["latency_seconds"] for record in records]
    tps_values = [record["tokens_per_second"] for record in records]
    valid_input_tokens = [value for value in input_tokens if value is not None]
    valid_output_tokens = [value for value in output_tokens if value is not None]
    valid_latencies = [value for value in latencies if value is not None and value > 0]
    valid_tps = [value for value in tps_values if value is not None]
    aggregate_tps = None
    if valid_output_tokens and valid_latencies:
        aggregate_tps = sum(valid_output_tokens) / sum(valid_latencies)

    return {
        "model_name": args.model_name,
        "adapter_type": args.adapter_type,
        "prompt_variant": args.prompt_variant,
        "tier": tier,
        "examples": total,
        "generation_errors": generation_errors,
        "schema_adherence_pct": percentage(schema_valid, total),
        "emr_pct": percentage(exact_match, total),
        "action_precision": action_precision,
        "action_recall": action_recall,
        "action_f1": action_f1,
        "product_precision": product_precision,
        "product_recall": product_recall,
        "product_f1": product_f1,
        "quantity_precision": quantity_precision,
        "quantity_recall": quantity_recall,
        "quantity_f1": quantity_f1,
        "tier3_correction_accuracy_pct": percentage(exact_match, total) if tier == "tier_3" else None,
        "mean_input_tokens": (
            sum(valid_input_tokens) / len(valid_input_tokens) if valid_input_tokens else None
        ),
        "mean_output_tokens": (
            sum(valid_output_tokens) / len(valid_output_tokens) if valid_output_tokens else None
        ),
        "mean_latency_seconds": sum(valid_latencies) / len(valid_latencies) if valid_latencies else None,
        "aggregate_tps": aggregate_tps,
        "mean_tps": sum(valid_tps) / len(valid_tps) if valid_tps else None,
        "energy_kwh": optional_float(energy.get("energy_kwh")),
        "emissions_kg": optional_float(energy.get("emissions_kg")),
        "average_gpu_power_w": optional_float(energy.get("average_gpu_power_w")),
        "energy_tracker_error": energy.get("tracker_error"),
        "peak_training_vram_gb": peak_training_vram_gb,
        "peak_inference_vram_gb": peak_inference_vram_gb,
    }


def write_metric_tables(
    rows: Sequence[Mapping[str, Any]],
    results_dir: Path,
    pandas_module: Any,
) -> None:
    dataframe = pandas_module.DataFrame(rows)
    atomic_write_text(results_dir / "peft_mdpi_metrics.csv", dataframe.to_csv(index=False))
    atomic_write_json(results_dir / "peft_mdpi_metrics.json", list(rows))
    atomic_write_text(
        results_dir / "peft_mdpi_metrics.tex",
        dataframe.to_latex(
            index=False,
            escape=True,
            na_rep="--",
            float_format=lambda value: f"{value:.4f}",
        ),
    )


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 4-bit LoRA or IA3 adapter for noisy retail JSON parsing."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt_variants",
        nargs="+",
        default=list(DEFAULT_PROMPT_VARIANTS),
        help="Any of: minimal, extended, few_shot. Accepts spaces or commas.",
    )
    parser.add_argument("--train_tiers", nargs="+", default=list(TIER_COLUMNS))
    parser.add_argument("--eval_tiers", nargs="+", default=list(TIER_COLUMNS))
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--report_to", choices=("none", "tensorboard"), default="none")
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args(argv)


def validate_arguments(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.max_seq_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--max_seq_length and --max_new_tokens must be positive.")
    if args.num_train_epochs <= 0:
        raise ValueError("--num_train_epochs must be positive.")
    if args.train_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("Batch size and gradient accumulation steps must be positive.")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive.")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora_dropout must be in [0, 1).")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup_ratio must be in [0, 1).")
    if args.progress_every <= 0:
        raise ValueError("--progress_every must be positive.")
    return (
        validate_tiers(args.train_tiers, "--train_tiers"),
        validate_tiers(args.eval_tiers, "--eval_tiers"),
    )


def run_single_experiment(args: argparse.Namespace) -> int:
    train_tiers, eval_tiers = validate_arguments(args)
    args.output_dir = args.output_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    dependencies = load_dependencies()
    torch_module = dependencies.torch
    if not torch_module.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for 4-bit PEFT training, but CUDA is unavailable.")
    if torch_module.cuda.device_count() != 1:
        LOGGER.warning(
            "Detected %d GPUs; this script intentionally trains on GPU 0 only.",
            torch_module.cuda.device_count(),
        )

    torch_module.manual_seed(args.seed)
    train_commands = load_dataset(args.train_dataset_path)
    validation_commands = load_dataset(args.validation_dataset_path)
    test_commands = load_dataset(args.test_dataset_path)
    validate_dataset_splits(train_commands, validation_commands, test_commands)
    train_examples = expand_examples(train_commands, train_tiers)
    validation_examples = expand_examples(validation_commands, train_tiers)
    test_examples = expand_examples(test_commands, eval_tiers)
    LOGGER.info(
        "Loaded %d train, %d validation, and %d test command IDs (%d train, %d validation, %d test examples).",
        len(train_commands),
        len(validation_commands),
        len(test_commands),
        len(train_examples),
        len(validation_examples),
        len(test_examples),
    )

    atomic_write_json(
        args.output_dir / "run_config.json",
        {
            "created_at": utc_timestamp(),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "train_tiers": train_tiers,
            "eval_tiers": eval_tiers,
        },
    )

    model, tokenizer, training_summary = train_adapter(
        args,
        dependencies,
        train_examples,
        validation_examples,
    )
    records_by_tier, energy_by_tier, peak_inference_vram_gb = evaluate_adapter(
        model,
        tokenizer,
        args,
        dependencies,
        test_examples,
        eval_tiers,
    )

    rows: list[dict[str, Any]] = []
    for tier in eval_tiers:
        rows.append(
            metric_row(
                records_by_tier[tier],
                tier,
                args,
                dependencies.metrics,
                energy_by_tier[tier],
                training_summary["peak_training_vram_gb"],
                peak_inference_vram_gb,
            )
        )
    all_records = [record for records in records_by_tier.values() for record in records]
    rows.append(
        metric_row(
            all_records,
            "overall",
            args,
            dependencies.metrics,
            aggregate_energy(energy_by_tier),
            training_summary["peak_training_vram_gb"],
            peak_inference_vram_gb,
        )
    )
    write_metric_tables(rows, args.results_dir, dependencies.pandas)
    atomic_write_json(
        args.results_dir / "peft_energy_metrics.json",
        {
            "created_at": utc_timestamp(),
            "by_tier": energy_by_tier,
            "overall": aggregate_energy(energy_by_tier),
        },
    )
    LOGGER.info("Adapter saved in %s", args.output_dir / "adapter")
    LOGGER.info("Metrics written to %s", args.results_dir / "peft_mdpi_metrics.csv")
    del model, tokenizer
    dependencies.torch.cuda.empty_cache()
    gc.collect()
    return 0


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.") or "run"


def experiment_name(
    model_name: str,
    adapter_type: str,
    prompt_variant: str,
) -> str:
    return "__".join(
        (slugify(model_name), adapter_type, prompt_variant)
    )


def resolve_prompt_variants(values: Sequence[str]) -> list[str]:
    variants: list[str] = []
    for value in values:
        variants.extend(
            part.strip().replace("-", "_") for part in value.split(",") if part.strip()
        )
    variants = list(dict.fromkeys(variants))
    unknown = sorted(set(variants).difference(PROMPT_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown prompt variants: {', '.join(unknown)}.")
    if not variants:
        raise ValueError("At least one prompt variant is required.")
    return variants


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    model_names = resolve_names(MODEL_NAMES)
    adapter_types = resolve_names(ADAPTER_TYPES)
    prompt_variants = resolve_prompt_variants(args.prompt_variants)
    if not model_names:
        raise ValueError("MODEL_NAMES must contain at least one model identifier.")
    invalid_adapter_types = sorted(set(adapter_types).difference({"lora", "ia3"}))
    if invalid_adapter_types:
        raise ValueError(
            "ADAPTER_TYPES contains unsupported adapter types: "
            + ", ".join(invalid_adapter_types)
        )
    if not adapter_types:
        raise ValueError("ADAPTER_TYPES must contain at least one adapter type.")

    PEFT_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    completed_runs: list[dict[str, str]] = []
    failed_runs: list[dict[str, str]] = []
    for model_name in model_names:
        for adapter_type in adapter_types:
            for prompt_variant in prompt_variants:
                name = experiment_name(model_name, adapter_type, prompt_variant)
                run_args = argparse.Namespace(**vars(args))
                run_args.model_name = model_name
                run_args.adapter_type = adapter_type
                run_args.prompt_variant = prompt_variant
                run_args.train_dataset_path = TRAIN_DATASET_PATH
                run_args.validation_dataset_path = VALIDATION_DATASET_PATH
                run_args.test_dataset_path = TEST_DATASET_PATH
                run_args.output_dir = (ADAPTER_OUTPUT_ROOT / name).resolve()
                run_args.results_dir = (PEFT_RESULTS_ROOT / name).resolve()
                run_args.resume_from_checkpoint = "auto"
                try:
                    LOGGER.info("Starting configured experiment %s.", name)
                    run_single_experiment(run_args)
                    completed_runs.append(
                        {
                            "experiment_name": name,
                            "model_name": model_name,
                            "train_dataset_path": str(TRAIN_DATASET_PATH),
                            "validation_dataset_path": str(VALIDATION_DATASET_PATH),
                            "test_dataset_path": str(TEST_DATASET_PATH),
                            "adapter_type": adapter_type,
                            "prompt_variant": prompt_variant,
                        }
                    )
                except Exception as error:
                    LOGGER.exception("Configured experiment %s failed.", name)
                    failed_runs.append(
                        {
                            "experiment_name": name,
                            "model_name": model_name,
                            "train_dataset_path": str(TRAIN_DATASET_PATH),
                            "validation_dataset_path": str(VALIDATION_DATASET_PATH),
                            "test_dataset_path": str(TEST_DATASET_PATH),
                            "adapter_type": adapter_type,
                            "prompt_variant": prompt_variant,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                finally:
                    gc.collect()

    atomic_write_json(
        PEFT_RESULTS_ROOT / "peft_run_summary_ia3.json",
        {
            "created_at": utc_timestamp(),
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
        },
    )
    if failed_runs:
        LOGGER.error(
            "%d of %d configured PEFT runs failed.",
            len(failed_runs),
            len(completed_runs) + len(failed_runs),
        )
        return 1
    LOGGER.info("Completed %d configured PEFT runs.", len(completed_runs))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted. Training checkpoints and predictions remain available for resumption.")
        raise SystemExit(130)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(2)

