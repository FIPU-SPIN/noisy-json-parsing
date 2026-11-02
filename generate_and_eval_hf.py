#!/usr/bin/env python3
"""Generate and evaluate noisy retail JSON parsing with Hugging Face models.

This is the Hugging Face counterpart of generate_and_eval.py. It preserves the
same dataset contract, checkpoint format, schema validation, metrics, and energy
tracking while using 4-bit Transformers generation instead of Ollama.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import logging
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from generate_and_eval import (
    BASE_DIR,
    aggregate_energy_runs,
    atomic_write_json,
    build_metric_rows,
    completed_keys,
    evaluation_records,
    load_dataset,
    load_json_list,
    make_result_record,
    resolve_names,
    start_energy_tracker,
    stop_energy_tracker,
    utc_timestamp,
    write_metrics_table,
)


DEFAULT_DATASET = BASE_DIR / "test_dataset_v3.csv"
DEFAULT_RESPONSES_DIR = BASE_DIR / "responses" / "huggingface"
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "huggingface"
MODEL_NAMES = (
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
)
TIERS = ("tier_0", "tier_1", "tier_2", "tier_3")
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
LOGGER = logging.getLogger("generate_and_eval_hf")


def require_module(module_name: str, package_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"Missing dependency '{package_name}'. Install it in the Hugging Face "
            "runtime before running this evaluator."
        ) from error


def load_runtime() -> SimpleNamespace:
    torch_module = require_module("torch", "torch")
    transformers_module = require_module("transformers", "transformers")
    require_module("accelerate", "accelerate")
    require_module("bitsandbytes", "bitsandbytes")
    require_module("codecarbon", "codecarbon")
    require_module("pandas", "pandas")
    require_module("sklearn.metrics", "scikit-learn")
    return SimpleNamespace(
        torch=torch_module,
        AutoModelForCausalLM=transformers_module.AutoModelForCausalLM,
        AutoTokenizer=transformers_module.AutoTokenizer,
        BitsAndBytesConfig=transformers_module.BitsAndBytesConfig,
    )


def compute_dtype(torch_module: Any) -> Any:
    return torch_module.bfloat16 if torch_module.cuda.is_bf16_supported() else torch_module.float16


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


def load_model_and_tokenizer(
    model_name: str,
    runtime: SimpleNamespace,
    trust_remote_code: bool,
    local_files_only: bool,
) -> tuple[Any, Any]:
    torch_module = runtime.torch
    if not torch_module.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Hugging Face 4-bit inference.")

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    token = os.getenv("HF_TOKEN")
    if token:
        model_kwargs["token"] = token

    tokenizer = runtime.AutoTokenizer.from_pretrained(model_name, **model_kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = compute_dtype(torch_module)
    quantization_config = runtime.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = runtime.AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        dtype=dtype,
        device_map={"": 0},
        **model_kwargs,
    )
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def build_generation_inputs(
    tokenizer: Any,
    command: str,
    max_input_tokens: int,
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
                max_length=max_input_tokens,
                return_dict=True,
                return_tensors="pt",
            )
        except (AttributeError, TypeError, ValueError) as error:
            LOGGER.warning("Chat template failed; using the fallback prompt: %s", error)

    return tokenizer(
        f"{PROMPT_VARIANTS[prompt_variant]}\n\nUser: {command}\n\nAssistant:",
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt",
    )


def model_input_device(model: Any) -> Any:
    return model.get_input_embeddings().weight.device


def query_model(
    model: Any,
    tokenizer: Any,
    command: str,
    runtime: SimpleNamespace,
    max_input_tokens: int,
    max_new_tokens: int,
    retries: int,
    retry_delay: float,
    prompt_variant: str = "extended",
) -> dict[str, Any]:
    torch_module = runtime.torch
    last_error: str | None = None
    for attempt in range(1, retries + 2):
        try:
            inputs = build_generation_inputs(tokenizer, command, max_input_tokens, prompt_variant)
            device = model_input_device(model)
            inputs = {
                name: value.to(device) if hasattr(value, "to") else value
                for name, value in inputs.items()
            }
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise RuntimeError("Tokenizer did not return input_ids.")

            prompt_eval_count = int(input_ids.shape[-1])
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "use_cache": True,
            }
            if tokenizer.pad_token_id is not None:
                generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
            if tokenizer.eos_token_id is not None:
                generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

            torch_module.cuda.synchronize()
            started_at_ns = time.perf_counter_ns()
            with torch_module.inference_mode():
                outputs = model.generate(**inputs, **generation_kwargs)
            torch_module.cuda.synchronize()
            eval_duration_ns = time.perf_counter_ns() - started_at_ns

            generated_ids = outputs[0, prompt_eval_count:]
            eval_count = int(generated_ids.shape[-1])
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            tokens_per_second = (
                eval_count / (eval_duration_ns / 1_000_000_000)
                if eval_duration_ns > 0
                else None
            )
            return {
                "status": "ok",
                "response": response,
                "error": None,
                "attempts": attempt,
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
                "eval_duration_ns": eval_duration_ns,
                "tokens_per_second": tokens_per_second,
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


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate noisy retail JSON parsing with Hugging Face models."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_NAMES),
        help="Hugging Face model IDs, separated by spaces or commas.",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=list(TIERS),
        help="Tier names, separated by spaces or commas.",
    )
    parser.add_argument(
        "--prompt-variants",
        nargs="+",
        default=list(DEFAULT_PROMPT_VARIANTS),
        help="Any of: minimal, extended, few_shot. Accepts spaces or commas.",
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N command IDs.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


def validate_arguments(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.retries < 0 or args.retry_delay < 0:
        raise ValueError("--retries and --retry-delay cannot be negative.")
    if args.max_input_tokens <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Token limits must be positive.")

    models = resolve_names(args.models)
    tiers = resolve_names(args.tiers)
    unknown_tiers = sorted(set(tiers).difference(TIERS))
    if unknown_tiers:
        raise ValueError(f"Unknown tiers: {', '.join(unknown_tiers)}.")
    if not models or not tiers:
        raise ValueError("At least one model and one tier are required.")
    return models, tiers


def run_prompt_variant(
    args: argparse.Namespace,
    dataset: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    tiers: Sequence[str],
    prompt_variant: str,
    runtime: SimpleNamespace,
) -> bool:
    responses_file = args.responses_dir / prompt_variant / "results.json"
    results_dir = args.results_dir / prompt_variant
    energy_path = results_dir / "energy_runs.json"
    results = load_json_list(responses_file, f"{prompt_variant} Hugging Face response checkpoint")
    energy_runs = load_json_list(energy_path, f"{prompt_variant} Hugging Face energy checkpoint")
    done = completed_keys(results)
    LOGGER.info(
        "%s: loaded %d saved responses and %d completed checkpoints.",
        prompt_variant,
        len(results),
        len(done),
    )

    for model_name in models:
        pending_tiers = {
            tier: [item for item in dataset if (model_name, item["id"], tier) not in done]
            for tier in tiers
        }
        if not any(pending_tiers.values()):
            LOGGER.info("%s %s: already complete.", prompt_variant, model_name)
            continue

        LOGGER.info("%s: loading %s in 4-bit mode.", prompt_variant, model_name)
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            runtime,
            args.trust_remote_code,
            args.local_files_only,
        )
        try:
            for tier, pending_items in pending_tiers.items():
                if not pending_items:
                    continue
                LOGGER.info(
                    "%s %s %s: processing %d commands.",
                    prompt_variant,
                    model_name,
                    tier,
                    len(pending_items),
                )
                tracker = start_energy_tracker(results_dir, model_name, tier)
                scope_started_at = utc_timestamp()
                scope_started_monotonic = time.monotonic()
                completed_in_scope = 0
                try:
                    for index, item in enumerate(pending_items, start=1):
                        query_result = query_model(
                            model,
                            tokenizer,
                            item["commands"][tier],
                            runtime,
                            args.max_input_tokens,
                            args.max_new_tokens,
                            args.retries,
                            args.retry_delay,
                            prompt_variant,
                        )
                        record = make_result_record(
                            model_name,
                            tier,
                            item,
                            query_result,
                            prompt_variant,
                        )
                        results.append(record)
                        atomic_write_json(responses_file, results)
                        completed_in_scope += 1
                        if record["status"] == "ok":
                            done.add((model_name, item["id"], tier))
                            LOGGER.info(
                                "%s %s %s %s (%d/%d): schema=%s exact=%s tps=%s",
                                prompt_variant,
                                model_name,
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
                                model_name,
                                tier,
                                item["id"],
                                index,
                                len(pending_items),
                                record["error"],
                            )
                finally:
                    energy_record: dict[str, Any] = {
                        "model": model_name,
                        "tier": tier,
                        "started_at": scope_started_at,
                        "finished_at": utc_timestamp(),
                        "duration_seconds": time.monotonic() - scope_started_monotonic,
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
                            model_name,
                            tier,
                        )
                    energy_runs.append(energy_record)
                    atomic_write_json(energy_path, energy_runs)
        finally:
            del model, tokenizer
            runtime.torch.cuda.empty_cache()
            gc.collect()

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
    models, tiers = validate_arguments(args)
    prompt_variants = resolve_prompt_variants(args.prompt_variants)
    runtime = load_runtime()
    if not runtime.torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible. Submit this evaluator to a GPU node.")

    dataset = load_dataset(args.dataset)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    any_failed_requests = False
    for prompt_variant in prompt_variants:
        any_failed_requests = run_prompt_variant(
            args,
            dataset,
            models,
            tiers,
            prompt_variant,
            runtime,
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