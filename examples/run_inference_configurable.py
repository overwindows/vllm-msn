#!/usr/bin/env python3
"""
Configurable vLLM inference script for ablation study experiments.
Accepts all configuration parameters as command-line arguments.
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import time

from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from vllm.logger import init_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configurable vLLM inference for ablation study"
    )

    # Model configuration
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=[None, "fp8", "awq", "gptq"],
                        help="Quantization method")
    parser.add_argument("--kv_cache_dtype", type=str, default="auto",
                        help="KV cache dtype (auto, fp8_e5m2, fp8_e4m3)")

    # MTP configuration
    parser.add_argument("--speculative_model", type=str, default=None,
                        help="Path to MTP assistant model")
    parser.add_argument("--num_speculative_tokens", type=int, default=5,
                        help="Number of speculative tokens")

    # Memory configuration
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75,
                        help="GPU memory utilization ratio")
    parser.add_argument("--max_num_seqs", type=int, default=128,
                        help="Maximum number of sequences")
    parser.add_argument("--max_num_batched_tokens", type=int, default=6144,
                        help="Maximum batched tokens")
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="Maximum model length")

    # Performance configuration
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Disable CUDA graphs (eager mode)")
    parser.add_argument("--enable_cuda_graphs", action="store_true",
                        help="Enable CUDA graphs (ignored if enforce_eager)")

    # Input/Output
    parser.add_argument("--input_path", type=str, default=None,
                        help="Input file path (optional for smoke test)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--num_test_samples", type=int, default=100,
                        help="Number of samples for smoke test (if no input)")

    # Sampling configuration
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max tokens to generate")

    return parser.parse_args()


async def generate_batch(
    engine: AsyncLLMEngine,
    prompts: List[str],
    sampling_params: SamplingParams
) -> List[Dict[str, Any]]:
    """Generate completions for a batch of prompts."""

    results = []
    request_ids = []

    # Submit all prompts
    for i, prompt in enumerate(prompts):
        request_id = f"request_{i}_{int(time.time() * 1000)}"
        request_ids.append(request_id)
        await engine.add_request(
            request_id=request_id,
            inputs=prompt,
            params=sampling_params
        )

    # Collect results
    completed = set()
    while len(completed) < len(prompts):
        async for request_output in engine.engine_step_async():
            if request_output.finished:
                completed.add(request_output.request_id)

                # Extract generated text
                output_text = request_output.outputs[0].text if request_output.outputs else ""

                results.append({
                    "request_id": request_output.request_id,
                    "prompt": prompts[request_ids.index(request_output.request_id)][:100] + "...",
                    "output": output_text,
                    "num_tokens": len(request_output.outputs[0].token_ids) if request_output.outputs else 0
                })

    return results


async def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("vLLM Configurable Inference")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"Batch size: {args.max_num_seqs}")
    logger.info(f"GPU memory: {args.gpu_memory_utilization}")
    logger.info(f"CUDA graphs: {not args.enforce_eager}")
    logger.info(f"MTP: {args.speculative_model is not None}")
    logger.info("=" * 70)
    logger.info("")

    # Build engine args
    logger.info("Building AsyncEngineArgs...")

    # Handle enforce_eager / enable_cuda_graphs logic
    enforce_eager = args.enforce_eager
    if args.enable_cuda_graphs and not args.enforce_eager:
        enforce_eager = False

    engine_args = AsyncEngineArgs(
        model=args.model_path,
        dtype=args.dtype,
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enforce_eager=enforce_eager,
        trust_remote_code=True,
        # MTP configuration
        speculative_model=args.speculative_model if args.speculative_model else None,
        num_speculative_tokens=args.num_speculative_tokens if args.speculative_model else None,
        # Logging
        disable_log_stats=False,
        disable_log_requests=True,
    )

    logger.info("Configuration:")
    logger.info(f"  dtype: {args.dtype}")
    logger.info(f"  quantization: {args.quantization}")
    logger.info(f"  kv_cache_dtype: {args.kv_cache_dtype}")
    logger.info(f"  gpu_memory_utilization: {args.gpu_memory_utilization}")
    logger.info(f"  max_num_seqs: {args.max_num_seqs}")
    logger.info(f"  max_num_batched_tokens: {args.max_num_batched_tokens}")
    logger.info(f"  enforce_eager: {enforce_eager}")
    logger.info(f"  speculative_model: {args.speculative_model}")
    logger.info("")

    # Initialize engine
    logger.info("Initializing AsyncLLMEngine...")
    start_time = time.time()
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    init_time = time.time() - start_time
    logger.info(f"✓ Engine initialized in {init_time:.2f} seconds")
    logger.info("")

    # Prepare sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    # Get test prompts
    if args.input_path and Path(args.input_path).exists():
        logger.info(f"Loading prompts from {args.input_path}...")
        prompts = []

        # Try to parse as JSONL first (vLLM format with messages field)
        try:
            with open(args.input_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)

                    # Extract messages field if present
                    if 'messages' in data and isinstance(data['messages'], list):
                        # Format messages for vLLM (system + user)
                        prompt_parts = []
                        for msg in data['messages']:
                            role = msg.get('role', '')
                            content = msg.get('content', '')
                            if role == 'system':
                                prompt_parts.append(f"System: {content}")
                            elif role == 'user':
                                prompt_parts.append(f"User: {content}")
                        prompts.append("\n\n".join(prompt_parts))
                    else:
                        # Fallback: treat whole JSON as string
                        prompts.append(json.dumps(data))

            logger.info(f"  Loaded {len(prompts)} prompts from JSONL")
        except json.JSONDecodeError:
            # Not JSON, treat as plain text (one prompt per line)
            with open(args.input_path) as f:
                prompts = [line.strip() for line in f if line.strip()]
            logger.info(f"  Loaded {len(prompts)} prompts from text file")
    else:
        logger.info(f"No input file, generating {args.num_test_samples} test prompts...")
        prompts = [
            f"Write a short story about {topic}."
            for topic in [
                "artificial intelligence", "space exploration", "time travel",
                "ancient civilizations", "future cities", "ocean depths",
                "quantum physics", "virtual reality", "genetic engineering",
                "climate change"
            ] * (args.num_test_samples // 10 + 1)
        ][:args.num_test_samples]

    logger.info(f"Total prompts: {len(prompts)}")
    logger.info("")

    # Run inference
    logger.info("Starting inference...")
    start_time = time.time()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    with open(output_path, 'w') as f:
        # Process in batches
        batch_size = args.max_num_seqs
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size} ({len(batch_prompts)} prompts)...")

            batch_start = time.time()
            results = await generate_batch(engine, batch_prompts, sampling_params)
            batch_time = time.time() - batch_start

            # Write results
            for result in results:
                f.write(json.dumps(result) + '\n')
                total_tokens += result['num_tokens']

            logger.info(f"  Batch completed in {batch_time:.2f}s ({len(batch_prompts) / batch_time:.2f} req/s)")

    total_time = time.time() - start_time

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("Inference Complete!")
    logger.info("=" * 70)
    logger.info(f"Total prompts: {len(prompts)}")
    logger.info(f"Total tokens generated: {total_tokens}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Throughput: {len(prompts) / total_time:.2f} requests/sec")
    logger.info(f"Throughput: {total_tokens / total_time:.2f} tokens/sec")
    logger.info(f"Average latency: {total_time / len(prompts):.2f} sec/request")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
