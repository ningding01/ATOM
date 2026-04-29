# Qwen3.5 SGLang ATOM Backend Benchmark

This note describes how to benchmark Qwen3.5-35B-A3B-FP8 in SGLang with the
ATOM model package enabled.

## Server Launch

Enable the ATOM SGLang model package and launch the SGLang server:

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_DISABLE_CUDNN_CHECK=1

python3 -m sglang.launch_server \
  --model-path /raid/models/Qwen3.5-35B-A3B-FP8 \
  --tp-size 1 \
  --mem-fraction-static 0.85 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --port 9527 \
  --host 0.0.0.0
```

Wait until the server log shows that the server is ready.

## Prefill Benchmark

For prefill performance, use a long input and generate only one token. Read the
prefill/input throughput from the `sglang.bench_serving` output.

```bash
python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random \
  --dataset-path /home/logs/ShareGPT_V3_unfiltered_cleaned_split.json \
  --random-input 4096 \
  --random-output 1 \
  --random-range-ratio 1 \
  --num-prompts 128 \
  --max-concurrency 128 \
  --output-file "prefill_4k_o1.jsonl" \
  --request-rate 100 \
  --port 9527 \
  --flush-cache
```

Record the `input throughput` from the benchmark output. This is the primary
prefill metric for the 4K input / 1 output-token case.

## Decode Benchmark

For decode performance, use the same 4K input length and generate 2K output
tokens. The benchmark drives enough requests to fill the batch, while the decode
throughput should be read from the SGLang server log after the batch size is
fully saturated.

```bash
python3 -m sglang.bench_serving --backend sglang \
  --dataset-name random \
  --dataset-path /home/logs/ShareGPT_V3_unfiltered_cleaned_split.json \
  --random-input 4096 \
  --random-output 2048 \
  --random-range-ratio 1 \
  --num-prompts 128 \
  --max-concurrency 128 \
  --output-file "decode_4k_o2k.jsonl" \
  --request-rate 100 \
  --port 9527 \
  --flush-cache
```

In the server log, wait until decode batches reach the saturated batch size, then
record the steady-state `gen throughput (token/s)` from the `Decode batch` log
lines.

Example log line:

```text
Decode batch, #running-req: ..., #full token: ..., cuda graph: True, gen throughput (token/s): ...
```

## Multimodal Benchmark

For multimodal performance, run the MMMU SGLang benchmark against a running
SGLang server. The example below uses the Hugging Face mirror endpoint and
connects to port `9527`:

```bash
HF_ENDPOINT=https://hf-mirror.com \
python3 /opt/sglang/benchmark/mmmu/bench_sglang.py \
  --port 9527 \
  --concurrency 48
```

## Metrics To Report

Report these metrics for each run:

- Prefill: `input throughput` from `sglang.bench_serving` with
  `--random-input 4096 --random-output 1`.
- Decode: steady-state `gen throughput (token/s)` from SGLang server logs with
  `--random-input 4096 --random-output 2048` after batch size is saturated.
- Multimodal: MMMU benchmark results from `bench_sglang.py`, including the
  server port and concurrency.
- Benchmark configuration: input length, output length, number of prompts,
  max concurrency, request rate, and whether cache was flushed.

## Benchmark Results

| Model | Hardware | TP | Input/Output | Prefill Throughput | Prefill Drop | Decode Throughput (tok/s) | Decode Drop | TPOT Mean | TPOT Mean Increase | TPOT P99 | TPOT P99 Increase | Max Decode BS |
|-------|----------|----|--------------|--------------------|--------------|---------------------------|-------------|-----------|--------------------|----------|-------------------|---------------|
| Qwen3.5-35B-A3B-FP8 | H20 | TP1 | 4k/2k | 27036.34 | — | 3684 | — | 69.32 ms | — | 76.63 ms | — | 224 |
| Qwen3.5-35B-A3B-FP8 | H20 | TP1 | 6k/2k | 26172.81 | — | 2818 | — | 68.95 ms | — | 77.56 ms | — | 168 |
| Qwen3.5-35B-A3B-FP8 | H20 | TP1 | 14k/2k | 23666.91 | — | 1403 | — | 114.35 ms | — | 79.65 ms | — | 80 |
| Qwen3.5-35B-A3B-FP8 | H20 | TP1 | 30k/2k | 19289.6 | — | 684 | — | 73.51 ms | — | 87.94 ms | — | 40 |
| Qwen3.5-35B-A3B-FP8 | H20 | TP1 | 62k/2k | 13991.71 | — | 375 | — | 59.5 ms | — | 76.03 ms | — | 16 |
| Qwen3.5-35B-A3B-FP8 | H20 | TP1 | 126k/2k | 9005.56 | — | 181 | — | 68.34 ms | — | 91.99 ms | — | 8 |
| Qwen3.5-35B-A3B-FP8 | MI308 | TP1 | 4k/2k | 17475.28 | -35.4% | 2220 | -39.7% | 114.01 ms | +64.5% | 128.68 ms | +67.9% | 224 |
| Qwen3.5-35B-A3B-FP8 | MI308 | TP1 | 6k/2k | 17170.66 | -34.4% | 1880 | -33.3% | 104.33 ms | +51.3% | 118.79 ms | +53.2% | 168 |
| Qwen3.5-35B-A3B-FP8 | MI308 | TP1 | 14k/2k | 15400.48 | -34.9% | 1110 | -20.9% | 89.78 ms | -21.5% | 107.30 ms | +34.7% | 80 |
| Qwen3.5-35B-A3B-FP8 | MI308 | TP1 | 30k/2k | 12423.04 | -35.6% | 630 | -7.9% | 86.82 ms | +18.1% | 110.39 ms | +25.5% | 40 |
| Qwen3.5-35B-A3B-FP8 | MI308 | TP1 | 62k/2k | 8943.53 | -36.1% | 320 | -14.7% | 77.01 ms | +29.4% | 105.70 ms | +39.0% | 16 |
| Qwen3.5-35B-A3B-FP8 | MI308 | TP1 | 126k/2k | 5789.72 | -35.7% | 170 | -6.1% | 85.42 ms | +25.0% | 124.91 ms | +35.8% | 8 |

