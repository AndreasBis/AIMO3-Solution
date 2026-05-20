# A. Model Summary

## A1. Background

| Field | Value |
| :--- | :--- |
| Competition Name | AI Mathematical Olympiad - Progress Prize 3 |
| Team Name | Andreas Bisiadis |
| Private Leaderboard Score | 41.5 |
| Private Leaderboard Place | 546 |
| Public Leaderboard Reference | 41/50 for the selected Mk I Version 16 notebook; 42/50 peak for the public no-suffix notebook lineage |
| Team Member Name | Andreas Bisiadis |

This submission was developed for Kaggle's AI Mathematical Olympiad - Progress Prize 3 competition. The selected solution is an inference-only notebook based on GPT-OSS-120B with tool-integrated Python reasoning. The notebook selected for the final submission was Mk I Version 16:

```text
code/inference/gpt-oss-120b-with-tools/aimo-3-gpt-oss-120b-with-tools-version-16.ipynb
```

This selected notebook scored 41/50 on the public leaderboard and 41.5 on the private leaderboard. The same public, no-suffix notebook lineage reached a peak public score of 42/50 in Version 11.

## A2. Background On Team

This was a solo submission by Andreas Bisiadis. The technical work represented in this repository focused on long-context LLM inference systems, vLLM backend tuning, tool-integrated mathematical reasoning, Jupyter kernel sandbox reliability, prompt minimization, and answer aggregation under Kaggle's fixed H100 runtime constraints.

Primary competition work included:

- Building and iterating 68 notebook versions and previews across Mk I, Mk II, Mk III, and Mk IV.
- Developing the public no-suffix notebook lineage, `gpt-oss-120b-with-tools`, through 22 versions. This is the only lineage intended to correspond to the publicly available Kaggle notebook.
- Running later Mk II, Mk III, and Mk IV development lines as follow-up research notebooks. These lines tested more advanced pipeline ideas, but they did not improve beyond the public notebook lineage's 42/50 public leaderboard peak.
- Migrating the inference backend in later experiments through vLLM versions up to `vllm==0.17.0`, while the selected public Mk I Version 16 notebook used `vllm==0.11.2`.
- Designing a stateful Python tool sandbox with persistent Jupyter kernels, port retry logic, output truncation, process cleanup, and CPU isolation.
- Evaluating prompt, timeout, context length, KV cache, pass@K, voting, and backend choices against reference problems and public leaderboard submissions.
- Selecting a final pipeline that emphasized reliability under the 5-hour competition notebook limit.

## A3. Summary

The final selected solution is an inference-time ensemble built around GPT-OSS-120B served locally with vLLM on a single Kaggle H100 GPU. Each math problem is solved by up to 8 parallel attempts using the Harmony prompt format, `ReasoningEffort.HIGH`, and a Python tool backed by persistent Jupyter kernels. Candidate integer answers are extracted from final answers using a `\boxed{}` parser with a narrow textual fallback for "final answer is", then aggregated with inverse-entropy weighting computed from top-logprob token entropy. The selected Mk I Version 16 notebook uses `vllm==0.11.2`, a 65,536-token context, FP8 E4M3 KV cache, async scheduling, disabled log stats, and prefix caching. No model training or fine-tuning is performed; development effort was concentrated on inference robustness, tool reliability, runtime allocation, and answer selection.

## A4. Feature Selection And Engineering

The effective inputs and engineered signals are:

- The original problem statement from Kaggle.
- A short system prompt identifying the assistant as a world-class IMO competitor and specifying the answer range and `\boxed{}` format.
- A short user suffix instructing the model that it has access to `math`, `numpy`, and `sympy`.
- Stateful Python outputs from each attempt's Jupyter sandbox.
- Per-attempt top-logprob entropy over generated tokens.
- Normalized candidate integer answers extracted from final answer text.

The main transformations are prompt composition, answer normalization, and entropy-based scoring. Boxed answer extraction accepts comma separators, strips them, converts to an integer, and keeps only values in `[0, 99999]`. Version 16 also includes a narrow case-insensitive fallback for text matching `final answer is <integer>`.

No external problem data or retrieval corpus is used. The only external artifacts are allowed Kaggle inputs: the competition data, the GPT-OSS-120B model weights, utility wheels, and local tiktoken encoding files needed for offline execution.

## A5. Training Method

No training, fine-tuning, reinforcement learning, or preference optimization is used in the submitted solution. The model is the pretrained GPT-OSS-120B checkpoint loaded from a Kaggle model dataset.

The ensemble is created at inference time:

- 8 attempts are launched per problem.
- Attempts use deterministic per-attempt seeds computed as `(seed + attempt_index) ** 2`, with `seed=42`.
- Sampling uses `temperature=1.0` and `min_p=0.02`.
- Each attempt can call a stateful Python tool for calculations and verification.
- Early stopping occurs when 4 attempts agree on the same answer.
- Final answer selection uses inverse entropy weighting:

```text
mean_entropy = average per-token Shannon entropy over top logprobs
weight = 1 / max(mean_entropy, entropy_floor)
entropy_floor = 1e-9
```

The answer with the largest total weight is submitted. If no valid answer is found, the notebook returns `0`.

## A6. Interesting Findings

The most important result was that inference-system reliability mattered more than adding more reasoning machinery. Strong leaderboard submissions consistently used pass@8, short prompts, FP8 KV cache, prefix caching, long context, and a 2x overprovisioned sandbox pool.

Key findings from the release history:

- `context_tokens=81920` avoided truncation failures observed at 65,536 tokens while preserving enough concurrency for 8 attempts.
- FP8 E4M3 KV cache was necessary for long-context concurrency; BF16/auto KV cache reduced concurrency and correlated with a lower public score.
- Pass@8 with early stop at 4 votes was a better runtime/performance tradeoff than pass@12 or pass@16 variants.
- Disabling vLLM prefix caching correlated with lower public scores.
- Short prompts were more stable than verbose problem-solving frameworks.
- The "IMO Gold Medalist" persona was more stable than broader competitor personas.
- A 2x sandbox pool (`workers=16` for `attempts=8`) repeatedly outperformed 1:1 worker/attempt configurations on the public leaderboard, even though the 10-problem local reference subset did not reveal the difference.
- For vLLM 0.17.0, the Marlin MoE backend was required to avoid CUDA OOM on the 120B model at `gpu_memory_utilization=0.99`.
- On the measured workload, `performance_mode="throughput"` was preferred over `interactivity` because prefix caching reduced repeated prefill cost and generation throughput dominated.
- CUDA 12.8 wheels outperformed CUDA 13.0 wheels in the logged experiments for this pipeline.

The later Mk II, Mk III, and Mk IV notebooks were useful research lines, but they are not the publicly available Kaggle notebook lineage. They tested superior-in-theory approaches such as vLLM upgrades, longer contexts, richer sandbox cleanup, explicit attention configuration, CUDA graph tuning, Marlin MoE backend selection, developer-channel prompt layouts, extra error interception, and XML-style prompts. Despite these improvements in engineering maturity, the follow-up pipelines did not outperform the public no-suffix notebook lineage's 42/50 public leaderboard peak.

Several explored ideas were not used because they reduced reliability or exceeded the runtime budget: fine-tuning, retrieval-augmented generation, in-context example retrieval, speculative decoding, entropy pruning from larger pass@K runs, sequential refinement, extra DSL/prover tools, developer-channel prompt variants, and eager-execution router/probing experiments.

## A7. Simpler Features And Methods

The closest simplified model that preserves most of the final result is the early Mk I baseline: GPT-OSS-120B with the Python tool, pass@8, prefix caching, 65,536-token context, and frequency-based voting. That baseline reached 40/50 on the public leaderboard, which is close to the selected Version 16 score of 41/50 and the public lineage peak of 42/50.

The most important single model is GPT-OSS-120B. The most important method is not training but reliable inference-time sampling with Python tool use and answer aggregation.

A very simple version with only one attempt would be faster but was not the selected approach because public leaderboard variance and hard-problem coverage favored multiple independent attempts.

## A8. Model Execution Time

Prediction time is bounded by the Kaggle notebook limit. The final selected notebook is configured for the 5-hour competition environment with:

| Runtime Setting | Value |
| :--- | :--- |
| Notebook limit | 17,400 seconds |
| Server startup timeout | 180 seconds |
| Base problem timeout | 240 seconds |
| High problem timeout | 840 seconds |
| Session timeout | 900 seconds |
| Python tool timeout | 6 seconds |
| Problems | 50 |
| Attempts per problem | 8 |
| Sandbox workers | 16 |

Actual runtime is stochastic because problem difficulty, answer agreement, tool calls, and generated token counts vary. The pipeline reserves time for remaining problems and gives surplus time to current hard problems while trying to finish within the competition's 5-hour limit.

The simpler Mk I-style baselines also perform no training. Their prediction runtime is similar or lower, but the selected Version 16 notebook combined tighter time allocation, refactored streaming/tool handling, regex compilation, and prefix caching in a more reliable public-notebook configuration.

## A9. References

Relevant references and sources used during development:

- Kaggle, AI Mathematical Olympiad - Progress Prize 3 competition environment and evaluation API.
- Kaggle, Winning Model Documentation Guidelines.
- OpenAI GPT-OSS-120B model documentation and Harmony chat format documentation.
- vLLM documentation and release behavior for versions 0.11.2, 0.15.1, 0.16.0, and 0.17.0.
- Flash Attention 3 backend documentation and vLLM attention configuration.
- OCP MXFP4 quantization format references.
- Internal release notes in `code/inference/**/RELEASES.md`.

# B. Submission Model

## B1. Archive Contents

The solution can be submitted as a Kaggle notebook solution. The selected final inference notebook is the public no-suffix Mk I notebook:

```text
code/inference/gpt-oss-120b-with-tools/aimo-3-gpt-oss-120b-with-tools-version-16.ipynb
```

The utility notebook used to prepare offline wheels and tiktoken encoding files is:

```text
code/inference/gpt-oss-120b-with-tools/aimo-3-utils.ipynb
```

The later Mk II, Mk III, and Mk IV directories document follow-up experiments. They are useful for explaining the development process, but the no-suffix `gpt-oss-120b-with-tools` lineage is the public notebook lineage, and Version 16 is the selected final submission notebook.

The release notes that document development history are under:

```text
code/inference/gpt-oss-120b-with-tools/RELEASES.md
code/inference/gpt-oss-120b-with-tools-mk-ii-previews/RELEASES.md
code/inference/gpt-oss-120b-with-tools-mk-ii/RELEASES.md
code/inference/gpt-oss-120b-with-tools-mk-iii-previews/RELEASES.md
code/inference/gpt-oss-120b-with-tools-mk-iii/RELEASES.md
code/inference/gpt-oss-120b-with-tools-mk-iv/RELEASES.md
```

No trained model file is produced by this repository. The pretrained model is loaded from the Kaggle GPT-OSS-120B model input.

## B2. README Requirements

### Hardware

The Kaggle competition notebook environment provides:

| Resource | Value |
| :--- | :--- |
| GPU | 1 x NVIDIA H100 SXM |
| GPU memory | 80 GB HBM3 |
| GPU bandwidth | 3.35 TB/s |
| CPU workers | 26 |
| System memory | 230 GB |
| Internet | Disabled during competition inference |

### Platform

The selected notebook metadata records:

| Platform Item | Value |
| :--- | :--- |
| Environment | Kaggle Notebook |
| Accelerator | NVIDIA H100 |
| Python | 3.12.12 |
| Kaggle Docker image version | 31236 |
| Internet | Disabled for inference notebook |

### Key Software

The selected public utility notebook downloads offline wheels for:

```text
unsloth
trl
vllm==0.11.2
openai_harmony
```

The utility notebook downloads Python 3.12 binary wheels using the CUDA 12.8 PyTorch index and the NVIDIA package index, then packages the wheels and tiktoken encodings into `wheels.tar.gz`. The selected inference notebook extracts that archive to:

```text
/kaggle/tmp/setup
```

and installs from:

```text
/kaggle/tmp/setup/wheels
```

Important runtime libraries imported by the selected notebook include `vllm`, `openai`, `openai_harmony`, `transformers`, `jupyter_client`, `pandas`, `polars`, and Kaggle's `kaggle_evaluation.aimo_3_inference_server`.

### How To Train

There is no training step.

### How To Predict

Run the selected Mk I Version 16 Kaggle notebook with the required Kaggle inputs attached:

- AI Mathematical Olympiad - Progress Prize 3 competition data.
- GPT-OSS-120B model input at `/kaggle/input/gpt-oss-120b/transformers/default/1`.
- Public utility input containing `/kaggle/input/aimo-3-utils/wheels.tar.gz`.

The notebook:

1. Extracts `/kaggle/input/aimo-3-utils/wheels.tar.gz` to `/kaggle/tmp/setup`.
2. Installs `unsloth`, `trl`, `vllm`, and `openai_harmony` from offline wheels.
3. Sets tokenizer, CUDA, and transformer-related environment variables.
4. Starts a local vLLM OpenAI-compatible server for GPT-OSS-120B.
5. Initializes 16 persistent Jupyter sandbox kernels.
6. Registers `predict()` with `kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer`.
7. Calls `serve()` during competition reruns or `run_local_gateway()` for local reference execution.

### Important Side Effects

The notebook intentionally modifies the runtime environment:

- Uninstalls several large packages before installing the offline wheel set: `keras`, `matplotlib`, `scikit-learn`, and `tensorflow`.
- Extracts the utility archive to `/kaggle/tmp/setup`.
- Starts a vLLM server process on port `8000`.
- Writes `vllm_server.log` in the working directory.
- Starts 16 Jupyter kernels and allocates local ports starting at `50000`.
- Resets and reuses sandbox kernels between problems.
- Does not overwrite competition input data.

### Key Assumptions

- The notebook runs in the Kaggle AIMO3 environment with one H100 GPU and no internet access.
- The GPT-OSS-120B model and `aimo-3-utils` wheel archive are attached as Kaggle inputs.
- The answer is an integer between 0 and 99999.
- The full test set contains 50 problems.
- The competition rerun environment sets `KAGGLE_IS_COMPETITION_RERUN`.
- The working directory can hold logs and kernel temporary files.

## B3. Configuration Files

There are no external configuration files required by the final notebook. Runtime configuration is centralized in the `CFG` class inside the selected notebook.

Important final configuration values include:

| Setting | Value |
| :--- | :--- |
| Model | GPT-OSS-120B |
| Model path | `/kaggle/input/gpt-oss-120b/transformers/default/1` |
| vLLM version | 0.11.2 |
| Reasoning effort | HIGH |
| KV cache dtype | `fp8_e4m3` |
| dtype | `auto` |
| Context length | 65,536 |
| Search tokens | 32 |
| Buffer tokens | 512 |
| Max sequences | 256 |
| GPU memory utilization | 0.96 |
| Prefix caching | Enabled |
| Async scheduling | Enabled |
| Disable log stats | Enabled |
| Temperature | 1.0 |
| min_p | 0.02 |
| top_logprobs | 5 |
| Attempts | 8 |
| Workers | 16 |
| Early stop | 4 matching answers |
| Max turns | 128 |
| Seed | 42 |
| Entropy score floor | `1e-9` in the inverse-entropy weighting expression |

## B4. Requirements

For the Kaggle notebook workflow, exact package installation is handled by the utility dataset. The repository `requirements.txt` corresponds to the utility notebook used by the selected Mk I Version 16 submission, `code/inference/gpt-oss-120b-with-tools/aimo-3-utils.ipynb`; it is the expanded pinned package list derived from the wheels downloaded by that utility notebook.

The selected utility notebook downloads binary wheels for Python 3.12 using:

```text
pip download --dest /kaggle/working/wheels --python-version 3.12 --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.nvidia.com unsloth trl vllm==0.11.2 openai_harmony
```

The inference notebook then installs offline with:

```text
python -m pip install --no-index --find-links /kaggle/tmp/setup/wheels unsloth trl vllm openai_harmony
```

The tiktoken encoding files are bundled into the same archive and loaded from:

```text
/kaggle/tmp/setup/tiktoken_encodings
```

## B5. Directory Structure

The main project directories relevant to this writeup are:

```text
.
code/
code/inference/
code/inference/gpt-oss-120b-with-tools/
code/inference/gpt-oss-120b-with-tools-mk-ii-previews/
code/inference/gpt-oss-120b-with-tools-mk-ii/
code/inference/gpt-oss-120b-with-tools-mk-iii-previews/
code/inference/gpt-oss-120b-with-tools-mk-iii/
code/inference/gpt-oss-120b-with-tools-mk-iv/
code/other/gpt-oss-120b-error-logs/
sandbox/
```

For a formal archive, generate `directory_structure.txt` from the project root with:

```text
find . -type d > directory_structure.txt
```

## B6. Settings

Kaggle's generic documentation template recommends a `SETTINGS.json` file. This repository uses a notebook-style competition submission instead. All I/O paths and runtime settings are centralized in the notebook's `CFG` class and in the Kaggle input attachments.

Primary paths:

| Purpose | Path |
| :--- | :--- |
| Model weights | `/kaggle/input/gpt-oss-120b/transformers/default/1` |
| Utility archive | `/kaggle/input/aimo-3-utils/wheels.tar.gz` |
| Extracted offline wheels | `/kaggle/tmp/setup/wheels` |
| Tiktoken encodings | `/kaggle/tmp/setup/tiktoken_encodings` |
| Local gateway data | `/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv` |
| vLLM server log | `vllm_server.log` |

## B7. Serialized Trained Model

No serialized trained model is produced. The submission uses the pretrained GPT-OSS-120B checkpoint supplied as a Kaggle model input. All competition-specific behavior is implemented in the inference notebook.

## B8. Entry Points

The competition entry point is the `predict()` function in the selected notebook:

```text
code/inference/gpt-oss-120b-with-tools/aimo-3-gpt-oss-120b-with-tools-version-16.ipynb
```

The notebook creates:

```text
inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
```

Then it uses:

```text
inference_server.serve()
```

when `KAGGLE_IS_COMPETITION_RERUN` is set. For local reference execution, it uses:

```text
inference_server.run_local_gateway(("/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv",))
```

There are no separate `prepare_data.py`, `train.py`, or `predict.py` scripts because the Kaggle submission format is a notebook with a registered inference server callback.
