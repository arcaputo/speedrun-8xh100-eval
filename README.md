# speedrun-8xh100

A Kupanga benchmark on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)'s Track 1
speedrun: train GPT-2 (124M) on FineWeb to a val loss of **3.28** as fast as possible on **8×H100 SXM**.
Each scoring times the candidate against its parent's code **on the same machine**, as every speedrun PR does.

| File | What it is |
|---|---|
| `eval.toml` | Runs `python bench.py`; `train_time_s` is the primary metric, lower is better |
| `job.toml` | `python:3.12`, 8×H100 (GCP a3-highgpu-8g, spot, 3 retries), 32+ CPUs, 256+ GB, 130 min limit |
| `requirements.txt` | The speedrun's pinned stack: torch 2.10.0+cu128, `kernels` (FA3 from the HF hub), CUDA 12.8 header wheels, scipy |
| `bench.py` | Fetches the data, rebuilds the parent, runs both N times, parses logs, runs the t-test |

## What it does

1. Downloads `kjj0/fineweb10B-gpt2` (the val shard and 9 training shards), as `python
   data/cached_fineweb10B.py 9` would, into a read-only folder it gives the run as `DATA_PATH`.
2. Rebuilds the parent's code by reverse-applying `$HARNESS_CHANGES_FILE` (what the candidate changed
   since its experiment started) to a copy of the subject. The base node has no parent, so it is
   timed on its own.
3. Runs `torchrun --standalone --nproc_per_node=8 train_gpt.py` 8 times for the candidate and 3 times
   for the parent, alternating (c p c c p c c p …), so drift on the machine affects both. The
   compile cache is shared, so only the first run of each code compiles (about 7 minutes).
4. Reads each run's last `step:N/N val_loss:X train_time:Yms` line from `logs/<uuid>.txt`.
   `train_time` leaves out compiling and warmup, as on the leaderboard.
5. Reports:
   - the candidate's mean `train_time_s` and its standard error
   - the mean `val_loss`
   - the speedrun's `p_value` from `scipy.stats.ttest_1samp(losses, 3.28, alternative="less")`
   - `time_delta_s`, the candidate minus the parent on this machine

A candidate whose mean val_loss isn't below 3.28 with p < 0.01 is still reported, with `meets_target` = 0:
its time doesn't count, because a time only counts at the target loss. If a run crashes or times out, the
candidate gets no score either, and the message shows the run's last output.

## Environment

`python:3.12` has no CUDA toolkit, but the subject compiles NVRTC kernels at import time (`#include <cuda_bf16.h>`),
and torch's `_compile_kernel` needs `CUDA_HOME`. So `bench.py` merges the include folders of NVIDIA's pip wheels
(`nvidia-cuda-runtime-cu12`, `-nvcc-` for `crt/`, `-cccl-`) into one CUDA root, sets `CUDA_HOME` to it, and links it at
`/usr/local/cuda`, which #87's kernel names. The `pytorch/pytorch` CUDA images aren't an option: their system Python
refuses Kupanga's plain `pip install` (PEP 668). Before the first run, the subject's `get_kernel(...)` hub kernels
are fetched once, so the 8 ranks don't all download them into one cache at the same moment. A failed run shows
the first rank's traceback.

`kernels` is pinned to 0.13.0. Newer releases refuse kernels from accounts that aren't trusted Hugging Face
organizations (0.14) and require a `version=` or `revision=` on every `get_kernel` (0.15). Records up to #86 call
`get_kernel('varunneal/flash-attention-3')` with neither, from a personal account, so they fail at import on 0.16.1.
0.13.0 still accepts `version=`, which #87's `kernels-community/flash-attn3` call uses.

## Subjects with several records

A subject may hold one record at its root (`subjects/<name>/train_gpt.py`) or several, one folder each
(`subjects/<name>/<record>/train_gpt.py`). In the second case the eval times the one folder the candidate
changed, against that folder's parent code; a candidate that changes several folders, or the base node
(which changes none), has nothing to time.

## Rules

The `speedrun-rules` QA eval checks these before this eval spends GPU time:
- `val_tokens = 10485760`
- the data comes from `DATA_PATH` and nothing under `data/` changes
- no torch.compile or inductor flags are added
- `train_gpt.py` stays the entry point and prints its final line

This eval trusts the run's own validation, as the speedrun does: it doesn't score a checkpoint itself.

## Settings

| Variable | Default | |
|---|---|---|
| `KX_RUNS` | 8 | Candidate runs. With 5, three jumps that matched their record's speedup at means of 3.2782–3.2790 missed p < 0.01 (p = 0.03); 8 clear a mean about 0.001 under 3.28 |
| `KX_BASELINE_RUNS` | 3 | Parent runs (time noise is about 0.1 s) |
| `KX_TRAIN_SHARDS` | 9 | Training shards of 100M tokens |
| `KX_RUN_TIMEOUT_S` | 1500 | First run of each code; later runs get half |

If the time limit gets close, the eval stops starting new runs and scores what it has, as long as it
has at least 4 candidate runs.

## Notes

- Cost: a node's scoring takes about 100–110 minutes of one a3-highgpu-8g (the base's 10 runs about 70), around $50/hour on spot,
  plus 5–20 minutes waiting for the machine. Measured on the base: the first run 672 s of wall time, later ones 392 s.
- Kupanga cuts the changes diff at 400,000 characters. A candidate that changes more than that can't
  have its parent rebuilt, so keep each node to one technique. (#92's whole diff is 548,000.)
- Local check, no GPU: `python bench.py --parse <log.txt> ...` parses saved speedrun logs and prints the
  summary. #87's ten logs give 80.61 s, val_loss 3.2785, p = 0.0047.
