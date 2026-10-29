# speedrun-8xh100

A Kupanga benchmark on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)'s Track 1
speedrun: train GPT-2 (124M) on FineWeb to a val loss of **3.28** as fast as possible on **8×H100 SXM**.
Each scoring times the candidate against its parent's code **on the same machine**, as every speedrun PR does.

| File | What it is |
|---|---|
| `eval.toml` | Runs `python bench.py`; `train_time_s` is the primary metric, lower is better |
| `job.toml` | `python:3.12`, 8×H100 (GCP a3-highgpu-8g, spot, 3 retries), 32+ CPUs, 256+ GB, 90 min limit |
| `requirements.txt` | The speedrun's pinned stack: torch 2.10.0+cu128, `kernels` (FA3 from the HF hub), scipy |
| `bench.py` | Fetches the data, rebuilds the parent, runs both N times, parses logs, runs the t-test |

## What it does

1. Downloads `kjj0/fineweb10B-gpt2` (the val shard and 9 training shards), as `python
   data/cached_fineweb10B.py 9` would, into a read-only folder it gives the run as `DATA_PATH`.
2. Rebuilds the parent's code by reverse-applying `$HARNESS_CHANGES_FILE` (what the candidate changed
   since its experiment started) to a copy of the subject. The base node has no parent, so it is
   timed on its own.
3. Runs `torchrun --standalone --nproc_per_node=8 train_gpt.py` 10 times for the candidate and 4 times
   for the parent, alternating (c p c c p c c p …), so drift on the machine affects both. The
   compile cache is shared, so only the first run of each code compiles (about 7 minutes).
4. Reads each run's last `step:N/N val_loss:X train_time:Yms` line from `logs/<uuid>.txt`.
   `train_time` leaves out compiling and warmup, as on the leaderboard.
5. Reports:
   - the candidate's mean `train_time_s` and its standard error
   - the mean `val_loss`
   - the speedrun's `p_value` from `scipy.stats.ttest_1samp(losses, 3.28, alternative="less")`
   - `time_delta_s`, the candidate minus the parent on this machine

A candidate whose mean val_loss isn't below 3.28 with p < 0.01 gets **no score**, because a time only
counts at the target loss. The error message still gives its times. If a run crashes or times out, the
candidate gets no score either, and the message shows the run's last output.

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
| `KX_RUNS` | 10 | Candidate runs. At #87's spread (about 0.0013), 8 runs only just reach p < 0.01 |
| `KX_BASELINE_RUNS` | 4 | Parent runs (time noise is about 0.1 s) |
| `KX_TRAIN_SHARDS` | 9 | Training shards of 100M tokens |
| `KX_RUN_TIMEOUT_S` | 1500 | First run of each code; later runs get half |

If the time limit gets close, the eval stops starting new runs and scores what it has, as long as it
has at least 4 candidate runs.

## Notes

- Cost: about 60–75 minutes of one a3-highgpu-8g, around $50/hour on spot.
- Kupanga cuts the changes diff at 400,000 characters. A candidate that changes more than that can't
  have its parent rebuilt, so keep each node to one technique. (#92's whole diff is 548,000.)
- Local check, no GPU: `python bench.py --parse <log.txt> ...` parses saved speedrun logs and prints the
  summary. #87's ten logs give 80.61 s, val_loss 3.2785, p = 0.0047.
