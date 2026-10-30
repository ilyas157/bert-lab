# Accelerating BERT training on HPC

Fine-tuning `google-bert/bert-base-uncased` on SQuAD v1.1 with PyTorch on NVIDIA A100 GPUs
on the FinisTerrae III supercomputer (CESGA), as part of the Máster in HPC (HPC Tools course).

| Deliverable | What | Result |
|---|---|---|
| [BASELINE](BASELINE/) | Single A100: FP32 baseline, then mixed precision, batch size and `torch.compile`, with profiling | **6.0×** faster than FP32 (49 → 8.2 min), F1 87.6 |
| [DISTRIBUTED](DISTRIBUTED/) | DistributedDataParallel on 1 node × 2 A100 and 2 nodes × 2 A100, with profiling | **3.0×** on 4 GPUs (75% efficiency), **14.8×** vs FP32 baseline, F1 87.0 |

Each folder has its own README with the method, the SLURM scripts, the logs and the analysis.
Each deliverable is tagged (`BASELINE`, `DISTRIBUTED`).

**Stack:** PyTorch 2.11 (CUDA 12.8), Hugging Face `transformers`/`datasets`, NCCL, SLURM, TensorBoard, `torch.profiler`.
