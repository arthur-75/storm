# STORM: Stepwise Token Optimization with Reward-Guided Beam Search

Official implementation of **STORM**, a self-supervised method for lexical query expansion introduced in:

> Arthur Satouf, Giulio D'Erasmo, Yuxuan Zong, Habiboulaye Amadou Boubacar, Pablo Piantanida, and Benjamin Piwowarski.  
> **STORM: Stepwise Token Optimization with Reward-Guided Beam Search.** EMNLP 2026 Main Conference.

[[Paper](https://arxiv.org/abs/2606.10621)] [[PDF](https://arxiv.org/pdf/2606.10621)] [[Models](https://huggingface.co/collections/Arthur-75/storm)]

STORM trains an LLM query rewriter using retrieval feedback. During training, partial keyword sequences are repeatedly scored against a BM25 index; low-reward branches are pruned before generation continues. The retrieval reward therefore guides both token-level exploration and importance-weighted policy updates. At inference time, the model produces short keyword expansions that can be issued to a standard BM25 index without re-encoding the corpus.

This repository contains two workflows:

- `train/`: self-training a causal language model to generate query-expansion keywords, using retrieval quality as the reward.
- `eval/`: generating multilingual query expansions and evaluating retrieval on MIRACL & BEIR.

The checked-in YAML files contain cluster-specific placeholder paths. Copy a config and update its paths before running it.

## Paper setup at a glance

The main experiments use the following setup. The provided YAML is an editable experiment configuration, so verify it against this table when reproducing a reported result.

| Setting | Paper configuration |
| --- | --- |
| Backbones | Qwen3-0.6B, 1.7B, 4B, and 8B |
| Training data | Approximately 80k MS MARCO passage-retrieval training queries |
| Training supervision | BM25 retrieval reward with dense cross-encoder pseudo-labels |
| Holdout | 500 queries excluded from optimization for checkpoint selection |
| Optimization | AdamW, one epoch, learning rate `1e-6`, batch size 32, gradient accumulation 8 |
| Training generation | Maximum 32 tokens; reward-guided beam width 5 mixed with nucleus sampling |
| Inference generation | 6 beams, 3 beam groups, diversity penalty 1.0, 3 returned sequences |
| Numerical precision | Float32 for training and evaluation |
| Released evaluation workflows | BEIR and zero-shot MIRACL evaluation with nDCG@10 |
| Hardware used |  NVIDIA RTX A6000/H100 GPU|

The suffixes STORM32, STORM64, and STORM128 in the paper denote the inference-time maximum number of generated tokens. The three returned rewrite sequences are concatenated into one keyword string and submitted to BM25 in a single retrieval call.

## 1. Environment setup

Python 3.10 or 3.11 and a CUDA-capable machine are recommended. Pyserini also requires a working Java runtime (Java 21 is recommended for current Pyserini releases).

```bash
python -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch transformers peft accelerate datasets \
  pyyaml tqdm numpy scipy pandas numba \
  beir ir-datasets pyserini pytrec-eval
java -version
```

Use versions of PyTorch and Transformers that are compatible with the CUDA drivers on the target machine. If `pytrec-eval` is unavailable from the package index, install it from its upstream repository or through Conda.

All commands below are run from the directory that contains the cloned `storm/` repository:

```bash
cd /path/to/parent-of-storm
source env/bin/activate
```

### Required Transformers modification (Important for training)

**This step is required for training.** STORM uses a modified Hugging Face generation implementation for reward-guided beam search. After installing the dependencies, replace the installed Transformers generation utility with the version provided by this repository:

```bash
cp storm/train/hf_utils/utils.py \
  env/lib/python3.10/site-packages/transformers/generation/utils.py
```

The destination above assumes that the virtual environment is named `env` and uses Python 3.10. If your environment is elsewhere, locate the correct destination with:

```bash
python -c "import transformers.generation.utils as u; print(u.__file__)"
```

Copy `storm/train/hf_utils/utils.py` to the printed path. This modification is tied to the installed Transformers version; reinstalling or upgrading Transformers may overwrite it, in which case it must be copied again.

## 2. Train

Edit `storm/train/config/config_qwen8b.yaml`. At minimum, replace these machine-specific values:

| Config key | Meaning |
| --- | --- |
| `yaml_file` | Root directory in which the experiment config is copied. |
| `pipeline.base_model_name_or_path` | Hugging Face model ID or local base-model directory. |
| `data.pkl_path` | Data directory. Include the trailing `/` because the current code concatenates filenames directly. |
| `data.dl_data_root` | Optional root containing `trec-dl-2019`, `trec-dl-2020`, and `trec-dl-hard`; remove or set to `null` to disable those evaluations. |
| `training.output_dir` | Checkpoint directory template. |
| `training.last_ckpt` | Directory containing the resume marker `last_ckpt.json`. |
| `logging.debug_log_path` | Training debug-log path. |
| `eval.log_path`, `eval.dump_jsonl` | Holdout-evaluation outputs. |

`data.pkl_path` must contain:

```text
<data.pkl_path>/
├── msmarco_queries_qrels.pkl   # pickle containing (queries, qrels)
├── dictCE.tsv                  # cross-encoder scores used by the reward
├── dict_ndcg.json              # baseline NDCG by query ID
└── msmarco_docIndex            # Pyserini Lucene BM25 index
```

The two score files have different roles:

- `dictCE.tsv` contains the prepared relevance judgments used to compute the training reward. They are derived from the cross-encoder scores in the [OpenSearch MS MARCO Hard Negatives LLM Scores dataset](https://huggingface.co/datasets/opensearch-project/msmarco-hard-negatives-llm-scores) and normalized by the maximum score.
- `dict_ndcg.json` contains the nDCG@100 score of every initial query under those cross-encoder judgments. These baseline scores are saved so that each generated query can be compared with its corresponding initial query in the JSON diagnostic logs; this file is primarily for training diagnostics.

If `training.msmarco_v` is non-empty, it is appended to `msmarco_docIndex`. For SPLADE rewards, set `reward.retrieval_type: splade` and update `reward.index_path` and `reward.splade_query_encoder`.

Useful switches are:

- `data.max_queries`: limit the number of training queries for a smoke test.
- `pipeline.run_generation` / `pipeline.run_training`: enable the two training stages.
- `eval.enabled`: run holdout evaluation during optimization.
- `lora.enabled: false`: fine-tune and save the full model.
- `lora.enabled: true`: fine-tune a parameter-efficient LoRA adapter. Configure `r`, `alpha`, `dropout`, and `target_modules` in the `lora` section.
- `device`, `model_parallel`, and generation/training batch sizes: adapt these to the available GPUs.

Training uses float32 in the paper configuration.

### Training prompt

Training uses the following prompt, with Chain-of-Thought reasoning disabled:

```text
From the query generate new semantic related keywords.
Output the result strictly as a single comma-separated line.
[QUERY]: {query}
[KEYWORDS]:
```

The first two lines are the system message. The original query and the `[KEYWORDS]:` marker are passed in the user message.

Run training:

```bash
python -u storm/train/run_qwen_multi.py \
  --config storm/train/config/config_qwen8b.yaml
```

The runner copies the effective YAML to `<yaml_file>/<exp_name>/config.yaml`. Checkpoints are saved under `training.output_dir`; `training.last_ckpt/last_ckpt.json` is used to resume the next invocation automatically.

For a small smoke test, set `data.max_queries` to a small value, reduce both generation and evaluation batch sizes, set `pipeline.num_iterations: 1`, and increase `training.save_steps` if frequent checkpoints are unnecessary.

### Reproducing the paper's training regime

For a paper-aligned run, select one of the Qwen3 checkpoints (`0.6B`, `1.7B`, `4B`, or `8B`) and ensure the following values are set:

```yaml
generation:
  max_new_tokens: 32
  num_beams: 5

training:
  learning_rate: 1.0e-6
  num_train_epochs: 1
  gradient_accumulation_steps: 8

eval:
  enabled: true
  num_queries: 500
```

Set the generation batch size to 32 to match the paper when memory permits. The checked-in 8B example currently uses a smaller per-device generation batch and larger gradient accumulation, which is useful on memory-constrained hardware but is not the exact paper configuration.

## 3. Evaluation

Evaluation supports BEIR and MIRACL. Both use the same STORM generation and BM25 retrieval pipeline. In the selected experiment YAML, set:

- `experiment.name`: the experiment name used in output paths.
- `model.base_model`: the Qwen3 Hugging Face model ID or local model directory.
- `model.adapter_path`: the trained STORM checkpoint. It may be a LoRA checkpoint containing `adapter_config.json` or a full checkpoint containing `config.json`.
- `paths.output_root`: destination for generated queries and evaluation results.
- `generation.community_tools_path`: use `storm/eval/community_tools` when running from the parent directory.
- `model.dtype`: keep this set to `float32`. All reported STORM evaluations use float32.

### Maximum generation length

The maximum generation length depends on the model size and applies to both BEIR and MIRACL:

| Qwen3 backbone | `generation.query_decoding.max_new_tokens` |
| --- | ---: |
| 0.6B | **32** |
| 1.7B | **32** |
| 4B | **64** |
| 8B | **64**/**128** |

### Inference decoding

The reported experiments use **grouped beam search** (diverse beam search). It was slightly better than the alternatives in our experiments, but it is not required to run STORM. Standard beam search can be used with `num_beam_groups: 1` and `diversity_penalty: 0.0`. Greedy decoding is also supported by setting `num_beams: 1`, `num_beam_groups: 1`, `num_return_sequences: 1`, and `diversity_penalty: 0.0`.

The grouped-beam-search configuration used in the paper is:

```yaml
generation:
  query_decoding:
    max_new_tokens: 64  # use 32 for 0.6B/1.7B; 64 for 4B/8B
    num_beams: 6
    num_beam_groups: 3
    diversity_penalty: 1.0
    num_return_sequences: 3
    repetition_penalty: 1.0
    do_sample: false
```

### Evaluation prompts

BEIR uses the following prompt(same as the Training):

```text
From the query generate new semantically related keywords.
Output the result strictly as a single comma-separated line.
[QUERY]: {query}
[KEYWORDS]:
```

MIRACL uses the same format with an additional target-language instruction:

```text
From the query generate new semantic related keywords.
Output the result strictly as a single comma-separated line.
The keywords must be in {language}.
[QUERY]: {query}
[KEYWORDS]:
```

`{query}` is replaced by the input query and `{language}` by the language of the MIRACL dataset.

### BEIR

Edit `storm/eval/configs/experiment_beir.yaml`. Set `paths.data_root` to the BEIR datasets, `paths.original_index_root` to the BM25 Lucene indexes, and update the model and output paths described above.

Run:

```bash
source env/bin/activate

python storm/eval/run_retrieval_pipeline.py \
  --config storm/eval/configs/experiment_beir.yaml
```

### MIRACL

Edit `storm/eval/configs/experiment_miracl.yaml`. Set `paths.miracl_root` to the local MIRACL topics, qrels, and Lucene indexes, then update the model and output paths described above.

Activate the environment and run MIRACL evaluation from the directory containing `storm/`:

```bash
source env/bin/activate

python storm/eval/run_retrieval_pipeline.py \
  --config storm/eval/configs/experiment_miracl.yaml
```

The pipeline generates multilingual keyword expansions and evaluates them with BM25. Outputs, per-language summaries, and the global MIRACL summary are written below `paths.output_root`.

## Results

All values below are taken from the paper. STORM results use the Qwen3-8B backbone and float32 inference.

### BEIR results

nDCG@10 on the 12 out-of-domain BEIR datasets:

| Method | NFC | SCID | SciF | Touché | FiQA | Covid | Signal | News | RB04 | HotpotQA | NQ | DBP | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 32.2 | 14.9 | 67.9 | 44.2 | 23.6 | 59.5 | 33.1 | 39.5 | 40.8 | 63.3 | 30.6 | 31.8 | 40.1 |
| RM3 | 33.1 | 14.7 | 64.6 | 35.0 | 19.2 | 59.3 | 31.5 | 42.6 | 44.3 | 51.3 | 23.1 | 30.8 | 37.5 |
| SPLADE-v2 | 34.7 | 15.9 | 70.4 | 24.7 | 34.7 | 72.7 | 30.1 | 41.5 | 46.8 | 68.7 | 53.8 | 43.7 | 44.8 |
| HyDE-8B | 31.2 | 13.4 | 68.2 | 33.9 | 16.3 | 58.3 | 19.7 | 36.5 | 41.4 | 49.7 | 33.4 | 30.0 | 36.0 |
| MuGI-8B | 35.4 | 15.3 | 71.7 | 46.3 | 24.5 | 69.0 | 34.8 | 46.3 | 48.4 | 65.7 | 44.8 | 39.8 | 45.2 |
| W2P-8B | 36.6 | 15.4 | 70.8 | 49.1 | 26.3 | 74.4 | 35.0 | 47.7 | 50.4 | 67.0 | 45.2 | 39.3 | 46.4 |
| QUESTER-8B | 36.3 | 15.2 | 71.3 | 41.4 | 26.3 | 69.8 | 30.5 | 46.8 | 53.1 | 63.4 | 42.3 | 36.0 | 44.4 |
| **STORM-8B (64 tokens)** | **36.8** | **15.8** | **72.8** | **44.4** | **28.4** | **79.6** | **33.5** | **50.2** | **56.0** | **65.9** | **46.3** | **40.1** | **47.5** |

### Comparison with GPT-4.1 rewriters

nDCG@10 on three in-domain and five BEIR datasets. STORM uses Qwen3-8B; the other LLM rewriters use GPT-4.1.

| Method | DL'19 | DL'20 | DL-HARD | SciFact | Covid | FiQA | DBP | News | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RM3 | 52.2 | 49.0 | 25.1 | 64.6 | 59.3 | 19.2 | 30.8 | 42.6 | 42.9 |
| GenQR | 54.8 | 53.7 | 29.2 | 72.6 | 68.7 | 23.0 | 34.4 | 46.5 | 47.9 |
| GenQREnsemble | 55.9 | 55.3 | 27.0 | 72.5 | 75.3 | 23.9 | 36.0 | 48.6 | 49.3 |
| QA-Expand | 68.3 | 64.2 | 30.2 | 70.6 | 70.7 | 26.4 | 37.0 | 45.0 | 51.6 |
| Q2K | 59.4 | 57.6 | 34.5 | 70.9 | 71.5 | 26.9 | 37.8 | 46.3 | 50.6 |
| Q2D (zero-shot) | 68.7 | 66.2 | 35.0 | 72.0 | 74.3 | 26.0 | 40.6 | 49.8 | 54.1 |
| LameR | 63.7 | 65.3 | 35.5 | 72.5 | 70.2 | 26.2 | 39.9 | 48.0 | 52.7 |
| MuGI | 69.5 | 65.8 | 36.5 | 73.5 | 71.4 | 26.4 | 41.0 | 51.6 | 54.5 |
| CSQE | 69.0 | 65.5 | 36.6 | 72.1 | 69.9 | 24.7 | 39.0 | 47.9 | 53.1 |
| **STORM-8B (64 tokens)** | **66.1** | **67.3** | **34.2** | **72.8** | **79.6** | **28.4** | **40.1** | **50.2** | **54.8** |

### MIRACL results

Zero-shot nDCG@10 on all 18 MIRACL languages. STORM is trained exclusively on English MS MARCO.

| Method | ar | bn | en | es | fa | fi | fr | hi | id | ja | ko | ru | sw | te | th | zh | de | yo | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 48.1 | 50.8 | 35.1 | 31.9 | 33.3 | 55.1 | 18.3 | 45.8 | 44.9 | 36.9 | 41.9 | 33.4 | 38.3 | 49.4 | 48.4 | 18.0 | 22.6 | 40.6 | 38.5 |
| mDPR | 49.9 | 44.3 | 39.4 | 47.8 | 48.0 | 47.2 | 43.5 | 38.3 | 27.2 | 43.9 | 41.9 | 40.7 | 29.9 | 35.6 | 35.8 | 51.2 | 49.0 | 44.4 | 42.1 |
| mContriever | 52.5 | 50.1 | 36.4 | 41.8 | 21.5 | 60.2 | 31.4 | 28.6 | 39.2 | 42.4 | 48.3 | 39.1 | 56.0 | 52.8 | 51.7 | 41.0 | 40.8 | 41.5 | 43.1 |
| mColBERT | 57.1 | 54.6 | 38.8 | 42.6 | 46.0 | 46.5 | 26.7 | 47.0 | 29.8 | 49.6 | 48.7 | 47.7 | 35.8 | 46.2 | 48.1 | 39.8 | 33.4 | 56.1 | 44.1 |
| HyDE-8B | 33.8 | 40.5 | 31.1 | 25.6 | 40.2 | 65.2 | 20.5 | 33.8 | 49.3 | 29.8 | 35.0 | 47.6 | 22.9 | 58.5 | 58.1 | 34.8 | 24.6 | 44.6 | 38.7 |
| **STORM-8B (64 tokens)** | **57.4** | **55.1** | **51.7** | **42.5** | **38.7** | **61.3** | **33.4** | **52.1** | **50.6** | **44.8** | **51.4** | **46.6** | **44.9** | **53.2** | **54.8** | **34.3** | **36.8** | **52.2** | **47.9** |
| **STORM-8B (128 tokens)** | **57.4** | **58.6** | **52.3** | **43.2** | **39.4** | **61.8** | **33.3** | **52.7** | **51.4** | **45.7** | **51.2** | **47.1** | **43.4** | **54.7** | **56.5** | **33.6** | **37.5** | **50.8** | **48.4** |

See Table 2 of the [paper](https://arxiv.org/pdf/2606.10621) for per-language results and the additional generation-length analysis. For the released evaluation recipe in this repository, use 32 tokens for the 0.6B/1.7B models and 64 tokens for the 4B/8B models.

## Quick example notebook

[`quick_run_example.ipynb`](quick_run_example.ipynb) contains a small example for loading STORM and running query expansion on one input. After activating the environment, open it with Jupyter:

```bash
source env/bin/activate
jupyter notebook storm/quick_run_example.ipynb
```

## Troubleshooting

- **CUDA out of memory:** lower the generation batch sizes or adjust `device_map`; keep `float32` to match the reported setup.
- **Java/Pyserini error:** confirm `java -version`, `python -c "import pyserini"`, and that configured Lucene index paths exist.
- **Model or adapter cannot be loaded:** verify that the base-model path is readable and that the checkpoint contains either `adapter_config.json` (LoRA) or `config.json` (full model).
- **Artifacts unexpectedly reused:** change the relevant stage from `auto` to `force`, or choose a new `paths.output_root`.
- **Training path not found:** ensure `data.pkl_path` ends in `/`; the training code currently forms several paths by string concatenation.

## Citation

If you use STORM, please cite:

```bibtex
@article{satouf2026storm,
  title   = {STORM: Stepwise Token Optimization with Reward-Guided Beam Search},
  author  = {Satouf, Arthur and D'Erasmo, Giulio and Zong, Yuxuan and
             Amadou Boubacar, Habiboulaye and Piantanida, Pablo and
             Piwowarski, Benjamin},
  journal = {arXiv preprint arXiv:2606.10621},
  year    = {2026},
  doi     = {10.48550/arXiv.2606.10621}
}
```
