## Experiment Configs

This directory contains predefined experiment settings for various models. The purpose of these config file is to facilitate inference run experiments, specifically on a slurm cluster.

We specify model-specific arguments such as GPU memory requirements in the appropriate JSON file, which can be passes to `slurm_scripts.submit_inference` as a positional argument in position 1.

We then specify experiment specific arguments as usual, e.g.:

```bash
python -m run exp_configs/rtx/bloom-560m.json \
    --prompt_json "prompts/p0.json" \
    --examples "data/asset/dataset/asset.valid.jsonl" \
    --input_file "data/asset/dataset/asset.test.orig" \
    --n_refs 1 --few_shot_n 3 --seed 489 
```
Where `rtx` is a directory containing experiment configs for a server with 8 RTX 3090 GPUs (24GB). It's assumed that these GPUs support parallelisation.

The directory `cluster` contains config settings for inference experiments on a SLURM cluster with 8x A100s (80GB) and T4s (16GB) GPUs. Here, we always use A100s for parallelisation.

To adjust for your own GPU configuration, simply create a new folder and edit the appropriate config file.

## Observations

Below are some observations from running inference with LLMs:

1. Beam search uses significantly more GPU memory compared to sampling-based decoding methods with `num_beams=1`
<!-- 2. Inference with `bigscience/bloom-560m` on a single T4 (16GB) GPU takes ~10 seconds per batch with `batch_size=4, max_new_tokens=100, num_beams=1, num_return_sequences=1, do_sample=True, top_p=0.9` and uses ~6GB GPU memory -->
<!-- 3. Inference with `bigscience/bloom-1b1` on a single T4 (16GB) GPU takes ~10 seconds per batch with `batch_size=4, max_new_tokens=100, num_beams=1, num_return_sequences=1, do_sample=True, top_p=0.9` and uses ~8GB GPU memory -->
2. The model footprint (loaded as 8bit-int) is roughly 1GB per 1B parameters. To run batched inference, you need to account for sufficient headroom.
3. Inference with GPT-NeoX is very slow! (~ 1.5 hours on ASSET test set (359))

The following table contains some statistics observed during test inference runs with the following params (unless otherwise specified): `batch_size=8, max_new_tokens=100, num_beams=1, num_return_sequences=1, do_sample=True, top_p=0.9`

|           model           | Footprint (8bit-int) |  Inference time  | Inference GPU mem |             # GPUS             |       rtx          |        cluster     |
| :-----------------------: | :------------------: | :--------------: | :---------------: | :----------------------------: | :----------------: | :----------------: |
| bigscience/bloom-560m     |      0.78 GB         | ~10 secs (bs=4)  |        ~6GB       | 1 T4-16GB                      | :white_check_mark: |                    |
| bigscience/bloomz-560m    |      0.78 GB         | ~10 secs (bs=4)  |        ~6GB       | 1 T4-16GB                      | :white_check_mark: |                    |
| bigscience/bloom-1b1      |      1.35 GB         | ~10 ses (bs=4)   |        ~8GB       | 1 T4-16GB                      | :white_check_mark: |                    |
| bigscience/bloomz-1b1     |      1.35 GB         | ~10 ses (bs=4)   |        ~8GB       | 1 T4-16GB                      | :white_check_mark: |                    |
| bigscience/bloom-3b       |      3.39 GB         |                  |                   | 1 T4-16GB                      | :white_check_mark: |                    |
| bigscience/bloomz-3b1     |      3.39 GB         |                  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| bigscience/bloom-7b1      |      7.54 GB         |                  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| bigscience/bloom          |    167.5 GB	       | ~45 secs (bs=8)  |                   | 4 A100-80GB                    |                    |                    |
| bigscience/bloomz         |    167.5 GB	       | ~45 secs (bs=8)  |                   | 4 A100-80GB                    |                    |                    |
| facebook/opt-1.3b         |      1.32 GB         |                  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| facebook/opt-iml-max-1.3b |      1.32 GB         |                  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| facebook/opt-6.7b         |      6.40 GB         |                  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| facebook/opt-13b          |     12.22 GB         |                  |                   | 1 RTX 3090-24GB                | :white_check_mark: |                    |
| facebook/opt-30b          |     28.26 GB         | ~27 secs (bs=8)  |       ~60GB       | 4 RTX 3090-24GB / 1 A100-80GB  | :white_check_mark: |                    |
| facebook/opt-iml-max-30b  |     28.26 GB         | ~27 secs (bs=8)  |       ~60GB       | 4 RTX 3090-24GB / 1 A100-80GB  | :white_check_mark: |                    |
| facebook/opt-66b          |     61.65 GB         | ~40 secs (bs=8)  |       ~150GB      | 2 A100-80GB                    | :white_check_mark: |                    |
| facebook/llama-7B         |      6.58 GB         | ~6 secs (bs=8)   |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| facebook/llama-13b        |     12.5 GB          |                  |                   | 1 RTX 3090-24GB / 1 A100-80GB  | :white_check_mark: |                    |
| facebook/llama-30b        |     30.81 GB         | ~116 secs (bs=8) |                   | 4 RTX 3090-24GB / 1 A100-80GB  | :white_check_mark: |                    |
| facebook/llama-65b        |     61.45 GB         |                  |                   | 7 RTX 3090-24GB / 1 A100-80GB  | :white_check_mark: |                    |
| EleutherAI/gpt-j-6b       |      6.13 GB         | ~16 secs (bs=8)  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| EleutherAI/gpt-neox-20b   |     61.45 GB         |                  |                   | 3 RTX 3090-24GB / 1 A100-80GB  | :white_check_mark: |                    |
|                           |                      |                  |                   |                                |                    |                    |
|         Enc-Dec           |                      |                  |                   |                                |                    |                    |
|                           |                      |                  |                   |                                |                    |                    |
| ul2 (20b)                 |     30.24 GB         |  ~30 secs (bs=8) |                   | 2 RTX 3090-24GB / 1 A100-80GB  | :white_check_mark: |                    |
| flan-ul2 (20b)            |     30.24 GB         |                  |                   | 2 RTX 3090-24GB / 1 A100-80GB  | :white_check_mark: |                    |
| t0_3b                     |      4.18 GB         |                  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| t0 (11b)                  |     16.24 GB         |                  |                   | 1 RTX 3090-24GB / 1 A100-80GB  | :white_check_mark: |                    |
| t0pp (11b)                |     16.24 GB         |                  |                   | 1 RTX 3090-24GB / 1 A100-80GB  | :white_check_mark: |                    |
| t5-small (77m)            |      0.12 GB         |                  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| t5-base (250m)            |      0.38 GB         |                  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| t5-large (780m)           |      1.17 GB         |                  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| t5-xl (3b)                |      4.18 GB         |                  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| t5-xxl (11b)              |     16.24 GB         |                  |                   | 1 RTX 3090-24GB / 1 A100-80GB  | :white_check_mark: |                    |
| flan-t5-small (77m)       |      0.12 GB         |                  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| flan-t5-base (250m)       |      0.38 GB         |                  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| flan-t5-large (780m)      |      1.17 GB         |                  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| flan-t5-xl (3b)           |      4.18 GB         |                  |                   | 1 RTX 3090-24GB / 1 T4-16GB    | :white_check_mark: |                    |
| flan-t5-xxl (11b)         |     16.24 GB         |                  |                   | 1 RTX 3090-24GB / 1 A100-80GB  | :white_check_mark: |                    |

## Notes

- T0* models fail to run with `protobuf==4.22.1`. Downgraded to `protobuf==3.20.0` to fix.

<!-- t0_3b
```
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
``` -->
