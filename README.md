## LLM Inference

Experimental repository for inference with LLMs supported by HuggingFace.

## Setup

For environment setup commands, see `setup_env.sh`.

## Examples

```bash
python inference.py \
    --model_name_or_path "bigscience/bloom-560m" \
    --max_new_tokens 500 \
    --few_shot_n 3 \
    --delimiter '\\n\\n' \
    --input_file examples/rrgen.jsonl
```

The argument `--input_file` should be a JSONL file containing the following dictionary-like object on each line:

```json
{
    "examples": 
        [
            "few-shot prompt 1", 
            "few-shot prompt 2", 
            "few-shot prompt 3", 
            "etc."
        ], 
    "src": "src text / prompt for continuation"
}
```