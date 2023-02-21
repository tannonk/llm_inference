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

As input, `inference.py` expects a jsonl file containing the following dictionary object on each line:

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