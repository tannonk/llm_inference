## Prompts

We save pre-defined prompts is a simple JSON format.

The keys correspond to the names of arguments expected by `llm_inference.py`. For example:

```json
{
    "prompt_prefix": "Rewrite the complex sentence with simple sentence(s). Keep the meaning same, but make it simpler.",
    "prompt_suffix": "Complex: {input}\nSimple:",
    "prompt_template": "Complex: {complex}\nSimple: {simple}",
    "example_separator": "\n\n",
    "prompt_format": "prefix_initial",
    "source_field": "complex",
    "target_field": "simple"
}
```

To use a pre-defined prompt, simply pass the path of the `prompt.json` file to `inference.py` with the arugment `--prompt_json`.

Note, even though LangChain allows you to load pre-defined prompts, we found that doing it this way allows for more flexibility.