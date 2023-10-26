# LLM Outputs for BLESS

This folder contains model outputs and evaluation results generated.

Note, the generated outputs for Newsela are **not included** in this repository due to the strict licensing agreement for using the dataset. If you would like to use these outputs, please first obtain access to the [Newsela corpus](https://newsela.com/data/) and then contact us directly.

## Files Included

For each model that we investigate, we generated the following files:

- `<output_file>.jsonl`: The generated outputs of the model on the input file.
- `<output_file>.json`: Command line arguments used for the inference run.
- `<output_file>.log`: Log file of the inference run (not included in this repo).
- `<output_file>.eval`: Log file of the automatic evaluation with results.

The `<output_file>` naming convention follows the format:

```
[eval dataset]_[example dataset]_[prompt ID]_[few-shot example selection method]_[# few-shot examples]_[# reference sentences in few-shot examples]_[random seed]
```

For example, `asset-test_asset-valid_p0_random_fs3_nr1_s489.jsonl` indicates:

- outputs for ASSET test set
- few-shot examples taken from ASSET validation set
- pre-defined prompt `p0` (defined [here](https://github.com/tannonk/llm_inference/blob/main/prompts/p0.json))
- few-shot examples were randomly sampled
- the prompt is constructed on the fly with 3 few-shot examples
- each few-shot example is shown with 1 ground-truth reference sentence
- random seed used was 489

## Output Format

The output format is a verbose `json-lines` file.

Each line contains a dictionary-like object with the keys: `input_prompt`, `model_output`, `source`, `references`. Note, we include the original source sentence and ground-truth references from the test set to facilitate post-hoc evaluation!

Here's an example:

```json
{
    "input_prompt": "Rewrite the complex sentence with simple sentence(s). Keep the meaning same, but make it simpler.\n\nComplex: This may be seen in the demand to remove the chief of staff of the Austrian Army, Alfred Jansa, from his position in January 1938.\nSimple: There was a demand to remove the Austrian Army chief of staff from office in 1938. His name was Alfred Jansa.\n\nComplex: Oxnard is the largest city in Ventura County, California in terms of population.\nSimple: Oxnard is the most populated city in Ventura County, California.\n\nComplex: 1263 - The chieftains of the eastern part of Iceland become the last to pledge fealty to the Norwegian king, bringing a more complete end to the Icelandic Commonwealth and the Icelandic civil war.\nSimple: The leaders of the eastern part of Iceland become the last to pledge loyalty to the Norwegian king, bringing a more complete end to the Icelandic Commonwealth and the Icelandic civil war.\n\nComplex: Gable also earned an Academy Award nomination when he portrayed Fletcher Christian in 1935's Mutiny on the Bounty.\nSimple:", 
    "model_output": "He received an Academy Award nomination for his portrayal of Fletcher Christian in 1935's Mutiny on the Bounty.", 
    "source": "Gable also earned an Academy Award nomination when he portrayed Fletcher Christian in 1935's Mutiny on the Bounty.", 
    "references": [
        "Gable earned an Academy Award nomination for portraying Fletcher Christian in Mutiny on the Bounty.", 
        "Gable also earned an Oscar nomination when he portrayed Fletcher Christian in 1935's Mutiny on the Bounty.", 
        "Gable won an Academy Award vote when he acted in 1935's Mutiny on the Bounty as Fletcher Christian.", 
        "Gable also won an Academy Award nomination when he played Fletcher Christian in the 1935 film Mutiny on the Bounty.", 
        "Gable was nominated for an Academy Award for portraying Fletcher Christian in 1935's Mutiny on the Bounty.", 
        "Gable also earned an Academy Award nomination in 1935 for playing Fletcher Christian in \"Mutiny on the Bounty\".", 
        "Gable also earned an Academy Award nomination when he played Fletcher Christian in 1935's Mutiny on the Bounty.", 
        "Gable recieved an Academy Award nomination for his role as Fletcher Christian. The film was Mutiny on the Bounty (1935).", 
        "Gable earned an Academy Award nomination for his role as Fletcher Christian in the 1935 film Mutiny on the Bounty.", 
        "Gable also got an Academy Award nomination when he played Fletcher Christian in 1935's movie, Mutiny on the Bounty."
 ]
}
```
