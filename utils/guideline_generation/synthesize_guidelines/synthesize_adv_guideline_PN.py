
from tqdm import tqdm
import argparse
import json
import os

# Dataset mapper (adjust the paths as needed for your datasets)
dataset_mapper = {
    "richere-en": {
        "output_dir": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/synthesized_guidelines/richere-en_sibling_examples/",
        "advanced_prompt_dir": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/prompts/richere-en_15sibling_adv_guidelines/"
        # "output_dir": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/synthesized_guidelines/richere-en_random_examples/",
        # "advanced_prompt_dir": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/prompts/richere-en_15random_adv_guidelines/"
    }
}

# Prompt template
PROMPT = """You are an expert in summarizing NLP event extraction guidelines. Your goal is to consolidate multiple detailed descriptions into a single concise, comprehensive "Advanced" guideline.

### Input Format ###
Event Type: Event Type Name
```json
{
  "Event Definition": [
    "Definition 1",
    "Definition 2",
    "Definition 3",
    "Definition 4",
    "Definition 5"
  ],
  "Arguments Definitions": {
    "mention": [
      "Definition 1",
      "Definition 2",
      "Definition 3",
      "Definition 4",
      "Definition 5"
    ],
    "Argument1": [
      "Definition 1",
      "Definition 2",
      "Definition 3",
      "Definition 4",
      "Definition 5"
    ],
    // Add additional arguments as necessary
  }
}
```

### Task ###
1. Consolidate the 5 definitions under "Event Definition" into a single definition:
   - Highlight all critical points and examples from the five definitions.
   - Ensure the description is concise, comprehensive, and clear, using formal language that non-experts can understand.

2. Do the same for each argument under "Arguments Definitions," producing a single advanced definition for each. 

### Output Format ###
```json
{
  "Event Definition": "Consolidated advanced guideline for the event type.",
  "Arguments Definitions": {
    "mention": "Consolidated advanced guideline for the mention argument.",
    "Argument1": "Consolidated advanced guideline for Argument1.",
    "Argument2": "Consolidated advanced guideline for Argument2."
    // Add additional arguments as applicable
  }
}
```

### Guidelines to Summarize ###
Event Type: [###Event_Type###]
```json
[###Detailed_Guidelines###]
```
"""

def create_advanced_prompt(event_type, detailed_guidelines):
    # Generates a prompt asking GPT-4o to consolidate five detailed guidelines into one advanced guideline.
    advanced_prompt = PROMPT.replace("[###Event_Type###]", event_type)
    guidelines_json = json.dumps(detailed_guidelines, indent=4)
    advanced_prompt = advanced_prompt.replace("[###Detailed_Guidelines###]", guidelines_json)
    return advanced_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", help="Dataset Name", default="richere-en")
    args = parser.parse_args()

    # Get dataset-specific directories
    dataset_name = args.dataset_name
    dataset_info = dataset_mapper[dataset_name]

    # Input and output directories
    input_dir = dataset_info["output_dir"]
    output_dir = dataset_info["advanced_prompt_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Process all JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    for json_file in tqdm(json_files, desc="Generating Advanced Prompts"):
        with open(os.path.join(input_dir, json_file), "r") as infile:
            detailed_guidelines = json.load(infile)

        event_type = json_file.replace(".json", "")
        advanced_prompt = create_advanced_prompt(event_type, detailed_guidelines)

        output_file = os.path.join(output_dir, f"adv_{event_type}.txt")
        with open(output_file, "w") as outfile:
            outfile.write(advanced_prompt)


    # "ACE": {
    #     # "output_dir": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/synthesized_guidelines/ACE_15sibling_samples/",
    #     # "advanced_prompt_dir": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/prompts/ACE_15sibling_adv_guidelines/"
    #     "output_dir": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/synthesized_guidelines/ACE_15random_samples/",
    #     "advanced_prompt_dir": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/prompts/ACE_15random_adv_guidelines/"
    # }
