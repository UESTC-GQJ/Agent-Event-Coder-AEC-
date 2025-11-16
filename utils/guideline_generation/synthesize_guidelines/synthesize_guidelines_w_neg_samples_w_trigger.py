from tabulate import tabulate
from tqdm import tqdm
import argparse
import json
import re
import os
import random

MAX_SAMPLES = 5

dataset_mapper = {
    "ACE": {
        "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/Sep_master_data_ace05-en.json", 
        "def_file": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/def_Sep_master_data_ace05-en.json", 
        "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/Master_event_dataclasses_ace05-en.json", 
        "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/ace05-en_event_type_files"
    }
}

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_name", help="Dataset Name", default="ACE")


PROMPT = """You are an expert in annotating NLP datasets for event extraction.Your task is to generate precise and detailed annotation guidelines for the event type [###Event_Type###]

### Input Format ###
```
Event Schema:
Event Name and its parent class
Arguments:
Arguments separated by new lines. If there are no arguments None will be given.

Examples:
Examples provided to illustrate the event type and arguments.
```

### Instructions ###
1. Identify and List All Unique Arguments: 
   - Carefully review the schema to identify all arguments relevant to the event type.
   - Please remember that the examples may not cover all the arguments in the list. In some cases, you may not have arguments at all, in such cases, you can have an empty list for arguments. 
2. Define the Event Type: Write 5 clear and specific definitions, starting with "The event is triggered by ...":
   - Include example triggers.
   - Highlight key characteristics and scope of the event.
   - Compare and contrast with closely related events using provided negative examples.
   - Explain how triggers and outcomes differ for similar event types.
3. Identify the Event Trigger (Mention Argument): 
    - Define the event trigger ("mention") with 5 specific definitions, and include examples of valid triggers while addressing variations, edge cases, and distinctions from related event types.
4. Define Each Argument:** For each argument, provide 5 definitions with detailed examples, starting with "Examples are ...":
   - Explain the role and importance of each argument.
   - Include domain knowledge and address edge cases to clarify ambiguities.
5. Focus on Distinctions: Use positive examples to describe the event, and negative examples to clarify what the event is not. Explicitly state differences using phrases like:
   - "Unlike [Related Event Type], this event does not ..."
   - "Triggers such as [Trigger] are indicative of [Related Event Type], not this event type."
6. Structured Output: Present the output in the following JSON format:
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
       "Argument2": [
         "Definition 1",
         "Definition 2",
         "Definition 3",
         "Definition 4",
         "Definition 5"
       ]
       // Add more arguments if applicable
     }
   }
   ```

### Output Requirements ###
- Use detailed yet concise language for event and argument definitions.
- Incorporate diverse and domain-relevant examples for each definition.
- Avoid copying examples directly from provided data, create unique variations.

"""
def create_prompt(event_type, positive_examples, negative_examples, display_name, all_guidelines):
    demos = f"\n"
    schema_information = all_guidelines[event_type]
    schema_information_desc = f"Event Schema:\n{event_type.replace('(', ' which is a child event type of super class ').replace(')', '')}\n"
    schema_information_desc += "Arguments:\n"

    # Use a separate counter to correctly number arguments
    arg_counter = 1
    for key in schema_information["attributes"].keys():
        if key == "mention":
            continue
        schema_information_desc += f"Argument {arg_counter} -> {key}\n"
        arg_counter += 1

    # Add positive examples
    demos += "### The below examples are positive examples, as they match the Event Type being annotated: ###\n\n"
    for idx, example in enumerate(positive_examples):
        text = example["text"]
        event_mentions = example["event_mentions"][event_type]["results"]
        for event_mention in event_mentions:
            trigger = event_mention["mention"]
            arguments = [(key, value) for key, value in event_mention.items() if key != "mention"]
            demos += f"Example {idx + 1}\n"
            demos += f"#### Event Type ####\n{event_type}\n"
            demos += f"### Input Text ###\n{text.strip()}\n"
            demos += f"### Event Trigger ###\n{trigger.strip()}\n"
            demos += "### Event Arguments ###\n"
            for arg_key, arg_val in arguments:
                demos += f"For argument \"{arg_key}\" extracted spans {arg_val}\n"
            demos += "\n"

    # Add negative examples
    # demos += "### The below examples are negative examples, as they are from different Event Types for contrast: ###\n\n"
    demos += "### The following examples are negative examples, as they illustrate different event types provided for contrast and differentiation: ###\n\n"
    for idx, example in enumerate(negative_examples, start=len(positive_examples) + 1):
        negative_event_type = next(iter(example["event_mentions"]))
        text = example["text"]
        event_mentions = example["event_mentions"][negative_event_type]["results"]
        for event_mention in event_mentions:
            trigger = event_mention["mention"]
            arguments = [(key, value) for key, value in event_mention.items() if key != "mention"]
            demos += f"Example {idx}\n"
            demos += f"#### Event Type ####\n{negative_event_type}\n"
            demos += f"### Input Text ###\n{text.strip()}\n"
            demos += f"### Event Trigger ###\n{trigger.strip()}\n"
            demos += "### Event Arguments ###\n"
            for arg_key, arg_val in arguments:
                demos += f"For argument \"{arg_key}\" extracted spans {arg_val}\n"
            demos += "\n"

    prompt = PROMPT.replace("[###Event_Type###]", event_type.replace("(", " which is a child event type of super class ").replace(")", ""))
    prompt += schema_information_desc.strip() + "\n\n" + demos.strip()
    return prompt

def create_examples(event_type_dict, event_type, full_argument_list):
    print("****"*50)
    print("Positive Examples")
    def extract_field_types(results):
        field_types = set()
        for result in results:
            field_types.update(result.keys())
        return field_types

    sorted_data = sorted(event_type_dict, key=lambda x: len(x["event_mentions"][event_type]["results"]), reverse=True)

    all_fields = set(full_argument_list) - set(["mention"]) # Assuming 'full_argument_list' contains all fields required
    covered_fields = set()
    selected_examples = []

    while covered_fields != all_fields and len(selected_examples) < MAX_SAMPLES:
        # print(covered_fields)
        # print(all_fields)
        # print('-'*100)
        best_example = None
        best_new_fields = set()

        for example in sorted_data:
            if example in selected_examples:
                continue  # Skip examples already added
            results = example["event_mentions"][event_type]["results"]
            # print(results)
            # xx
            fields = extract_field_types(results)- set(["mention"])
            # print(fields)
            # xx
            new_fields = fields - covered_fields
            if new_fields and len(new_fields) > len(best_new_fields):
                best_example = example
                best_new_fields = new_fields

        if best_example:
            selected_examples.append(best_example)
            covered_fields.update(best_new_fields)
    
    if len(selected_examples) < MAX_SAMPLES:
        remaining_examples = [ex for ex in sorted_data if ex not in selected_examples]
        remaining_slots = MAX_SAMPLES - len(selected_examples)
        selected_examples.extend(remaining_examples[:remaining_slots])

    # formatted_examples = tabulate(selected_examples, headers="keys")
    print(f"Finished event type {event_type} with {len(selected_examples)} samples. Arguments were: {full_argument_list}. Total {len(selected_examples)} number of samples")
    print(f"All positive examples look like: {selected_examples}")
    return selected_examples#formatted_examples

#############################################################################################################################################################
# ############# This below is the function to add negative samples such that if I want to add 10 samples then I will from 4 different event types ###################
#############################################################################################################################################################
# def create_negative_examples(all_event_types, current_event_type, dataset_name, num_negatives):
#     print("****" * 50)
#     print("Negative Examples")

#     # Exclude the current event type and 'None' from the pool of event types to select negatives
#     selectable_event_types = [et for et in all_event_types if et != current_event_type and et != "None"]
#     print(f"Selectable event types for negatives: {selectable_event_types}")

#     negative_examples = []
#     num_types = min(4, len(selectable_event_types))  # Determine the number of event types to select
#     num_selected_per_type = max(1, num_negatives // num_types)  # Ensure at least one negative per type

#     print(f"Number of negatives to select per type: {num_selected_per_type}")

#     # Randomly select different event types to fetch negatives from
#     selected_event_types = random.sample(selectable_event_types, num_types)
#     print(f"Selected event types for negatives: {selected_event_types}")

#     # Function to count attributes in an example's results section
#     def count_attributes_in_results(example, event_type):
#         results = example['event_mentions'][event_type]['results']
#         return max(len(result.keys()) for result in results) if results else 0

#     # Fetch examples from each of the selected event types
#     for event_type in selected_event_types:
#         file_path = f"{dataset_mapper[dataset_name]['example_dir']}/{event_type}.json"
#         try:
#             with open(file_path, 'r') as file:
#                 examples = [json.loads(line) for line in file]
                
#                 # Sort examples by the number of attributes in the results section
#                 examples_sorted = sorted(
#                     examples,
#                     key=lambda ex: count_attributes_in_results(ex, event_type),
#                     reverse=True
#                 )

#                 print(f"Loaded and sorted {len(examples_sorted)} examples from {event_type}")

#                 # Select top examples based on attribute coverage
#                 num_to_select = min(num_selected_per_type, len(examples_sorted))
#                 selected_samples = examples_sorted[:num_to_select]
#                 negative_examples.extend(selected_samples)

#                 print(f"Selected {len(selected_samples)} negatives from {event_type}")
#                 for sample in selected_samples:
#                     attr_count = count_attributes_in_results(sample, event_type)
#                     print(f"Example with {attr_count} attributes: {sample['text']}")
#         except FileNotFoundError:
#             print(f"Warning: File not found for event type {event_type} at {file_path}")

#     # print("Final negative examples list:")
#     # for idx, example in enumerate(negative_examples):
#     #     event_type_key = next(iter(example['event_mentions']))
#     #     attr_count = count_attributes_in_results(example, event_type_key)
#     #     print(f"Negative Example {idx + 1} ({event_type_key} with {attr_count} attributes): {example['text']}")
#     # print("-----" * 50)
#     print("Negative examples list:")
#     print(negative_examples)  # Directly print the list of all negative examples

#     print("****" * 50)
#     return negative_examples


############################################################################################################################################################
# ############# This below is the function to add negative samples such that I add n example from 15 randomly choosen event types ###################
############################################################################################################################################################
def create_negative_examples(all_event_types, current_event_type, dataset_name, num_negatives, examples_per_event=1):
    print("****" * 50)
    print("Negative Examples")

    # Exclude the current event type and 'None' from the pool of event types to select negatives
    selectable_event_types = [et for et in all_event_types if et != current_event_type and et != "None"]
    print(f"Selectable event types for negatives: {selectable_event_types}")

    # Randomly select up to 15 event types (or fewer if there are less than 15 available)
    selected_event_types = random.sample(selectable_event_types, min(15, len(selectable_event_types)))
    print(f"Randomly selected event types for negatives: {selected_event_types}")

    negative_examples = []

    # Function to count attributes in an example's results section
    def count_attributes_in_results(example, event_type):
        results = example['event_mentions'][event_type]['results']
        return max(len(result.keys()) for result in results) if results else 0

    # Fetch examples from each of the selected event types
    for event_type in selected_event_types:
        file_path = f"{dataset_mapper[dataset_name]['example_dir']}/{event_type}.json"
        try:
            with open(file_path, 'r') as file:
                examples = [json.loads(line) for line in file]
                
                # Sort examples by the number of attributes in the results section
                examples_sorted = sorted(
                    examples,
                    key=lambda ex: count_attributes_in_results(ex, event_type),
                    reverse=True
                )

                print(f"Loaded and sorted {len(examples_sorted)} examples from {event_type}")

                # Select up to `examples_per_event` examples from this event type
                num_to_select = min(examples_per_event, len(examples_sorted))
                selected_samples = examples_sorted[:num_to_select]
                negative_examples.extend(selected_samples)

                print(f"Selected {len(selected_samples)} negatives from {event_type}")
                for sample in selected_samples:
                    attr_count = count_attributes_in_results(sample, event_type)
                    print(f"Example with {attr_count} attributes: {sample['text']}")
        except FileNotFoundError:
            print(f"Warning: File not found for event type {event_type} at {file_path}")

    print("Negative examples list:")
    print(negative_examples)  # Directly print the list of all negative examples

    print("****" * 50)
    return negative_examples

# #############################################################################################################################################################
# ############ This below is the function to add negative samples such that I add 1 examples each from only the sibling event types  ###################
# #############################################################################################################################################################
# def create_negative_examples(all_event_types, current_event_type, dataset_name, num_negatives, examples_per_event=1):
#     print("****" * 50)
#     print("Negative Examples")

#     # Extract the superclass of the current event type
#     current_superclass = current_event_type.split('(')[-1].rstrip(')')
#     print(f"Superclass of current event type: {current_superclass}")

#     # Filter sibling event types
#     sibling_event_types = [
#         et for et in all_event_types
#         if et != current_event_type and et.split('(')[-1].rstrip(')') == current_superclass
#     ]
#     print(f"Sibling event types for negatives: {sibling_event_types}")

#     # Randomly select up to 15 sibling event types
#     selected_event_types = random.sample(sibling_event_types, min(15, len(sibling_event_types)))
#     print(f"Randomly selected sibling event types for negatives: {selected_event_types}")

#     negative_examples = []

#     # Function to count attributes in an example's results section
#     def count_attributes_in_results(example, event_type):
#         results = example['event_mentions'][event_type]['results']
#         return max(len(result.keys()) for result in results) if results else 0

#     # Fetch examples from the selected sibling event types
#     for event_type in selected_event_types:
#         file_path = f"{dataset_mapper[dataset_name]['example_dir']}/{event_type}.json"
#         try:
#             with open(file_path, 'r') as file:
#                 examples = [json.loads(line) for line in file]

#                 # Sort examples by the number of attributes in the results section
#                 examples_sorted = sorted(
#                     examples,
#                     key=lambda ex: count_attributes_in_results(ex, event_type),
#                     reverse=True
#                 )

#                 print(f"Loaded and sorted {len(examples_sorted)} examples from {event_type}")

#                 # Select up to `examples_per_event` examples per sibling event type
#                 num_to_select = min(examples_per_event, len(examples_sorted))
#                 selected_samples = examples_sorted[:num_to_select]
#                 negative_examples.extend(selected_samples)

#                 print(f"Selected {len(selected_samples)} negatives from {event_type}")
#                 for sample in selected_samples:
#                     attr_count = count_attributes_in_results(sample, event_type)
#                     print(f"Example with {attr_count} attributes: {sample['text']}")
#         except FileNotFoundError:
#             print(f"Warning: File not found for event type {event_type} at {file_path}")

#     print("Negative examples list:")
#     print(negative_examples)  # Directly print the list of all negative examples

#     print("****" * 50)
#     return negative_examples


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = args.dataset_name
    #this file has key value pairs of event and attributes
    all_guidelines = json.load(open(dataset_mapper[dataset_name]["event_arg_ontology"]))
    with open(dataset_mapper[dataset_name]["def_file"]) as f:
        event_type_dict = json.load(f)

    all_event_types = list(event_type_dict.keys())

    for event_type in event_type_dict:
        print("event_type")
        print(event_type)
        if event_type == "None":
            continue
        full_argument_list = event_type_dict[event_type]["Arguments"]
        # print(full_argument_list)
        # xx
        examples = create_examples([json.loads(x) for x in open(f"{dataset_mapper[dataset_name]['example_dir']}/{event_type}.json")], event_type, full_argument_list)
        # negative_examples = create_negative_examples(all_event_types, event_type, dataset_name, 10)  # Example with num_negatives set to 10
        negative_examples = create_negative_examples(all_event_types,event_type,dataset_name,num_negatives=10,examples_per_event=1)
        # prompt = create_prompt(event_type, examples, event_type_dict[event_type]["DisplayName"], all_guidelines)
        prompt = create_prompt(event_type,positive_examples=examples, negative_examples=negative_examples,display_name=event_type_dict[event_type]["DisplayName"], all_guidelines=all_guidelines)
        os.makedirs(f"prompts/{dataset_name}", exist_ok=True)
        with open(f"prompts/{dataset_name}/prompt_" + event_type+".txt", "w") as f:
            f.write(prompt)
        # print(f"Covered event type {event_type} with ")
        # print("-*"*100)
    # print(event_type_dict)