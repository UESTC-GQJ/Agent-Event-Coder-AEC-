import re
import os
import copy
import json
import random
import argparse
from utils import *
from glob import glob
from tqdm import tqdm

PY_ADD = "./../code_schema_generation/python_event_defs/"
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="./../../TextEE/processed_data")
parser.add_argument("--dataset_name", type=str, default="ace05-en")
parser.add_argument("--annotate_schema", default=False)
parser.add_argument("--guideline_file", type=str, default="ace05-en")
parser.add_argument("--add_negative_samples", default=False)
parser.add_argument("--output_dir", type=str, default="./processed_code_prompts/")
parser.add_argument("--negative_sample_count", type=int, default=15)


all_annotated_schema = {}
def annotate_schema_fun(schema, guidelines, do_print = False):
    assert type(schema) == str, "Schema should be a string"
    if(do_print):
        print("Schema: ", schema)
        # xxx
    event_name, annotated_guidelines = None, None
    annotated_schema = ""
    for line in schema.split("\n"):
        if(line.strip()=="@dataclass"):
            annotated_schema += line + "\n"
        elif(line.strip().startswith("class")):
            # event_name = line.replace("class", "").replace(":", "").replace("(", "").replace(")", "").replace("Event", "").strip()
            event_name = line.split("(")[0].replace("class", "").strip()
            # print("Event Name: ", event_name)
            annotated_guidelines = guidelines[event_name]
            docstring = random.choice(annotated_guidelines["description"])
            annotated_schema += line + f"\n    \"\"\"{docstring}\"\"\"\n"
        elif (line.strip()==""):
            continue
        else:
            arg_name = line.split(":")[0].strip()
            if(arg_name=="mention"):
                random_comments = "The text span that triggers the event."#random.choice(annotated_guidelines["Event Definition"])
                line = f"    {arg_name}: str # {random_comments}"
            else:
                random_comments = random.choice(annotated_guidelines["attributes"][arg_name])
                line = f"    {arg_name}: List # {random_comments}"
            annotated_schema += line + "\n"
    if(all_annotated_schema.get(event_name) is None):
        all_annotated_schema[event_name] = []
    all_annotated_schema[event_name].append(annotated_schema)
    # print(annotated_schema)
    assert type(annotated_schema) == str, "Annotated schema should be a string"
    return annotated_schema
        
text_sep = "# This is the text to analyze"
task_sep = "# The following lines describe the task definition"
def group_by_event_type(sample):
    new_samples = []
    current_text = sample["text"]
    current_wnd_id = sample["wnd_id"]
    if(len(sample["event_mentions"]) == 0):# if there are no event mentions, return the sample as is
        return [sample]
    #otherwise, group the events by their event type
    grouped_events = {}
    for event in sample["event_mentions"]:
        event_type = event["event_type"]
        if(event_type not in grouped_events):
            grouped_events[event_type] = dict([(key, val) for key, val in event.items() if key != "event_mentions"])
            grouped_events[event_type].update({"event_mentions": []})# set it to empty list so that we can group by event type
        grouped_events[event_type]["event_mentions"].append(event)
    for event_type, event in grouped_events.items():
        new_event = copy.deepcopy(event)
        new_event["text"] = current_text
        new_event["wnd_id"] = current_wnd_id
        new_samples.append(new_event)
    return new_samples

#given an input text, converts it into code format.
def prepare_input(sample, text, event_schemas, task_type, dataset_name, is_dev_test_data = False, current_event = None, do_annotate_schema = False, guidelines = None):
    footer = task_to_prompts[task_type]["footer"]
    try:
        event_name = sample["event_mentions"][0]["event_type"]# get the current event name
    except:
        event_name = random.choice(list(event_schemas.keys()))# if there is no event name, get a random event name. This happens for non-event instances
    if(is_dev_test_data):# when we are preparing the test/dev datasets, we "may" pass the current event name because we are going to enumerate over all the events.
        event_name = current_event
    original_event_name = event_name + ""# a copy for debug
    event_name = clean_event_name(event_name, dataset_name).split("(")[0].strip()#normalize the event name
    schema = event_schemas[event_name]# we pass the list of schemas for the event name and extract corresponding to current event type
    annotated_schema = schema if not do_annotate_schema else annotate_schema_fun(schema, guidelines, False)# optionally, add guidelines to the schema
    # print(f"Annotated Schema:\n{annotated_schema}")
    # xxx
    prompt = task_sep + "\n\n" + annotated_schema + "\n\n" + text_sep + f"\ntext = \"{text}\"\n\n{footer}"
    instruction = task_to_prompts[task_type]["header"]
    return instruction, prompt    

def prepare_test_dev_input(text, sample, event_schemas, event_name, dataset_name, is_dev_test_data = False, do_annotate_schema = False, guidelines = None):
    header = task_to_prompts[dataset_to_task_mapper[dataset_name]]["header"]
    footer = task_to_prompts[dataset_to_task_mapper[dataset_name]]["footer"]
    new_instances = []
    if(event_name is None):# if there is no event name, get a random event name. This happens for non-event instances
        for event in all_annotated_schema:
            annotated_schema = random.choice(all_annotated_schema[event])
            prompt = task_sep + "\n\n" + annotated_schema + "\n\n" + text_sep + f"\ntext = \"{text}\"\n\n{footer}"
            instruction = header
            new_instance = copy.deepcopy(sample)
            new_instance["input"] = prompt
            new_instance["instruction"] = instruction
            new_instance["output"] = "[]"
            new_instances.append(new_instance)
    else:# 
        # print(sample)
        event_name = clean_event_name(event_name, dataset_name).split("(")[0].strip()#normalize the event name
        instruction, prompt = prepare_input(sample, text, event_schemas, dataset_to_task_mapper[dataset_name], dataset_name, is_dev_test_data = is_dev_test_data, current_event = event_name, do_annotate_schema = do_annotate_schema, guidelines=guidelines)
        assert type(prompt) == str, "Prompt should be a string"
        new_instance = copy.deepcopy(sample)
        new_instance["input"] = prompt
        new_instance["instruction"] = header
        if not is_dev_test_data:
            new_instance["output"] = prepare_train_output(sample["event_mentions"], dataset_name)
        else:
            new_instance["output"] = "[]"
        eval(new_instance["output"])# if this doesn't work: I have written something wrong, please reach out.
        new_instances.append(new_instance)
    return new_instances


def prepare_train_output(labels, dataset_name):
    event_schema = load_clean_schema(dataset_name)
    if(len(labels) == 0):
        return "[]"
    output_string = "["
    for label in labels:
        event_name = label["event_type"]
        event_name = clean_event_name(event_name, dataset_name).split("(")[0].strip()
        output_string += event_name + f"(mention={label['trigger']['text']!r}, "
        arg_gold = {}
        for g_arg in label["arguments"]:
            g_arg["role"] = g_arg["role"].lower()
            arg_name = g_arg["role"]
            if(arg_name not in arg_gold):
                arg_gold[arg_name] = []
            arg_gold[arg_name].append(g_arg["text"])
        for schema_arg in event_schema[event_name]:
            if(schema_arg in arg_gold):
                arg_string = f"{schema_arg}={arg_gold[schema_arg]}"
            else:
                arg_string = f"{schema_arg}=[]"
            output_string += arg_string + ", "
        output_string += "), "
    output_string +=  "]"
    output_string = output_string[:-4] + ")]"
    return output_string.replace(", )", ")")

def prepare_test_dev_dataset(dataset, dataset_name, do_annotate_schema = False, is_dev_test_data = False, add_negative_sample = False, guidelines = None):
    all_schema = load_init_prompts(dataset_name)
    # print(all_schema.keys())
    # xxx
    event_counts = 0
    new_dataset = []
    for sample in tqdm(dataset, total = len(dataset)):
        events = set([clean_event_name(x["event_type"], dataset_name) for x in sample["event_mentions"]])
        # clean_event_name(event_name, dataset_name).split("(")[0].strip()
        # event_counts += (dataset_event_counts[dataset_name])# - len(events))
        
        if(len(sample["event_mentions"])==0):# if there are no event mentions, then simply enumerate all the events" (text, sample, event_name, dataset_name, is_dev_test_data = False):
            # print("Here....")
            # print(json.dumps(sample, indent=4))
            non_event_instances = prepare_test_dev_input(sample["text"], sample, all_schema, None, dataset_name, is_dev_test_data = True, guidelines=guidelines, do_annotate_schema = do_annotate_schema)
            new_dataset.extend(non_event_instances)
            continue
        grouped_events, covered_events = {}, set()
        for event in sample["event_mentions"]:
            event_type = event["event_type"]
            covered_events.add(clean_event_name(event_type, dataset_name).split("(")[0].strip())
            if(event_type not in grouped_events):
                grouped_events[event_type] = dict([(key, val) for key, val in event.items() if key != "event_mentions"])
                grouped_events[event_type].update({"event_mentions": []})
            grouped_events[event_type]["event_mentions"].append(event)
        for event_type, event in grouped_events.items():
            new_event_instances = prepare_test_dev_input(sample["text"], event, all_schema, event_type, dataset_name, guidelines=guidelines, do_annotate_schema = do_annotate_schema)
            new_dataset.extend(new_event_instances)
        uncovered_events = set(all_schema.keys()) - covered_events
        # print("Uncovered Events: ", uncovered_events, len(uncovered_events), "covered_events: ", covered_events, len(covered_events))
        # xxx
        for event_type in uncovered_events:
            new_non_event_instance = prepare_test_dev_input(sample["text"], sample, all_schema, event_type, dataset_name, do_annotate_schema=do_annotate_schema, guidelines=guidelines, is_dev_test_data = True)
            new_dataset.extend(new_non_event_instance)
    # print("Event Counts: ", event_counts)
    return new_dataset



def annotate_dataset(dataset, dataset_name, task_type, do_annotate_schema = False, is_dev_test_data = False, guidelines = None):
    all_schema = load_init_prompts(dataset_name)
    # print(all_schema.keys())
    new_dataset = []
    for sample in tqdm(dataset, total = len(dataset)):
        instruction, code_input_text = prepare_input(sample, sample["text"], all_schema, task_type, dataset_name, is_dev_test_data = is_dev_test_data, current_event = None, do_annotate_schema = do_annotate_schema, guidelines = guidelines)
        label = prepare_train_output(sample["event_mentions"], dataset_name)
        assert type(code_input_text) == str, "Code input text should be a string"

        sample["input"] = code_input_text
        sample["instruction"] = instruction
        sample["output"] = label
        ### verify
        hhh = eval(label) # if this doesn't work: I have written something wrong, please reach out.
        # print(f"Verified: {len(hhh)}")
        ### verify
        new_dataset.append(sample)
    return new_dataset


def prepare_dataset(input_dir, dataset_name, add_negative_sample = False, annotate_schema = False, guidelines = None, output_dir = "./processed_code_prompts/", nagative_sample_count = None):
    dataset = load_dataset(input_dir, dataset_name)
    train_data, dev_data, test_data = dataset["train"], dataset["dev"], dataset["test"]
    task_type = dataset_to_task_mapper[dataset_name]
    grouped_train, grouped_dev, grouped_test = [], [], []
    annotated_train, annotated_dev, annotated_test = [], [], []
    if(add_negative_sample):
        train_negative_samples = get_negative_samples(train_data, dataset_name, guidelines, nagative_sample_count, do_annotate_schema=annotate_schema)
    for sample in train_data:
        grouped_train.extend(group_by_event_type(sample))
    # for sample in dev_data:                              ---
    #     grouped_dev.extend(group_by_event_type(sample))    |
    #                                                         \
    #                                                           - Not Needed but kept for debugging
    #                                                         / 
    # for sample in test_data:                               |
    #     grouped_test.extend(group_by_event_type(sample)) ---
    annotated_train = annotate_dataset(grouped_train, dataset_name, dataset_to_task_mapper[dataset_name], annotate_schema, guidelines=guidelines)
    annotated_dev = prepare_test_dev_dataset(dev_data, dataset_name, do_annotate_schema=annotate_schema, guidelines=guidelines)
    annotated_test = prepare_test_dev_dataset(test_data, dataset_name, do_annotate_schema=annotate_schema, guidelines=guidelines)
    clean_train = final_clean_dataset(annotated_train, dataset_name)
    clean_dev = final_clean_dataset(annotated_dev, dataset_name)
    clean_test = final_clean_dataset(annotated_test, dataset_name)
    if(add_negative_sample):
        clean_train = clean_train + train_negative_samples
    ###
    save_dataset(output_dir, dataset_name, {"train": clean_train,
                                           "dev":   clean_dev,
                                           "test":  clean_test})

if __name__ == "__main__":
    args = parser.parse_args()
    # dir = "./../../TextEE/processed_data"
    dataset_name = args.dataset_name
    dir = args.input_dir
    annotate_schema = args.annotate_schema
    add_negative_samples = args.add_negative_samples
    guidelines_files = args.guideline_file
    output_dir = args.output_dir
    ###
    import_star(dataset_name, PY_ADD)### used to verify if the output follows the python format.
    ###
    guidelines = None
    if(annotate_schema):
        if(not guidelines_files.endswith(".json")):    
            guidelines = load_guidelines(f"./../guidelines/{dataset_name}_event_guidelines.json")
        else:
            guidelines = load_guidelines(guidelines_files)
    prepare_dataset(dir, dataset_name, add_negative_sample = add_negative_samples, annotate_schema = annotate_schema, guidelines=guidelines, output_dir = output_dir, nagative_sample_count=args.negative_sample_count)
    print("Length of all annotated schema: ", len(all_annotated_schema))
