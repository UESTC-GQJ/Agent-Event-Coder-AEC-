import importlib
import random
import json
import sys
import re
import os
dataset_to_task_mapper = {
    "ace05-en": "e2e",
    "speed": "ed",
    "genia2013": "e2e",
    "maven": "ed",
    "rams": "eae",
    "fewevent": "ed",
    "casie": "e2e",
    "geneva": "eae",
    "mlee": "e2e",
    "genia2011": "e2e",
    "m2e2": "e2e",
    "phee": "e2e",
    "richere-en": "e2e",
    "muc4": "eae",
    "mee": "ed",
    "wikievents": "eae"
}

schema_splitters = {
    "ace05-en": ":",
    "speed": "None",
    "genia2013": "None",
    "maven": "None",
    "rams": ".",
    "fewevent": ".",
    "casie": ":",
    "geneva": "None",
    "mlee": "None",
    "genia2011": "None",
    "m2e2": ":",
    "phee": "None",
    "richere-en": ":",
    "muc4": "None",
    "mee": "_",
    "wikievents": "None"
}

dataset_event_counts = {
    "ace05-en": 33,
    "speed": 7,
    "wikievents": 50,
}

# EAE needs special cleaning. We need to execute the python-code output. After doing ouput=eval(output), we can extract output.mention for trigger and output.__class__.__name__ for event type.
# once triggers are extracted we cam format the input prompt by replacing the trigger with the mention and event type with the class name.
def final_clean_EAE_dataset(dataset):
    clean_EAE_dataset = []
    # print("Final clean ", len(dataset))
    for d in dataset:
        current_output = d["output"]
        ex_outputs = eval(current_output)
        input = d["input"]
        input=input.replace("{trigger}", "__TMP_TRIGGER__").replace("{event_name}", "__TMP_EVENT__")
        input = input.replace("{", "{{").replace("}", "}}")
        input = input.replace("__TMP_TRIGGER__", "{trigger}").replace("__TMP_EVENT__", "{event_name}")
        if(len(ex_outputs)<=0):
            clean_EAE_dataset.append(d)
            continue
        for output in ex_outputs:
            event_type = output.__class__.__name__
            trigger = output.mention
            # print(input, "<", trigger, "<", event_type, "<")
            formatted_input = input.format(trigger = trigger, event_name = event_type)
            new_instance = {"input": formatted_input, "output": str([output])}
            clean_EAE_dataset.append(new_instance)
    return clean_EAE_dataset


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
    # print(annotated_schema)
    assert type(annotated_schema) == str, "Annotated schema should be a string"
    return annotated_schema

PY_ADD = "./../PyCode-TextEE/code_schema_generation/python_event_defs/"
def final_clean_dataset(dataset, dataset_name):
    import_star(dataset_name, PY_ADD)
    clean_dataset = []
    for d in dataset:
        new_instance = {"input": d["input"], "output": d["output"]}
        clean_dataset.append(new_instance)
        if(d.get("covered_events") is not None):
            new_instance["covered_events"] = d["covered_events"]
    # print("from clean_data before ", len(clean_dataset))
    if(dataset_to_task_mapper[dataset_name] == "eae"):
        clean_dataset = final_clean_EAE_dataset(clean_dataset)
    # print("from clean_data after ", len(clean_dataset))
    return clean_dataset


def clean_event_name(event_type, dataset_name):
    splitter = schema_splitters[dataset_name]
    event_type = event_type.replace("n/a", "Na")
    event_type_parts = event_type.split(splitter)
    if len(event_type_parts) > 1:
        parent_event_name = event_type_parts[0].replace("-", "").replace(".", "_")
        if(dataset_name=="casie"):
            parent_event_name = parent_event_name.title()
        event_type = "_".join(event_type_parts[1:]).replace("-", "").replace(".", "_")
        if(dataset_name=="rams"):
            event_type = f"{parent_event_name}_{event_type}(Event)"
        else:
            parent_event_name = parent_event_name + "Event"
            event_type = f"{event_type}({parent_event_name})"
    else:
        original_event_type = event_type + ""
        event_type = event_type.replace("-", "").replace(".", "_") + "(Event)"
        if(dataset_name=="speed"):
            event_type = event_type.title()
            
    return event_type

task_to_prompts = {
    "e2e":{
        "header": "# This is an event extraction task where the goal is to extract structured events from the text. A structured event contains an event trigger word, an event type, the arguments participating in the event, and their roles in the event. For each different event type, please output the extracted information from the text into python-style dictionaries where the first key will be 'mention' with the value of the event trigger. Next, please output the arguments and their roles following the same format. The event type definitions and their argument roles are defined next.",
        "footer":"# The list called result should contain the instances for the following events according to the guidelines above:\nresult = \n"
    },
    "ed":{
        "header": "# This is an event detection task where the goal is to identify event triggers and their types in the text. For each event, please output the extracted information into python-style dictionaries where the key is 'mention' with the value of the event trigger. The event type definitions are defined next.",
        "footer":"# The list called result should contain the instances for the following events according to the guidelines above:\nresult = \n"
    },
    "eae":{
        "header": "# This is an event argument extraction task where the goal is to extract the arguments of a given event trigger in the text. The event trigger and its type are provided. Please output the extracted arguments and their roles into python-style dictionaries. The event type definitions and their argument roles are defined next.",
        "footer":"# The list called result contains the instances for the following events according to the guidelines above\n# 1. \"{trigger}\" triggers a {event_name} event.\n\nresult = \n"
    },
}


def load_init_prompts(dataset_name):
    init_prompt_address = f"./../code_schema_generation/init_prompts/{dataset_name}.txt"
    init_prompt = "".join(open(init_prompt_address).readlines())
    prompt_mapper = {}
    sep_classes = [x.strip() for x in init_prompt.split("\n\n")]
    for classes in sep_classes:
        for lines in classes.split("\n"):
            if(lines.startswith("class")):
                event_name = lines.split("(")[0].replace("class ", "").strip()
                prompt_mapper[event_name] = classes
    return prompt_mapper


def load_event_defs(dataset_name: str, PY_ADD: str):
    sys.path.append(PY_ADD)                       # 1. make Python see the folder
    mod = importlib.import_module(f"{dataset_name}_definitions_new")  # 2. import
    globals().update(mod.__dict__)                # 3. expose Attack, Die, … globally
    return mod 

def load_clean_schema(dataset_name):
    class_pat  = re.compile(r'^class\s+(\w+)\s*\(')   # captures class name
    field_pat  = re.compile(r'^\s+(\w+)\s*:')         # captures each field
    file_path = f"./../code_schema_generation/init_prompts/{dataset_name}.txt"
    event_args = {}
    current = None
    with open(file_path, encoding="utf‑8") as fh:
        for line in fh:
            # start of a new @dataclass block ─ ignore the decorator line itself
            if line.startswith("@dataclass"):
                current = None
                continue

            # new class definition
            m_class = class_pat.match(line)
            if m_class:
                current = m_class.group(1)
                event_args[current] = []
                continue

            # attribute lines that belong to the current class
            if current:
                m_field = field_pat.match(line)
                if m_field:
                    name = m_field.group(1)
                    if name != "mention":          # skip the generic field
                        event_args[current].append(name.lower())
                # blank line ⇒ class body ended
                elif not line.strip():
                    current = None
    return event_args


import sys, importlib, inspect

def import_star(dataset_name: str, path: str):
    if path not in sys.path:
        sys.path.append(path)
    mod = importlib.import_module(f"{dataset_name}_definitions_new")
    names = getattr(mod, "__all__", [n for n in mod.__dict__ if not n.startswith("_")])
    caller_globals = inspect.currentframe().f_back.f_globals
    caller_globals.update({n: getattr(mod, n) for n in names})
    return mod         # still handy to have


def load_guidelines(guideline_address):
    with open(guideline_address, "r") as f:
        guidelines = json.load(f)
    return guidelines


def load_dataset(input_dir, dataset_name):
    train_file = os.path.join(input_dir, dataset_name, "split1", "train.json")
    dev_file = os.path.join(input_dir, dataset_name, "split1", "dev.json")
    test_file = os.path.join(input_dir, dataset_name, "split1", "test.json")
    train_data = [json.loads(x) for x in open(train_file)]
    dev_data = [json.loads(x) for x in open(dev_file)]
    test_data = [json.loads(x) for x in open(test_file)]
    return {"train":train_data, "dev": dev_data, "test": test_data}

def save_dataset(input_dir, dataset_name, data):
    # input_dir = "./"
    os.makedirs(os.path.join(input_dir, dataset_name), exist_ok=True)
    train_file = os.path.join(input_dir, dataset_name, "train.json")
    dev_file = os.path.join(input_dir, dataset_name, "dev.json")
    test_file = os.path.join(input_dir, dataset_name, "test.json")
    with open(train_file, "w") as f:
        json.dump(data["train"], f, indent=4)
    with open(dev_file, "w") as f:
        json.dump(data["dev"], f, indent=4)
    with open(test_file, "w") as f:
        json.dump(data["test"], f, indent=4)
    print("Dataset Train saved to: ", train_file, "with size: ", len(data["train"]))
    print("Dataset Dev saved to: ", dev_file, "with size: ", len(data["dev"]))
    print("Dataset Test saved to: ", test_file, "with size: ", len(data["test"]))


def load_guidelines(guideline_address):
    guideline = json.load(open(guideline_address))
    new_guidelines = {}
    for key, value in guideline.items():
        key = key[:key.index("(")].strip()
        new_guidelines[key] = value
    return new_guidelines

"""
schema = event_schemas[event_name]# we pass the list of schemas for the event name and extract corresponding to current event type
annotated_schema = schema if not do_annotate_schema else annotate_schema_fun(schema, guidelines, False)# optionally, add guidelines to the schema
# print(f"Annotated Schema:\n{annotated_schema}")
# xxx
prompt = task_sep + "\n\n" + annotated_schema + "\n\n" + text_sep + f"\ntext = \"{text}\"\n\n{footer}"
instruction = task_to_prompts[task_type]["header"]
"""
text_sep = "# This is the text to analyze"
task_sep = "# The following lines describe the task definition"
def get_negative_samples(samples, dataset_name, guidelines, k= 15, do_annotate_schema=False):
    ###
    footer = task_to_prompts[dataset_to_task_mapper[dataset_name]]["footer"]
    instruction = task_to_prompts[dataset_to_task_mapper[dataset_name]]["header"]
    all_schema = load_init_prompts(dataset_name)
    new_samples = []
    for sample in samples:
        current_text, covered_events = sample["text"], set()
        for x in sample["event_mentions"]:
            cur_event = clean_event_name(x["event_type"], dataset_name).split("(")[0].strip()
            covered_events.add(cur_event)
        uncovered_events = set(all_schema.keys()) - covered_events
        # k_dash = k - len(covered_events)
        # print(k, len(all_schema.keys()), len(covered_events))
        uncovered_events = random.sample(list(uncovered_events), k = min(k, len(all_schema.keys())))
        # print(covered_events)
        # print(uncovered_events)

        if(len(covered_events)>0):
            for uc_event in uncovered_events:
                schema = all_schema[uc_event]
                annotated_schema = schema if not do_annotate_schema else annotate_schema_fun(schema, guidelines, False)
                text = sample["text"]
                prompt = task_sep + "\n\n" + annotated_schema + "\n\n" + text_sep + f"\ntext = \"{text}\"\n\n{footer}"
                new_samples.append({"input": prompt, "output": "[]"})
    # print("added -ve samples: ", len(new_samples), do_annotate_schema)
    return new_samples

        
