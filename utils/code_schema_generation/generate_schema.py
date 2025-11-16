import argparse
import os
import sys
from glob import glob
from tqdm import tqdm
import json

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


parser = argparse.ArgumentParser(description="A simple script to demonstrate argument parsing.")
parser.add_argument(
    "-f", "--dataset_folder",
    type=str,
    help="Path to the floder containing the dataset.",
    required=True,
    default="./../../TextEE/processed_data/"
)

def clean_schema(schema, dataset_name):
    splitter = schema_splitters[dataset_name]
    cleaned_schema = {}
    # create mapper for new and old event types
    event_mapper = {}
    for event_type, attributes in schema.items():
        event_type = event_type.replace("n/a", "Na")
        event_type_parts = event_type.split(splitter)
        if len(event_type_parts) > 1:
            parent_event_name = event_type_parts[0].replace("-", "").replace(".", "_")
            if(dataset_name=="casie"):
                parent_event_name = parent_event_name.title()
            event_name = "_".join(event_type_parts[1:]).replace("-", "").replace(".", "_")
            if(dataset_name=="rams"):
                new_event_type = f"{parent_event_name}_{event_name}(Event)"
            else:
                parent_event_name = parent_event_name + "Event"
                new_event_type = f"{event_name}({parent_event_name})"
            cleaned_schema[new_event_type] = attributes
            # add the event type to the mapper
            if event_type not in event_mapper:
                event_mapper[event_type] = new_event_type
        else:
            original_event_type = event_type + ""
            event_type = event_type.replace("-", "").replace(".", "_") + "(Event)"
            if(dataset_name=="speed"):
                event_type = event_type.title()
            cleaned_schema[event_type] = attributes
            # add the event type to the mapper
            if event_type not in event_mapper:
                event_mapper[original_event_type] = event_type

    return cleaned_schema, event_mapper

# def create_init_prompt(dataset, schema):
    

def process_dataset(dataset):
    cur_schema = {}
    for jsn in dataset:
        event_mentions = jsn["event_mentions"]
        for event in event_mentions:
            event_type = event["event_type"]
            if event_type not in cur_schema:
                cur_schema[event_type] = set()
            # get the attributes
            arguments = [x.replace("-", "_") for x in event["arguments"]]
            for argument in arguments:
                role = argument["role"].lower().replace("-", "_")
                cur_schema[event_type].add(role)
            cur_schema[event_type].add("mention")
    #convert the set to a list
    for event_type in cur_schema:
        cur_schema[event_type] = list(cur_schema[event_type])
    return cur_schema



def generate_python_definitions(schema_address, dataset_name=None):
    if(dataset_name is None):
        ddf = "all_ee"
    else:
        ddf = dataset_name
    header="""from dataclasses import dataclass
from typing import List, Optional
from utils_typing import Entity, Event, Relation, dataclass


"""
    dump_string = ""
    parent_event_classes = set()
    for event_type, attributes in schema_address.items():
        #first we need to get the exact name of the event type and the parent event type. The format is EventType(ParentEventType)
        event_parts = event_type.split("(")
        event_name = event_parts[0]
        parent_event_name = event_parts[1].replace(")", "")
        parent_event_classes.add(parent_event_name)
        dump_string += f"@dataclass\n"
        dump_string += f"class {event_type}:\n"
        dump_string += f"    def __init__(self, mention: Optional[str] = None, "
        for attribute in attributes:
            if(attribute == "mention"):
                continue
            dump_string += f"{attribute}: Optional[List] = None, "
        dump_string = dump_string[:-2] + "):\n"
        dump_string += f"        self.mention = mention if mention is not None else []\n"
        for attribute in attributes:
            if(attribute == "mention"):
                continue
            dump_string += f"        self.{attribute} = {attribute} if {attribute} is not None else []\n"
        dump_string += f"    def __repr__(self):\n"
        dump_string += f"        return f\"{event_name}(mention='{{self.mention}}', " 
        for attribute in attributes:
            if(attribute == "mention"):
                continue
            dump_string += f"{attribute}={{self.{attribute}}}, "
        dump_string = dump_string[:-2] + ")\"\n"
        dump_string += f"\n"
    # now we need to create the parent event classes
    #append parent event classes to the beginning of the dump string
    for parent_event_class in parent_event_classes:
        dump_string = f"@dataclass\nclass {parent_event_class}:\n    pass\n\n" + dump_string
    # now we need to create the parent event classes
    dump_string = header + dump_string
    # now we need to create the parent event classes
    with open(f"./python_event_defs/{ddf}_definitions_new.py", "w") as f:
        f.write(dump_string)

        
def create_init_prompt(dataset, schema):
    """
    Saves dictionary in the format:
    for each event type, create a class with the name of the event type and the attributes as the class variables.
    @dataclass
    class Spread(Event):
        mention: str
        information_source: List
        population: List
        disease: List
        value: List
        trend: List
        place: List
        time: List
    """
    dump_string = ""
    for event_type, attributes in schema.items():
        dump_string += f"@dataclass\n"
        dump_string += f"class {event_type}:\n"
        # make sure always starts with mention
        dump_string += f"    mention: str\n"
        # remove mention from the attributes
        attributes = [attribute for attribute in attributes if attribute != "mention"]
        for attribute in attributes:
            dump_string += f"    {attribute}: List\n"
        dump_string += f"\n"
    os.makedirs(os.path.join(".", "init_prompts"), exist_ok=True)
    with open(os.path.join(".", "init_prompts", dataset + ".txt"), "w") as f:
        f.write(dump_string)

    
def main():
    args = parser.parse_args()
    dataset_folder = args.dataset_folder
    full_schema, full_mapper = {}, {}
    if not os.path.exists(dataset_folder):
        print(f"Error: The dataset folder '{dataset_folder}' does not exist.")
        sys.exit(1)
    # mapper dictionary: contains per dataset event name mapper. 
    directories = glob(os.path.join(dataset_folder, "*"))
    for directory in tqdm(directories, desc="Processing directories", unit="directory"):
        dataset_name = os.path.basename(directory)
        dataset_mapper = {}
        dataset_full_schema = {}
        if not os.path.isdir(directory):
            print(f"Error: The path '{directory}' is not a directory.")
            sys.exit(1)
        print(f"Processing directory: {directory}")
        split_dirs = glob(os.path.join(directory, "split*"))
        for split_dir in split_dirs:
            if not os.path.isdir(split_dir):
                print(f"Error: The path '{split_dir}' is not a directory.")
                sys.exit(1)
            files = glob(os.path.join(split_dir, "*.json"))
            for file in files:
                with open(file, "r") as f:
                    data = [json.loads(x) for x in f.readlines()]
                    dataset_schema = process_dataset(data)
                    #####
                    dataset_schema, event_mapper = clean_schema(dataset_schema, dataset_name=os.path.basename(directory))
                    # update the dataset mapper
                    for event_type, new_event_type in event_mapper.items():
                        if event_type not in dataset_mapper:
                            dataset_mapper[event_type] = new_event_type
                    for event_type in dataset_schema:
                        if event_type not in full_schema:
                            full_schema[event_type] = set()
                        attributes = dataset_schema[event_type]
                        for attribute in attributes:
                            full_schema[event_type].add(attribute)
                    # add the dataset schema to dataset_full_schema
                    for event_type in dataset_schema:
                        if event_type not in dataset_full_schema:
                            dataset_full_schema[event_type] = set()
                        attributes = dataset_schema[event_type]
                        for attribute in attributes:
                            dataset_full_schema[event_type].add(attribute)
        full_mapper[dataset_name] = dataset_mapper
        # convert the set to a list
        for event_type in dataset_full_schema:
            dataset_full_schema[event_type] = list(dataset_full_schema[event_type])
        create_init_prompt(dataset_name, dataset_full_schema)
        generate_python_definitions(dataset_full_schema, dataset_name)
    for event_type in full_schema:
        full_schema[event_type] = list(full_schema[event_type])
    with open(os.path.join(".", "schema.json"), "w") as f:
        json.dump(full_schema, f, indent=4)
    with open(os.path.join(".", "mapper.json"), "w") as f:
        json.dump(full_mapper, f, indent=4)
    print("Schema generated successfully.")
    generate_python_definitions(full_schema)
             


if __name__ == "__main__":
    main()