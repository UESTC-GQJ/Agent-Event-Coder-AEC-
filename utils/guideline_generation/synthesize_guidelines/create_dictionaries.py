from tabulate import tabulate
from tqdm import tqdm
import argparse
import json
import re
import os

parser = argparse.ArgumentParser()

#choices=["ace05-en", "casie", "richere-en", "fewevent", "geneva", "genia2011", "genia2013", "m2e2", "maven", "mee", "mlee", "muc4", "phee", "rams", "wikievents"]

dataset_mapper = {
	"ACE": {
    "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/Sep_master_data_ace05-en.json", 
    "def_file": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/def_Sep_master_data_ace05-en.json", 
    "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/Master_event_dataclasses_ace05-en.json", 
    "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/ace05-en_event_type_files"
    },

	"richere-en": {
    "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/richere-en/Sep_master_data_richere-en.json", 
    "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_richere-en.json", 
    "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/richere-en/Master_event_dataclasses_richere-en.json", 
    "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/richere-en/richere-en_event_type_files"
    }
}

parser.add_argument("-d", "--dataset_name", help = "Name of the dataset", default = "richere-en")

def update_dict(dict, key):
	if(dict.get(key) is None):
		dict[key] = 0
	dict[key] += 1

def build_event_dictionary(master_file_address):
	return_dict = {}
	event_count_dict = {}
	jsons = [json.loads(line.strip()) for line in open(master_file_address)]
	for jsn in tqdm(jsons, total = len(jsons), desc = "Processing the master file"):
		text = jsn["text"]
		events = jsn["event_mentions"]
		if(len(events)<=0):
			return_dict["None"] = {}
			update_dict(event_count_dict, "None")
			continue
		# print(jsn.keys())
		event_name = list(jsn["event_mentions"].keys())[0]
		update_dict(event_count_dict, event_name)
		parent_type = event_name[event_name.find("(")+1:event_name.find(")")]
		actual_event_name = event_name[:event_name.find("(")]
		actual_event_name = re.sub(r"(\w)([A-Z])", r"\1 \2", actual_event_name)
		parent_type = re.sub(r"(\w)([A-Z])", r"\1 \2", parent_type)
		results = jsn["event_mentions"][event_name]["results"]
		if(return_dict.get(event_name) is None):
			return_dict[event_name] = {"DisplayName": f"Event Type: {actual_event_name}\nParent Event Type: {parent_type}", "Arguments": set()}
		for result in results:
			for r_keys in result:
				# print(">>", r_keys)
				if(r_keys=="mention"):
					continue
				return_dict[event_name]["Arguments"].add(r_keys)
	return return_dict, event_count_dict

def main(args):
	dataset_name = args.dataset_name
	event_dict, event_count_dict = build_event_dictionary(dataset_mapper[dataset_name]["master_file"])
    #For geneva - 4 events
	# event_dict, event_count_dict = build_event_dictionary(dataset_mapper[dataset_name]["master_file"], dataset_name)
	table = [[key, count] for key, count in event_count_dict.items()]
	headers = ['Event Type', 'Count']
	# print(event_dict)
	for event_type in event_dict:
		if(event_type == "None"):
			continue
		event_dict[event_type]["Arguments"] = list(event_dict[event_type]["Arguments"])
	with open("def_" + os.path.split(dataset_mapper[dataset_name]["master_file"])[-1], "w") as f:
		json.dump(event_dict, f, indent = 4)
	print(tabulate(table, headers=headers, tablefmt='grid'))

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)



# Define the allowed events for the Geneva dataset
# ALLOWED_EVENTS_GENEVA = {
#     "A(Event)",
#     "B(Event)",
#     "C(Event)"
#}

# #FOR GENEVA - 4 EVENTS
# def build_event_dictionary(master_file_address, dataset_name):
#     return_dict = {}
#     event_count_dict = {}
#     jsons = [json.loads(line.strip()) for line in open(master_file_address)]
    
#     # Check if it's the Geneva dataset and apply event filtering
#     is_geneva = dataset_name == "wikievents"
    
#     for jsn in tqdm(jsons, total=len(jsons), desc="Processing the master file"):
#         text = jsn["text"]
#         events = jsn["event_mentions"]
        
#         if len(events) <= 0:
#             return_dict["None"] = {}
#             update_dict(event_count_dict, "None")
#             continue
        
#         event_name = list(jsn["event_mentions"].keys())[0]
        
#         # Skip events not in the allowed list if the dataset is Geneva
#         if is_geneva and event_name not in ALLOWED_EVENTS_GENEVA:
#             continue
        
#         update_dict(event_count_dict, event_name)
#         parent_type = event_name[event_name.find("(")+1:event_name.find(")")]
#         actual_event_name = event_name[:event_name.find("(")]
#         actual_event_name = re.sub(r"(\w)([A-Z])", r"\1 \2", actual_event_name)
#         parent_type = re.sub(r"(\w)([A-Z])", r"\1 \2", parent_type)
#         results = jsn["event_mentions"][event_name]["results"]
        
#         if return_dict.get(event_name) is None:
#             return_dict[event_name] = {
#                 "DisplayName": f"Event Type: {actual_event_name}\nParent Event Type: {parent_type}", 
#                 "Arguments": set()
#             }
        
#         for result in results:
#             for r_keys in result:
#                 if r_keys == "mention":
#                     continue
#                 return_dict[event_name]["Arguments"].add(r_keys)
    
#     return return_dict, event_count_dict


# "casie": {
#     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/casie/Sep_master_data_casie.json", 
#     "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_casie.json", 
#     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/casie/Master_event_dataclasses_casie.json", 
#     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/casie/casie_event_type_files"
#     },

# 	"fewevent": {
#     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/fewevent/Sep_master_data_fewevent.json", 
#     "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_fewevent.json", 
#     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/fewevent/Master_event_dataclasses_fewevent.json", 
#     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/fewevent/fewevent_event_type_files"
#     },

# 	"geneva": {
#     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/geneva/Sep_master_data_geneva.json", 
#     "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_geneva.json", 
#     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/geneva/Master_event_dataclasses_geneva.json", 
#     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/geneva/geneva_event_type_files"
#     },

# 	"genia2011": {
#     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/genia2011/Sep_master_data_genia2011.json", 
#     "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_genia2011.json", 
#     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/genia2011/Master_event_dataclasses_genia2011.json", 
#     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/genia2011/genia2011_event_type_files"
#     },

# 	"genia2013": {
#     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/genia2013/Sep_master_data_genia2013.json", 
#     "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_genia2013.json", 
#     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/genia2013/Master_event_dataclasses_genia2013.json", 
#     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/genia2013/genia2013_event_type_files"
#     },

# 	"m2e2": {
#     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/m2e2/Sep_master_data_m2e2.json", 
#     "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_m2e2.json", 
#     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/m2e2/Master_event_dataclasses_m2e2.json", 
#     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/m2e2/m2e2_event_type_files"
#     },

# 	"maven": {
#     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/maven/Sep_master_data_maven.json", 
#     "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_maven.json", 
#     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/maven/Master_event_dataclasses_maven.json", 
#     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/maven/maven_event_type_files"
#     },

# 	"mee": {
#     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/mee/Sep_master_data_mee.json", 
#     "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_mee.json", 
#     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/mee/Master_event_dataclasses_mee.json", 
#     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/mee/mee_event_type_files"
#     },

# 	"mlee": {
#     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/mlee/Sep_master_data_mlee.json", 
#     "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_mlee.json", 
#     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/mlee/Master_event_dataclasses_mlee.json", 
#     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/mlee/mlee_event_type_files"
#     },

# 	"muc4": {
#     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/muc4/Sep_master_data_muc4.json", 
#     "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_muc4.json", 
#     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/muc4/Master_event_dataclasses_muc4.json", 
#     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/muc4/muc4_event_type_files"
#     },

# 	"phee": {
#     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/phee/Sep_master_data_phee.json", 
#     "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_phee.json", 
#     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/phee/Master_event_dataclasses_phee.json", 
#     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/phee/phee_event_type_files"
#     },

# 	"rams": {
#     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/rams/Sep_master_data_rams.json", 
#     "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_rams.json", 
#     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/rams/Master_event_dataclasses_rams.json", 
#     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/rams/rams_event_type_files"
#     },
	
# 	"wikievents": {
#     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/wikievents/Sep_master_data_wikievents.json", 
#     "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_wikievents.json", 
#     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/wikievents/Master_event_dataclasses_wikievents.json", 
#     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/wikievents/wikievents_event_type_files"
#     },
#     "speed": {
#     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/speed/Sep_master_data_speed.json", 
#     "def_file": "/scratch/spati/tmp/LLaMA-Events/synthesize_guidelines/def_Sep_master_data_speed.json", 
#     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/speed/Master_event_dataclasses_speed.json", 
#     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/speed/speed_event_type_files"
#     }