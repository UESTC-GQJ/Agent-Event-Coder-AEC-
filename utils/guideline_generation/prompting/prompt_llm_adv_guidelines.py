from openai import AzureOpenAI
from glob import glob
from tqdm import tqdm
import argparse
import backoff
import getpass
import base64
import openai
import json
import ast
import os
import re
import time
from openai import OpenAIError

dataset_mapper = {
    "richere-en": {
        "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/richere-en/Sep_master_data_richere-en.json", 
        "def_file": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/def_Sep_master_data_richere-en.json", 
        "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/richere-en/Master_event_dataclasses_richere-en.json", 
        "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/richere-en/richere-en_event_type_files"
    }
}

# ALLOWED_EVENTS_GENEVA = {
#     "A(Event)",
#     "B(Event)",
#     "C(Event)"
# }

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--llm", help="Choice of LLM (GPT-4, GPT-4o, GPT-3.5, etc.)", default="gpt-4o")
parser.add_argument("-i", "--prompt_dir", help = "Directory in which prompts are stored", default = "./../synthesize_guidelines/prompts")
parser.add_argument("-t", "--temperature", default = 0)
parser.add_argument("-m", "--max_tokens", default = 4096)
parser.add_argument("-p", "--top_p", default = 0.7)
parser.add_argument("-f", "--freq_pen", default = 0.0)
parser.add_argument("-n", "--pres_pen", default=0.0)
parser.add_argument("-o", "--out_dir", default="./../synthesize_guidelines/synthesized_guidelines/")
parser.add_argument("-d", "--dataset_name", default="richere-en")

MODEL_MAP = {
    "gpt-4o": {"name": "gpt-4o-2024-05-13", "endpoint": "https://azure-openai-api-eastus2.openai.azure.com/", "api_version": "2024-04-01-preview"}
}


# guidelines = json.load(open("./../synthesize_guidelines/fewevent_event_dataclasses.json"))
@backoff.on_exception(
    backoff.expo,                    # Exponential backoff
    (openai.OpenAIError, IOError),   # Retry on specific exceptions
    max_tries=5                      # Maximum number of retries
)
# def parse_response(response, event_name, guidelines):
#     response = response if(type(response) == type({})) else ast.literal_eval(response)
#     # print(response)
#     event_name = event_name.replace("prompt_", "").strip()
#     return_dict = {event_name:{"description":None}, "attributes":{"mention": guidelines[event_name]["attributes"]["mention"]}}
#     # print("<<<", return_dict)
#     # print(">>>", type(response))
#     return_dict[event_name]["description"] = response["Event Definition"]
#     for key, value in response["Arguments Definitions"].items():
#         return_dict["attributes"][key] = value 
#     # return_dict[event_name]["description"]
#     return return_dict

def parse_response(response, event_name, guidelines):
    """
    Parses the LLM response and formats it into the desired JSON structure.
    """
    # Convert the response string to a dictionary if it's not already one
    response = response if isinstance(response, dict) else ast.literal_eval(response)
    
    # Clean the event name
    event_name = event_name.replace("prompt_", "").strip()
    
    # Initialize the return dictionary with the event name
    return_dict = {
        event_name: {
            "description": response.get("Event Definition", "No Event Definition Provided"),
            "attributes": {}
        }
    }
    
    # Extract "Arguments Definitions" and map them to the attributes
    arguments_definitions = response.get("Arguments Definitions", {})
    for key, value in arguments_definitions.items():
        return_dict[event_name]["attributes"][key] = value
    
    return return_dict


# def get_response(model, prompt, args, client, event_name, guidelines, dataset_name):
#     print("-*" * 50)
#     print(prompt)
#     event_name = os.path.split(event_name)[-1].replace(".txt", "")

#     retries = 3  # Maximum number of retries
#     delay = 20  # Start with a 20-second delay
#     cooldown_period = 60  # Cooldown period in seconds if rate limit is hit
#     batch_size = 5  # Number of requests to process before a cooldown
#     token_limit = 50000  # Example token limit for manual cooldown
#     total_tokens_used = 0  # Token usage tracker

#     # Retry logic
#     for attempt in range(retries):
#         try:
#             # Track token usage
#             prompt_tokens = len(prompt.split())  # Rough token estimate from prompt
#             total_tokens_used += prompt_tokens

#             # Check if we are near the token limit
#             if total_tokens_used > token_limit:
#                 print(f"Token usage approaching limit ({total_tokens_used}/{token_limit}). Applying cooldown...")
#                 time.sleep(cooldown_period)
#                 total_tokens_used = 0  # Reset after cooldown

#             # Send the request to the OpenAI API
#             response = client.chat.completions.create(
#                 model=model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=args.temperature,
#                 max_tokens=args.max_tokens,
#                 top_p=args.top_p,
#                 frequency_penalty=args.freq_pen,
#                 presence_penalty=args.pres_pen
#             )
#             response = response.choices[0].message.content
            
#             # Save logs
#             os.makedirs(f"./logs/{dataset_name}", exist_ok=True)
#             with open(f"./logs/{dataset_name}" + event_name + ".json", "w") as f:
#                 json.dump({"Prompt": prompt, "Response": response}, f, indent = 4)

#             # Extract JSON from the response
#             pattern = re.compile(r'json\n\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}\n', re.DOTALL)
#             matches = pattern.findall(response)
#             matches = [match.strip('json\n').strip('\n') for match in matches]
#             try:
#                 largest_json = max(matches, key=len)
#             except:
#                 largest_json = response.replace("```", "").replace("json", "")

#             print("--" * 50)
#             print(largest_json)
#             print("-*" * 50)

#             # Save parsed response
#             os.makedirs(f"{args.out_dir}", exist_ok=True)
#             with open(f"{args.out_dir}/{event_name}.json", "w") as f:
#                 json.dump(parse_response(largest_json, event_name, guidelines), f, indent=4)

#             # After processing a batch, wait for a cooldown
#             if (attempt + 1) % batch_size == 0:
#                 print(f"Processed {batch_size} requests. Applying batch cooldown...")
#                 time.sleep(cooldown_period)

#             # Add a delay between successful requests
#             time.sleep(delay)
#             return largest_json

#         except OpenAIError as e:
#             print(f"Rate limit exceeded. Retrying in {cooldown_period} seconds... (Attempt {attempt + 1}/{retries})")
#             time.sleep(cooldown_period)  # Wait for the cooldown period before retrying
#             cooldown_period *= 2  # Optionally double the cooldown period for exponential backoff

#     # If all retries fail, raise an exception
#     raise Exception(f"Failed to get response for {event_name} after {retries} retries due to rate limits.")


def get_response(model, prompt, args, client, event_name, guidelines, dataset_name):
    print("-*" * 50)
    print(prompt)
    event_name = os.path.split(event_name)[-1].replace(".txt", "")

    retries = 3  # Maximum number of retries
    delay = 20  # Start with a 20-second delay
    cooldown_period = 60  # Cooldown period in seconds if rate limit is hit
    batch_size = 5  # Number of requests to process before a cooldown
    token_limit = 50000  # Example token limit for manual cooldown
    total_tokens_used = 0  # Token usage tracker

    # Retry logic
    for attempt in range(retries):
        try:
            # Track token usage
            prompt_tokens = len(prompt.split())  # Rough token estimate from prompt
            total_tokens_used += prompt_tokens

            # Check if we are near the token limit
            if total_tokens_used > token_limit:
                print(f"Token usage approaching limit ({total_tokens_used}/{token_limit}). Applying cooldown...")
                time.sleep(cooldown_period)
                total_tokens_used = 0  # Reset after cooldown

            # Send the request to the OpenAI API
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                frequency_penalty=args.freq_pen,
                presence_penalty=args.pres_pen
            )
            response_content = response.choices[0].message.content
            
            # Save logs
            os.makedirs(f"./logs/{dataset_name}", exist_ok=True)
            with open(f"./logs/{dataset_name}/{event_name}.json", "w") as f:
                json.dump({"Prompt": prompt, "Response": response_content}, f, indent=4)

            # Extract JSON from the response
            pattern = re.compile(r'json\n\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}\n', re.DOTALL)
            matches = pattern.findall(response_content)
            matches = [match.strip('json\n').strip('\n') for match in matches]
            try:
                largest_json = max(matches, key=len)
            except:
                largest_json = response_content.replace("```", "").replace("json", "")

            print("--" * 50)
            print(largest_json)
            print("-*" * 50)

            # Save parsed response
            os.makedirs(f"{args.out_dir}", exist_ok=True)
            with open(f"{args.out_dir}/{event_name}.json", "w") as f:
                json.dump(parse_response(largest_json, event_name, guidelines), f, indent=4)

            # After processing a batch, wait for a cooldown
            if (attempt + 1) % batch_size == 0:
                print(f"Processed {batch_size} requests. Applying batch cooldown...")
                time.sleep(cooldown_period)

            # Add a delay between successful requests
            time.sleep(delay)
            return largest_json

        except OpenAIError as e:
            print(f"Rate limit exceeded. Retrying in {cooldown_period} seconds... (Attempt {attempt + 1}/{retries})")
            time.sleep(cooldown_period)  # Wait for the cooldown period before retrying
            cooldown_period *= 2  # Optionally double the cooldown period for exponential backoff

    # If all retries fail, raise an exception
    raise Exception(f"Failed to get response for {event_name} after {retries} retries due to rate limits.")


os.makedirs("./logs", exist_ok=True)
if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = args.dataset_name
    args.out_dir = os.path.join(args.out_dir, dataset_name)
    guidelines = json.load(open(dataset_mapper[dataset_name]["event_arg_ontology"]))
    openai_key = os.environ.get("OPENAI_API_KEY")
    if(openai_key is None):
        os.environ["OPENAI_API_KEY"] = openai_key = getpass.getpass(prompt='Enter your OPENAI_API_KEY: ')
    client = AzureOpenAI(
    api_key=openai_key,
    api_version=MODEL_MAP[args.llm]["api_version"],
    azure_endpoint=MODEL_MAP[args.llm]["endpoint"],
    azure_deployment=MODEL_MAP[args.llm]["name"]
    )
    prompt_files = glob(os.path.join(os.path.join(args.prompt_dir, dataset_name), "*.txt"))
    existing_files = [os.path.split(ff)[-1].replace("prompt_", "").replace(".json", "") for ff in glob(f"/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/synthesized_guidelines/{dataset_name}/*.json")]
    # print(existing_files)
    for prompt_file in tqdm(prompt_files, total = len(prompt_files), desc = "Prompting ..."):
        event_name = os.path.split(prompt_file)[-1].replace("prompt_", "").replace(".txt", "")

        # # Only process allowed events for Geneva
        # if dataset_name == "wikievents" and event_name not in ALLOWED_EVENTS_GENEVA:
        #     print(f"Skipping event {event_name} as it is not in the allowed list for Geneva.")
        #     continue  # Skip events not in the allowed list

        if(event_name in existing_files):
            print(f"Guidelines for event {event_name} already exists.")
            continue
        # xx
        prompt = "".join([lines for lines in open(prompt_file)])
        #model, prompt, args, client, event_name, guidelines, dataset_name):
        get_response(args.llm, prompt, args, client, prompt_file, guidelines, dataset_name)
        # break

        # Add a delay to avoid hitting the rate limit
        time.sleep(2)



    # "ACE": {
    #     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/Sep_master_data_ace05-en.json", 
    #     "def_file": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/def_Sep_master_data_ace05-en.json", 
    #     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/Master_event_dataclasses_ace05-en.json", 
    #     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/ace05-en_event_type_files"
    # },
