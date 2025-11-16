correct_code_schema = "/Volumes/Academic/Spring\'25/Prompt-Optimization-ACL-25/src/tasks/all_ee_definitions.py"
"""
correct_code_schema contains class defintions like shown below:
@dataclass
class transaction_transfermoney_giftgrantprovideaid(Event):
    def __init__(self, mention: Optional[List] = None, beneficiary: Optional[List] = None, giver: Optional[List] = None, money: Optional[List] = None, place: Optional[List] = None, recipient: Optional[List] = None):
        self.mention = mention if mention is not None else []
        self.beneficiary = beneficiary if beneficiary is not None else []
        self.giver = giver if giver is not None else []
        self.money = money if money is not None else []
        self.place = place if place is not None else []
        self.recipient = recipient if recipient is not None else []

    def __repr__(self):
        return f"transaction_transfermoney_giftgrantprovideaid(mention='{self.mention}', beneficiary={self.beneficiary}, giver={self.giver}, money={self.money}, place={self.place}, recipient={self.recipient})"

We need to only extract the class names and verify if they are in our nrewly generated schema.
"""

# first read correct_code_schema and extract the class names
import re
import os
import sys
import json
from tqdm import tqdm
from glob import glob


def extract_class_names_from_code_schema(code_schema_path):
    class_names = []
    with open(code_schema_path, 'r') as file:
        content = file.read()
    for line in content.splitlines():
        # Check if the line contains a class definition
        if line.strip().startswith('class '):
            class_name = line.split('class ')[-1].replace(":", "").strip()
            class_names.append(class_name)
            # print(class_name)
            # xxx
    return class_names

event_names = extract_class_names_from_code_schema(correct_code_schema)

print(event_names)
print(len(event_names))
# now let's read the newly generated schema and extract the class names
jsn = json.load(open("schema.json", "r"))
generatde_schema = jsn.keys()
print(generatde_schema)
print(len(generatde_schema))
# now let's compare the two lists
# check if the class names in event_names are in the generated schema
missing_classes = []
for class_name in event_names:
    if class_name not in generatde_schema:
        missing_classes.append(class_name)

print("Missing classes:")
print((missing_classes))