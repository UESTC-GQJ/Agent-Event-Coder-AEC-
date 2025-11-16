import numpy as np
from collections import defaultdict
from prettytable import PrettyTable
from tqdm import tqdm
from all_ee_definitions import *

import argparse
import json
from dataclasses import fields
import os, re
import difflib
import random
import inspect

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="./../demo/e2e_demo.json")

def compute_f1(pred_num, gold_num, match_num):
    precision = match_num / pred_num if pred_num > 0 else 0
    recall = match_num / gold_num if gold_num > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

def extract_objects(class_event):
    ob = {key: val for key, val in class_event.__dict__.items() if (key!="mention" and (not(key.startswith("__") and key.endswith("__"))))}
    return ob

def log_hallucinations_and_mismatches(pred_events, input_text):
    hallucination_dict = {"trg_hallucination": [], "arg_hallucination": []}
    try:
        for pred_event in pred_events:
            if pred_event:
                if pred_event.mention not in input_text:
                    hallucination_dict["trg_hallucination"].append({"trigger_span": pred_event.mention, "event_type": pred_event.__class__.__name__, "text": input_text})
                pred_objects = extract_objects(pred_event)
                for arg_role, arg_span in pred_objects.items():
                    if(arg_role!="mention" and isinstance(arg_span, list)):
                        for span in arg_span:
                            if span not in input_text:
                                hallucination_dict["arg_hallucination"].append({"arg_span": span, "arg_role": arg_role, "event_type": pred_event.__class__.__name__, "text": input_text})
    except:
        pass
    return hallucination_dict

def our_pretty_table(trigger_id_f1_scores=None, trigger_cls_f1_scores=None, arg_id_f1_scores=None, arg_cls_f1_scores=None, task_type="E2E"):
    table = PrettyTable()
    if task_type == "E2E":
        table.field_names = ["Metric", "Trigger Identification", "Trigger Classification", "Argument Identification", "Argument Classification"]
    elif task_type == "ED":
        table.field_names = ["Metric", "Trigger Identification", "Trigger Classification"]
    elif task_type == "EAE":
        table.field_names = ["Metric", "Argument Identification", "Argument Classification"]
    else:
        raise ValueError("Invalid task_type. Choose from 'E2E', 'ED', or 'EAE'.")
    if task_type == "E2E":
        table.add_row([
            "Micro - Precision", 
            f"{trigger_id_f1_scores['precision']*100:.2f}", 
            f"{trigger_cls_f1_scores['precision']*100:.2f}", 
            f"{arg_id_f1_scores['precision']*100:.2f}", 
            f"{arg_cls_f1_scores['precision']*100:.2f}"
        ])
        table.add_row([
            "Micro - Recall", 
            f"{trigger_id_f1_scores['recall']*100:.2f}", 
            f"{trigger_cls_f1_scores['recall']*100:.2f}", 
            f"{arg_id_f1_scores['recall']*100:.2f}", 
            f"{arg_cls_f1_scores['recall']*100:.2f}"
        ])
        table.add_row([
            "Micro - F1-Score", 
            f"{trigger_id_f1_scores['f1']*100:.2f}", 
            f"{trigger_cls_f1_scores['f1']*100:.2f}", 
            f"{arg_id_f1_scores['f1']*100:.2f}", 
            f"{arg_cls_f1_scores['f1']*100:.2f}"
        ])
    elif task_type == "ED":
        table.add_row([
            "Micro - Precision", 
            f"{trigger_id_f1_scores['precision']*100:.2f}", 
            f"{trigger_cls_f1_scores['precision']*100:.2f}"
        ])
        table.add_row([
            "Micro - Recall", 
            f"{trigger_id_f1_scores['recall']*100:.2f}", 
            f"{trigger_cls_f1_scores['recall']*100:.2f}"
        ])
        table.add_row([
            "Micro - F1-Score", 
            f"{trigger_id_f1_scores['f1']*100:.2f}", 
            f"{trigger_cls_f1_scores['f1']*100:.2f}"
        ])
    elif task_type == "EAE":
        table.add_row([
            "Micro - Precision", 
            f"{arg_id_f1_scores['precision']*100:.2f}", 
            f"{arg_cls_f1_scores['precision']*100:.2f}"
        ])
        table.add_row([
            "Micro - Recall", 
            f"{arg_id_f1_scores['recall']*100:.2f}", 
            f"{arg_cls_f1_scores['recall']*100:.2f}"
        ])
        table.add_row([
            "Micro - F1-Score", 
            f"{arg_id_f1_scores['f1']*100:.2f}", 
            f"{arg_cls_f1_scores['f1']*100:.2f}"
        ])
    return table

# used to extract arguments which are not generated 
def extract_missing_args(exception_msg):
    multiple_missing = re.findall(r"missing (\d+) required positional arguments: (.+)", exception_msg)
    if multiple_missing:
        _, args = multiple_missing[0]
        return [arg.strip("'") for arg in args.split(", ")]
    single_missing = re.findall(r"missing 1 required positional argument: '(\w+)'", exception_msg)
    if single_missing:
        return [single_missing[0]]
    return []

# the arguments hallucinated or generated which are not in schema
def extract_hallucinated_args(exception_msg):
    hallucinated_arguments = re.findall(r"got an unexpected keyword argument '(\w+)'", exception_msg)
    return hallucinated_arguments

def extract_undefined_name(exception_msg):
    undefined_name = re.findall(r"name '(\w+)' is not defined", exception_msg)
    return undefined_name[0] if undefined_name else None

# 
def safe_eval(eval_str: str, input_str: str = None):
    eval_str = eval_str.strip()
    result, missing_args, hallucinated_args, hallucinated_classes = [], [], [], []
    try:
        temp_result = eval(eval_str)
        for tr in temp_result:#need to do this to handle cases like [Arrest] instead of [Arrest(...)]
            if isinstance(tr, type):
                continue
            result.append(tr)
    except TypeError as e:
        exception_msg = str(e); 
        missing_args = extract_missing_args(exception_msg)
        hallucinated_args = extract_hallucinated_args(exception_msg)
    except NameError as e:
        exception_msg = str(e)
        print(exception_msg, input_str)
        undefined_name = extract_undefined_name(exception_msg)
        if undefined_name:
            hallucinated_classes.append(undefined_name)
        else:
            hallucinated_classes.append(exception_msg)
    except Exception as e:
        pass; 
    result = result if all(type(item).__module__ != 'builtins' for item in result) else []
    return {
        "results": result,
        "mismatched_args": {
            "missing_args": missing_args,
            "hallucinated_args": hallucinated_args,
            "hallucinated_classes": hallucinated_classes
        }
    }


def micro_ed_scores(json_data):
    trigger_id_pred_num, trigger_id_gold_num, trigger_id_match_num = 0, 0, 0
    trigger_cls_match_num = 0 # pred num, gold_num for both id and cls are same so only above 2 vars can be reused
    hallucination_list = []
    for j in tqdm(json_data, total = len(json_data), desc = "Evaluating Micro ED scores"):
        label = j["Label"]
        hals = []
        prediction = j["Prediction"].split("assistant")[-1].strip()
        input_text = j["Input"].split("\n\n#")[3].replace(" This is the text to analyze\ntext = ", "").strip()
        gold_eval = safe_eval(label, "gold")
        pred_eval = safe_eval(prediction, j["Prediction"])
        #######<<< hallucination and makes sure our gold is fine>>>#######
        gold_eval, label_arg_mismatch = gold_eval["results"], gold_eval["mismatched_args"]
        try:
            assert sum(len(x) for x in label_arg_mismatch.values()) <= 0# for labels or golds this should never happen
        except:
            print(label_arg_mismatch, ">>>>\nGE", label, "<<<<<\nPE", prediction)
        pred_eval, pred_arg_mismatch = pred_eval["results"], pred_eval["mismatched_args"]
        if (sum(len(x) for x in pred_arg_mismatch.values())>=1):
            hals.append({"pred": prediction, "hallucinations": pred_arg_mismatch})
        hal_dict_pred, hal_dict_gold = log_hallucinations_and_mismatches(pred_eval, input_text), []#log_hallucinations_and_mismatches(gold_eval, input_text)
        # assert sum(len(x) for x in hal_dict_gold.values()) <= 0 #this may not work because of tokenizatio
        if sum([len(val) for key, val in hal_dict_pred.items()])>0:
            hals.append(hal_dict_pred)
        #######   <<<----  log hallucination and makes sure our gold is fine  ---->>>   #######
        trigger_id_pred_num += len(pred_eval) if pred_eval else 0 # count number of predicted events# used for f1 score calc
        trigger_id_gold_num += len(gold_eval) if gold_eval else 0 # count number of gold events# used for f1 score calc
        # Triggers
        pred_trigger_id_data = set([x.mention for x in pred_eval])
        gold_trigger_id_data = set([x.mention for x in gold_eval])
        pred_trigger_cls_data = [(x.mention, x.__class__.__name__) for x in pred_eval]
        gold_trigger_cls_data = [(x.mention, x.__class__.__name__) for x in gold_eval]
        trigger_id_match_num += len(set(pred_trigger_id_data) & set(gold_trigger_id_data))# used for f1 score calc
        trigger_cls_match_num += len(set(pred_trigger_cls_data)  & set(gold_trigger_cls_data))# used for f1 score calc
        trigger_id_f1_scores = compute_f1(trigger_id_pred_num, trigger_id_gold_num, trigger_id_match_num)
    trigger_id_f1_scores = compute_f1(trigger_id_pred_num, trigger_id_gold_num, trigger_id_match_num)
    trigger_cls_f1_scores = compute_f1(trigger_id_pred_num, trigger_id_gold_num, trigger_cls_match_num)
    table = our_pretty_table(trigger_id_f1_scores, trigger_cls_f1_scores, task_type = "ED")
    print(table)
    return {
        "trigger_id_precision": trigger_id_f1_scores["precision"],
        "trigger_id_recall": trigger_id_f1_scores["recall"],
        "trigger_id_f1": trigger_id_f1_scores["f1"],
        "trigger_cls_precision": trigger_cls_f1_scores["precision"],
        "trigger_cls_recall": trigger_cls_f1_scores["recall"],
        "trigger_cls_f1": trigger_cls_f1_scores["f1"],
        "hallucinations": hallucination_list
    }
        

def micro_eae_scores(json_data):
    arg_id_pred_num, arg_id_gold_num, arg_id_match_num = 0, 0, 0
    arg_cls_match_num = 0 # pred num, gold_num for both id and cls are same so only above 2 vars can be reused
    hallucination_list = []
    for j in tqdm(json_data, total = len(json_data), desc = f"Evaluating Micro EAE scores"):
        label = j["Label"]
        hals = []
        prediction = j["Prediction"].split("assistant")[-1].strip()
        input_text = j["Input"]#.split("\n\n#")[3].replace(" This is the text to analyze\ntext = ", "").strip()
        ###
        gold_eval = safe_eval(label, "gold")
        pred_eval = safe_eval(prediction, j["Prediction"])
        #######<<< hallucination and makes sure our gold is fine>>>#######
        gold_eval, label_arg_mismatch = gold_eval["results"], gold_eval["mismatched_args"]
        assert sum(len(x) for x in label_arg_mismatch.values()) <= 0# for labels or golds this should never happen
        pred_eval, pred_arg_mismatch = pred_eval["results"], pred_eval["mismatched_args"]
        if (sum(len(x) for x in pred_arg_mismatch.values())>=1):
            hals.append({"pred": prediction, "hallucinations": pred_arg_mismatch})
        hal_dict_pred, hal_dict_gold = log_hallucinations_and_mismatches(pred_eval, input_text), []#log_hallucinations_and_mismatches(gold_eval, input_text)
        # assert sum(len(x) for x in hal_dict_gold.values()) <= 0
        if sum([len(val) for key, val in hal_dict_pred.items()])>0:
            hals.append(hal_dict_pred)
        #######   <<<----  log hallucination and makes sure our gold is fine  ---->>>   #######
        pred_trigger_cls_data = [(x.mention, x.__class__.__name__) for x in pred_eval]
        gold_trigger_cls_data = [(x.mention, x.__class__.__name__) for x in gold_eval]
        ## Args
        pred_objects = [extract_objects(x) for x in pred_eval]
        gold_objects = [extract_objects(x) for x in gold_eval]
        pred_arg_id_data, gold_arg_id_data = [], []
        pred_arg_cls_data, gold_arg_cls_data = [], []
        for pred_object, pred_trg_cls_data in zip(pred_objects, pred_trigger_cls_data):#safe as they have the same length
            for pred_role, pred_spans in pred_object.items():
                for pred_span in pred_spans:
                    pred_arg_id_data.append(pred_trg_cls_data + (pred_span,))## for args id
                    pred_arg_cls_data.append(pred_trg_cls_data + (pred_span, pred_role)) ## for args cls
        ##
        for gold_object, gold_trg_cls_data in zip(gold_objects, gold_trigger_cls_data):#safe as they have the same length
            for gold_role, gold_spans in gold_object.items():
                for gold_span in gold_spans:
                    gold_arg_id_data.append(gold_trg_cls_data + (gold_span,))## for args id
                    gold_arg_cls_data.append(gold_trg_cls_data + (gold_span, gold_role))## for args cls
        ## Args
        ### calculations🕺 
        arg_id_pred_num += len(pred_arg_id_data)
        arg_id_gold_num += len(gold_arg_id_data)
        arg_id_match_num += len(set(pred_arg_id_data) & set(gold_arg_id_data))
        arg_cls_match_num += len(set(pred_arg_cls_data) & set(gold_arg_cls_data))
        ### calculations🕺   
        if(len(hals)>0):
            hallucination_list.append(hals) 
    #### compute Precision, Recall, and F1 for Trigger Identification
    arg_id_f1_scores = compute_f1(arg_id_pred_num, arg_id_gold_num, arg_id_match_num)
    arg_cls_f1_scores = compute_f1(arg_id_pred_num, arg_id_gold_num, arg_cls_match_num)
    # print(arg_id_f1_scores, arg_cls_f1_scores)
    table = our_pretty_table(arg_id_f1_scores = arg_id_f1_scores, arg_cls_f1_scores = arg_cls_f1_scores, task_type = "EAE")
    print(table)
    return {
        "arg_id_precision": arg_id_f1_scores["precision"],
        "arg_id_recall": arg_id_f1_scores["recall"],
        "arg_id_f1": arg_id_f1_scores["f1"],
        "arg_cls_precision": arg_cls_f1_scores["precision"],
        "arg_cls_recall": arg_cls_f1_scores["recall"],
        "arg_cls_f1": arg_cls_f1_scores["f1"],
        "hallucinations": hallucination_list
    }
##### --------------------------------------------- Evaluation Script and Examples for MicroE2E scores ##### ---------------------------------------------## 
# Examples (when pred_events<gold_events)
"""
gold_labels = [
    Die(\n    mention=\"killed\",\n    victim=[\"people\"], \n    agent=[\"bomber\"], \n    place=[\"town\"],\n    instrument=[],\n    person=[]\n),
    Die(\n    mention=\"suicide\",\n   victim=[\"himself\"], \n   agent=[\"bomber\"], \n    place=[\"town\"],\n    instrument=[],\n    person=[]\n)
]
pred_labels = [
    Die(\n    mention=\"killed\",\n    victim=[\"people\"], \n    agent=[\"bomber\"], \n    place=[\"Haifa\"],\n    instrument=[],\n    person=[]\n)
]
"""

# Examples (when pred_events>gold_events)
"""
gold_labels = [
    Die(\n    mention=\"killed\",\n    victim=[\"people\"], \n    agent=[\"bomber\"], \n    place=[\"town\"],\n    instrument=[],\n    person=[]\n),
]
pred_labels = [
    Die(\n    mention=\"killed\",\n    victim=[\"people\"], \n    agent=[\"bomber\"], \n    place=[\"Haifa\"],\n    instrument=[],\n    person=[]\n),
    Die(\n    mention=\"suicide\",\n   victim=[\"himself\"], \n   agent=[\"bomber\"], \n    place=[\"town\"],\n    instrument=[],\n    person=[]\n)
]
"""

# Examples (when pred_events=gold_events)
"""
gold_labels = [
    Die(\n    mention=\"killed\",\n    victim=[\"people\"], \n    agent=[\"bomber\"], \n    place=[\"town\"],\n    instrument=[],\n    person=[]\n),
]
pred_labels = [
    Die(\n    mention=\"suicide\",\n   victim=[\"himself\"], \n   agent=[\"bomber\"], \n    place=[\"town\"],\n    instrument=[],\n    person=[]\n)
]
"""
def micro_e2e_scores(json_data):
    """
    ToDoS: 
    1- Log which args are not generated (when missing args exception is raised) : done
    2- Log which args are hallucinated (when unexpected var exception is raised) : done
    3- have asserts for checking gold triggers and args ---> they should pass ALWAYS : done
    4- Any safe and better way of extracting input_text? : 
    5- hallucination_list : done
    6- PrettyTable : done
    """
    # print(len(json_data))
    trigger_id_pred_num, trigger_id_gold_num, trigger_id_match_num = 0, 0, 0
    trigger_cls_match_num = 0 # pred num, gold_num for both id and cls are same so only above 2 vars can be reused
    arg_id_pred_num, arg_id_gold_num, arg_id_match_num = 0, 0, 0
    arg_cls_match_num = 0 # pred num, gold_num for both id and cls are same so only above 2 vars can be reused
    hallucination_list = []
    for j in tqdm(json_data, total = len(json_data), desc = "Evaluating Micro E2E scores"):
        label = j["Label"]
        hals = []
        prediction = j["Prediction"].split("assistant")[-1].strip()
        input_text = j["Input"]
        ###
        gold_eval = safe_eval(label, "gold")
        pred_eval = safe_eval(prediction, j["Prediction"])
        #######<<< hallucination and makes sure our gold is fine>>>#######
        gold_eval, label_arg_mismatch = gold_eval["results"], gold_eval["mismatched_args"]
        pred_eval, pred_arg_mismatch = pred_eval["results"], pred_eval["mismatched_args"]
        if (sum(len(x) for x in pred_arg_mismatch.values())>=1):
            hals.append({"pred": prediction, "hallucinations": pred_arg_mismatch})
        hal_dict_pred, hal_dict_gold = log_hallucinations_and_mismatches(pred_eval, input_text), []#log_hallucinations_and_mismatches(gold_eval, input_text)
        if sum([len(val) for key, val in hal_dict_pred.items()])>0:
            hals.append(hal_dict_pred)
        #######   <<<----  log hallucination and makes sure our gold is fine  ---->>>   #######
        trigger_id_pred_num += len(pred_eval) if pred_eval else 0 # count number of predicted events# used for f1 score calc
        trigger_id_gold_num += len(gold_eval) if gold_eval else 0 # count number of gold events# used for f1 score calc
        # Triggers
        pred_trigger_id_data = []
        for x in pred_eval:
            try:
                pred_trigger_id_data.append(x.mention)
            except:
                pred_trigger_id_data.append("")
        gold_trigger_id_data = set([x.mention for x in gold_eval])
        #
        #----------------------------------------------------------- VERIFY IF THIS IS CORRECT -----------------------------------------------------------#
        # trigger_id_pred_num += len(pred_trigger_id_data) if pred_eval else 0 # count number of predicted events# used for f1 score calc
        # trigger_id_gold_num += len(gold_trigger_id_data) if gold_eval else 0 # count number of gold events# used for f1 score calc
        #----------------------------------------------------------- VERIFY IF THIS IS CORRECT -----------------------------------------------------------#
        pred_trigger_cls_data = []
        for x in pred_eval:
            try:
                pred_trigger_cls_data.append((x.mention, x.__class__.__name__))
            except:
                pred_trigger_cls_data.append(("", ""))
        gold_trigger_cls_data = [(x.mention, x.__class__.__name__) for x in gold_eval]
        #
        try:
            trigger_id_match_num += len(set(pred_trigger_id_data) & set(gold_trigger_id_data))# used for f1 score calc
        except:
            trigger_id_match_num += 0
        try:
            trigger_cls_match_num += len(set(pred_trigger_cls_data)  & set(gold_trigger_cls_data))# used for f1 score calc
        except:
            trigger_cls_match_num += 0
        ## Args
        pred_objects = [extract_objects(x) for x in pred_eval]
        gold_objects = [extract_objects(x) for x in gold_eval]
        assert len(pred_objects) == len(pred_trigger_cls_data)
        assert len(gold_objects) == len(gold_trigger_cls_data)
        
        pred_arg_id_data, gold_arg_id_data = [], []
        pred_arg_cls_data, gold_arg_cls_data = [], []
        for pred_object, pred_trg_cls_data in zip(pred_objects, pred_trigger_cls_data):#safe as they have the same length
            for pred_role, pred_spans in pred_object.items():
                if(type(pred_spans)!=type([])):
                    continue
                for pred_span in pred_spans:
                    pred_arg_id_data.append(pred_trg_cls_data + (pred_span,))## for args id
                    pred_arg_cls_data.append(pred_trg_cls_data + (pred_span, pred_role)) ## for args cls
        ##
        for gold_object, gold_trg_cls_data in zip(gold_objects, gold_trigger_cls_data):#safe as they have the same length
            for gold_role, gold_spans in gold_object.items():
                for gold_span in gold_spans:
                    gold_arg_id_data.append(gold_trg_cls_data + (gold_span,))## for args id
                    gold_arg_cls_data.append(gold_trg_cls_data + (gold_span, gold_role))## for args cls
        ## Args
        ### calculations🕺
        arg_id_pred_num += len(pred_arg_id_data)
        arg_id_gold_num += len(gold_arg_id_data)
        try:
            arg_id_match_num += len(set(pred_arg_id_data) & set(gold_arg_id_data))
        except:
            arg_id_match_num += 0
        try:
            arg_cls_match_num += len(set(pred_arg_cls_data) & set(gold_arg_cls_data))
        except:
            arg_cls_match_num += 0
        ### calculations🕺   
        if(len(hals)>0):
            hallucination_list.append(hals) 
    #### compute Precision, Recall, and F1 for Trigger Identification
    trigger_id_f1_scores = compute_f1(trigger_id_pred_num, trigger_id_gold_num, trigger_id_match_num)
    trigger_cls_f1_scores = compute_f1(trigger_id_pred_num, trigger_id_gold_num, trigger_cls_match_num)
    arg_id_f1_scores = compute_f1(arg_id_pred_num, arg_id_gold_num, arg_id_match_num)
    arg_cls_f1_scores = compute_f1(arg_id_pred_num, arg_id_gold_num, arg_cls_match_num)
    table = our_pretty_table(trigger_id_f1_scores, trigger_cls_f1_scores, arg_id_f1_scores, arg_cls_f1_scores, task_type = "E2E")
    print(table)
    return {
        "trigger_id_precision": trigger_id_f1_scores["precision"],
        "trigger_id_recall": trigger_id_f1_scores["recall"],
        "trigger_id_f1": trigger_id_f1_scores["f1"],
        "trigger_cls_precision": trigger_cls_f1_scores["precision"],
        "trigger_cls_recall": trigger_cls_f1_scores["recall"],
        "trigger_cls_f1": trigger_cls_f1_scores["f1"],
        "arg_id_precision": arg_id_f1_scores["precision"],
        "arg_id_recall": arg_id_f1_scores["recall"],
        "arg_id_f1": arg_id_f1_scores["f1"],
        "arg_cls_precision": arg_cls_f1_scores["precision"],
        "arg_cls_recall": arg_cls_f1_scores["recall"],
        "arg_cls_f1": arg_cls_f1_scores["f1"],
        "hallucinations": hallucination_list
    }
##### --------------------------------------------- Evaluation Script and Examples for MicroE2E scores ##### ---------------------------------------------## 

def evaluator(json_data, original_json_data):
    datasets_splits = {}
    task_wise_scores, dataset_wise_scores = {}, {}
    for jsn_data, orig_jsn_data in zip(json_data, original_json_data):
        # print(orig_jsn_data)
        dataset_type = orig_jsn_data["task_type"].lower()
        # dataset_name = orig_jsn_data[0]["dataset"]
        if datasets_splits.get(dataset_type) is None:
            datasets_splits[dataset_type] = []
        datasets_splits[dataset_type].append(jsn_data)
    for dataset_split in datasets_splits:
        function_name = f'micro_{dataset_split}_scores'
        function = globals().get(function_name)
        # print(datasets_splits[dataset_split])
        print(f"Calling function {function} for the task type: {dataset_split}")
        # print(datasets_splits[dataset_split][0])
        result = function(datasets_splits[dataset_split])
        if(task_wise_scores.get(dataset_split) is None):
            task_wise_scores[dataset_split] = []
        task_wise_scores[dataset_split].append(result)
    arg_cls_scores, trg_cls_scores = [], []
    # print(task_wise_scores)
    for task_wise_key in task_wise_scores:
        if(task_wise_key=="ed"):
            trg_cls_scores.extend([x["trigger_cls_f1"] for x in task_wise_scores[task_wise_key]])
        else:
            arg_cls_scores.extend([x["arg_cls_f1"] for x in task_wise_scores[task_wise_key]])
    # print(trg_cls_scores)
    # print(arg_cls_scores)
    final_score = sum(trg_cls_scores+arg_cls_scores)/len(trg_cls_scores+arg_cls_scores)
    # print(final_score)
    return final_score


if __name__ == "__main__":
    # For Full Run
    args = parser.parse_args()
    orig_jsn = json.load(open(args.input_file))#["conversations"]

    # For Individual Run:
    full = micro_e2e_scores(orig_jsn)
