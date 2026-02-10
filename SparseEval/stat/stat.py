import sys
import os
import subprocess
import torch
import numpy as np

root = "./result_data/main_result"
error_dict = {}
tau_dict = {}
llm_dataset_list = [
    'arc', 'gsm8k', 'hellaswag', 'mmlu', 'truthfulqa', 'winogrande'
]
dataset_list = llm_dataset_list
method_list = ['sparse_eval',]
anchor_nums_list = [
    20, 
    40, 
    60, 
    80, 
    100
]

for dataset in os.listdir(root):
    if dataset not in dataset_list:
        continue
    dataset_path = os.path.join(root, dataset)
    for method in os.listdir(dataset_path):
        if method not in method_list:
            continue
        method_path = os.path.join(dataset_path, method)
        for anchor_nums in os.listdir(method_path):
            if int(anchor_nums) not in anchor_nums_list:
                continue
            anchor_nums_path = os.path.join(method_path, anchor_nums)
            load_path = os.path.join(anchor_nums_path, "main_result.pt")
            if not os.path.exists(load_path):
                continue
            result_data = torch.load(load_path, weights_only=False)
            error = np.mean([item['mae'] for item in result_data], axis=0)
            if 'kendalltau' in result_data[0]:
                # print(result_data)
                tau = np.mean([item['kendalltau'] if 'kendalltau' in item else item['kendalls_tau'] for item in result_data], axis=0)
            else:
                tau = np.mean([item['kendalls_tau'] for item in result_data], axis=0)
            if len(result_data)<10 and method in ['random', 'kmeans', 'sparse_eval']:
                postfix = f"({len(result_data)}/10)"
                # postfix = ""
            else:
                postfix = ""
            try:
                error_dict[f"{method}_{dataset}_{anchor_nums}"] = f"{error*100:.4f}% {postfix}"
                tau_dict[f"{method}_{dataset}_{anchor_nums}"] = f"{tau:.4f} {postfix}"
            except:
                error_dict[f"{method}_{dataset}_{anchor_nums}"] = f"{error[0]*100:.4f}% {postfix}"
                tau_dict[f"{method}_{dataset}_{anchor_nums}"] = f"{tau[0]:.4f} {postfix}"
            # method_list.add(method)
            # dataset_list.add(dataset)
            # anchor_nums_list.add(int(anchor_nums))

# show as a table, with dataset as rows, methods as sub-rows, and anchor numbers as columns
print(f"{'Dataset':<20}{'Method':<20}{'Anchor':<20}{'Error(%)':<20}{'Tau':<20}")
for dataset in sorted(dataset_list):
    for method in method_list:
        for anchor_nums in sorted(anchor_nums_list):
            key = f"{method}_{dataset}_{anchor_nums}"
            if key in error_dict:
                print(f"{dataset:<20}{method:<20}{anchor_nums:<20}{error_dict[key]:<20}{tau_dict[key]:<20}")   
