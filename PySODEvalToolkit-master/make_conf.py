import os
import json

GT_ROOT = "../gt"
PRED_ROOT = "../pred"

# output config filenames
DATASET_CONFIG = "./config_files/datasets.json"
METHOD_CONFIG = "./config_files/methods.json"

# Discover datasets from the gt folder
# datasets = [d for d in os.listdir(
#     GT_ROOT) if os.path.isdir(os.path.join(GT_ROOT, d))]
datasets = ['NLPR', 'SIP', 'NJU2K', 'DES', 'LFSD', 'STERE', 'SSD']
# datasets = ['DES']

# Create dataset_config dictionary
dataset_config = {}
for ds in datasets:
    dataset_config[ds] = {
        "mask": {
            "path": os.path.join(GT_ROOT, ds),
            "prefix": "",  # Assuming no prefix in ground truth filenames
            "suffix": ".png"
        }
    }

# Discover methods from the pred folder
# methods = [m for m in os.listdir(
#     PRED_ROOT) if os.path.isdir(os.path.join(PRED_ROOT, m))]

# methods = [
#     'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10'
# ]
methods = ['bbs_sl']
# Create method_configs dictionary
method_configs = {}
for m in methods:
    method_configs[m] = {}
    method_ds_dirs = [ds for ds in os.listdir(os.path.join(PRED_ROOT, m))
                      if os.path.isdir(os.path.join(PRED_ROOT, m, ds))]
    for ds in method_ds_dirs:
        if ds in datasets:  # Only include if the dataset exists in gt
            method_configs[m][ds] = {
                "path": os.path.join(PRED_ROOT, m, ds),
                "prefix": "",  # Assuming no prefix in prediction filenames
                "suffix": ".png"
            }

# Write dataset_config to JSON file
with open(DATASET_CONFIG, "w") as f:
    json.dump(dataset_config, f, indent=4)

# Write method_configs to JSON file
with open(METHOD_CONFIG, "w") as f:
    json.dump(method_configs, f, indent=4)

print("Config files created: dataset_config.json and method_configs.json")
