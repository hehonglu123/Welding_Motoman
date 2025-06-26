import numpy as np
import yaml
import os, glob

# find all directories in the current directory starting with "model_*"
model_dirs = glob.glob("model_*")

model_performance = {"RNN": {}, "GRU": {}, "LSTM": {}, "NARMA": {}}
model_dir_names = {"RNN":{}, "GRU":{}, "LSTM":{}, "NARMA":{}}
model_close_loop_inputsize = {"RNN": 4, "GRU": 4, "LSTM": 4, "NARMA": 12}
model_performance_openloop = {"RNN": {}, "GRU": {}, "LSTM": {}, "NARMA": {}}
model_dir_names_openloop = {"RNN":{}, "GRU":{}, "LSTM":{}, "NARMA":{}}
model_open_loop_inputsize = 2
# loop through each model directory and analyze the YAML files
for model_dir in model_dirs:
    # read the yaml file traininig_params.yaml
    with open(os.path.join(model_dir, "training_params.yaml"), 'r') as file:
        params = yaml.safe_load(file)
    # skip if epochs is not 5000
    if params['epochs'] != 5000:
        continue
    # load testing_loss.csv
    testing_loss = np.loadtxt(os.path.join(model_dir, "testing_loss.csv"), delimiter=',')
    # get the model type
    model_type = params['model_type']
    # get the input size
    model_input_size = params['model_input_size']
    # get the hidden size
    model_hidden_size = params['model_hidden_size']
    if type(model_hidden_size) is list:
        model_hidden_size = model_hidden_size[0]
    if model_input_size==model_close_loop_inputsize[model_type]:
        model_performance[model_type][model_hidden_size] = np.min(testing_loss)
        model_dir_names[model_type][model_hidden_size] = model_dir
    elif model_input_size==model_open_loop_inputsize:
        model_performance_openloop[model_type][model_hidden_size] = np.min(testing_loss)
        model_dir_names_openloop[model_type][model_hidden_size] = model_dir
    else:
        continue
    
# print out the table in a markdown format
# the columns are the model types and the rows are the hidden sizes, hidden sizes sorted from 3,8,16,64
print("# Model Performance Analysis")
print("| Hidden Size | " + " | ".join(model_performance.keys()) + " |")
print("|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |")
hidden_sizes = sorted(set(size for sizes in model_performance.values() for size in sizes.keys()))
for hidden_size in hidden_sizes:
    row = f"| {hidden_size:<11} | "
    for model_type in model_performance.keys():
        if hidden_size in model_performance[model_type]:
            row += f"{model_performance[model_type][hidden_size]:<7.4f} | "
        else:
            row += "N/A       | "
    print(row)
# print out the model directory names
print("\nModel Directory Names:")
print("| Hidden Size | " + " | ".join(model_dir_names.keys()) + " |")
print("|-------------|" + " | ".join(["---"] * len(model_dir_names.keys())) + " |")
for hidden_size in hidden_sizes:
    row = f"| {hidden_size:<11} | "
    for model_type in model_dir_names.keys():
        if hidden_size in model_dir_names[model_type]:
            row += f"{model_dir_names[model_type][hidden_size]:<20} | "
        else:
            row += "N/A                   | "
    print(row)
# print out the open loop performance
print("\nOpen Loop Performance:")
print("| Hidden Size | " + " | ".join(model_performance_openloop.keys()) + " |")
print("|-------------|" + " | ".join(["---"] * len(model_performance_openloop.keys())) + " |")
for hidden_size in hidden_sizes:
    row = f"| {hidden_size:<11} | "
    for model_type in model_performance_openloop.keys():
        if hidden_size in model_performance_openloop[model_type]:
            row += f"{model_performance_openloop[model_type][hidden_size]:<7.4f} | "
        else:
            row += "N/A       | "
    print(row)
# print out the model directory names for open loop
print("\nOpen Loop Model Directory Names:")
print("| Hidden Size | " + " | ".join(model_dir_names_openloop.keys()) + " |")
print("|-------------|" + " | ".join(["---"] * len(model_dir_names_openloop.keys())) + " |")
for hidden_size in hidden_sizes:
    row = f"| {hidden_size:<11} | "
    for model_type in model_dir_names_openloop.keys():
        if hidden_size in model_dir_names_openloop[model_type]:
            row += f"{model_dir_names_openloop[model_type][hidden_size]:<20} | "
        else:
            row += "N/A                   | "
    print(row)

# save the performance table as a csv file so I can open with excel
# the columns are the model types and the rows are the hidden sizes, hidden sizes sorted from 3,8,16,64
model_performance = {k: dict(sorted(v.items())) for k, v in model_performance.items()}
model_performance_openloop = {k: dict(sorted(v.items())) for k, v in model_performance_openloop.items()}
# the numbers are rounded to 4 decimal places
import pandas as pd
performance_df = pd.DataFrame(model_performance)
performance_df.index.name = 'Hidden Size'
performance_df.to_csv("close_loop_performance.csv", float_format='%.4f')

open_loop_performance_df = pd.DataFrame(model_performance_openloop)
open_loop_performance_df.index.name = 'Hidden Size'
open_loop_performance_df.to_csv("open_loop_performance.csv", float_format='%.4f')