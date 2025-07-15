import numpy as np
from scipy import stats
import yaml
import os, glob
from matplotlib import pyplot as plt

# find all directories in the current directory starting with "model_*"
model_dirs = glob.glob("model_*")

model_performance = {"RNN Open": {}, 'RNN Closed Pre-Trained Open': {}, "RNN Closed": {}}
model_dir_names = {"RNN Open": {}, 'RNN Closed Pre-Trained Open': {}, "RNN Closed": {}}
model_types = {"RNN Open": "RNN", 'RNN Closed Pre-Trained Open': "RNN", "RNN Closed": "RNN"}
model_inputsize = {"RNN Open": 2, 'RNN Closed Pre-Trained Open': 4, "RNN Closed": 4}
model_pretrained = {"RNN Open": False, 'RNN Closed Pre-Trained Open': True, "RNN Closed": False}
model_openloop = {"RNN Open": True, 'RNN Closed Pre-Trained Open': False, "RNN Closed": False}

# loop through each model directory and analyze the YAML files
for model_dir in model_dirs:
    # read the yaml file traininig_params.yaml
    with open(os.path.join(model_dir, "training_params.yaml"), 'r') as file:
        params = yaml.safe_load(file)
    
    # get the model type
    this_model_type = params['model_type']
    # get the input size
    this_model_input_size = params['model_input_size']
    # get the hidden size
    this_model_hidden_size = params['model_hidden_size']
    # get closed loop or open loop
    if 'open_loop' in params and params['open_loop']:
        this_model_open_loop = True
    elif this_model_type != 'NARMA' and this_model_input_size in [2,3]:
        this_model_open_loop = True
    else:    
        this_model_open_loop = False
    if type(this_model_hidden_size) is list:
        this_model_hidden_size = this_model_hidden_size[0]
    # get if model is pretrained
    this_model_pretrained = True if 'pre_trained_model_dir' in params.keys() and params['pre_trained_model_dir'] is not None else False

    this_key = ''
    for key in model_types.keys():
        if model_types[key] == this_model_type and model_inputsize[key] == this_model_input_size\
            and model_pretrained[key] == this_model_pretrained and model_openloop[key] == this_model_open_loop:
            this_key = key
            break
    if this_key == '':
        # print("modedirs:", model_dir)
        # print(f"Model type {this_model_type} with input size {this_model_input_size}, hidden size {this_model_hidden_size}, open loop {this_model_open_loop}, pretrained {this_model_pretrained} not found in model_types dictionary.")
        continue  # skip if no matching key found

    # load testing_loss.csv
    testing_loss = np.loadtxt(os.path.join(model_dir, "testing_loss.csv"), delimiter=',')
    # load test_dh_error
    test_dh_error = np.loadtxt(os.path.join(model_dir, "test_dh_error.csv"), delimiter=',')
    # load test_dw_error
    test_dw_error = np.loadtxt(os.path.join(model_dir, "test_dw_error.csv"), delimiter=',')
    # load test_cmd_v_feedrate
    test_cmd_v_feedrate = np.loadtxt("test_cmd_v_feedrate.csv", delimiter=',', skiprows=1)
    test_cmd_v_feedrate[:,0]=np.round(test_cmd_v_feedrate[:,0], 2)  # round cmd_v to 2 decimal places
    test_cmd_v_feedrate[:,1]=np.round(test_cmd_v_feedrate[:,1])  # round cmd_feedrate to integer
    # load training time elapsed
    training_time_elapsed = np.loadtxt(os.path.join(model_dir, "training_time.csv"), delimiter=',')
    training_time_elapsed = float(training_time_elapsed)
    
    # store the performance in the model_performance dictionary
    if this_model_hidden_size not in model_performance[this_key]:
        model_performance[this_key][this_model_hidden_size] = {}
    model_performance[this_key][this_model_hidden_size]['testing_loss'] = np.min(testing_loss)
    model_performance[this_key][this_model_hidden_size]['test_dh_error'] = np.mean(np.abs(test_dh_error))
    model_performance[this_key][this_model_hidden_size]['test_dw_error'] = np.mean(np.abs(test_dw_error))
    model_performance[this_key][this_model_hidden_size]['test_dh_error_95'] = stats.expon(scale=np.std(test_dh_error)).interval(0.95)[1]
    model_performance[this_key][this_model_hidden_size]['test_dw_error_95'] = stats.expon(scale=np.std(test_dw_error)).interval(0.95)[1]
    model_performance[this_key][this_model_hidden_size]['test_dh_error_max'] = np.max(np.abs(test_dh_error))
    model_performance[this_key][this_model_hidden_size]['test_dw_error_max'] = np.max(np.abs(test_dw_error))
    arg_max_error_dh = np.argmax(np.abs(test_dh_error))
    arg_max_error_dw = np.argmax(np.abs(test_dw_error))
    model_performance[this_key][this_model_hidden_size]['test_dh_error_argmax'] = (arg_max_error_dh//40, arg_max_error_dh%40)
    model_performance[this_key][this_model_hidden_size]['test_dw_error_argmax'] = (arg_max_error_dw//40, arg_max_error_dw%40)
    model_performance[this_key][this_model_hidden_size]['cmd_v_feedrate_dh_max_error'] = test_cmd_v_feedrate[arg_max_error_dh]
    model_performance[this_key][this_model_hidden_size]['cmd_v_feedrate_dw_max_error'] = test_cmd_v_feedrate[arg_max_error_dw]
    model_performance[this_key][this_model_hidden_size]['training_time_elapsed'] = round(training_time_elapsed)
    model_dir_names[this_key][this_model_hidden_size] = model_dir
    # test_dh_error = np.reshape(test_dh_error, (40,-1))  # ensure it's a 2D array
    # plt.plot(np.abs(test_dh_error), '-o')
    # plt.title(f"{model_type} - {model_hidden_size} hidden size - Closed Loop DH Error")
    # plt.show()

# print out the table in a markdown format
# the columns are the model types and the rows are the hidden sizes, hidden sizes sorted from 3,8,16,64
table_string = {}
table_string["Testing loss"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Testing loss"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["dh mean error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["dh mean error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["dw mean error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["dw mean error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["dh 95 error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["dh 95 error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["dw 95 error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["dw 95 error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["dh max error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["dh max error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["dw max error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["dw max error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["dh argmax error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["dh argmax error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["dw argmax error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["dw argmax error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["dh cmd_v_feedrate max error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["dh cmd_v_feedrate max error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["dw cmd_v_feedrate max error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["dw cmd_v_feedrate max error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["model dir"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["model dir"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["Training time"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Training time"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"

hidden_sizes = sorted(set(size for sizes in model_performance.values() for size in sizes.keys()))
for hidden_size in hidden_sizes:
    for stat_key in table_string.keys():
        table_string[stat_key] += f"| {hidden_size:<11} | "
    
    for model_type in model_performance.keys():
        if hidden_size in model_performance[model_type]:
            table_string["Testing loss"] += f"{model_performance[model_type][hidden_size]['testing_loss']:<7.4f} | "
            table_string["dh mean error"] += f"{model_performance[model_type][hidden_size]['test_dh_error']:<7.4f} | "
            table_string["dw mean error"] += f"{model_performance[model_type][hidden_size]['test_dw_error']:<7.4f} | "
            table_string["dh 95 error"] += f"{model_performance[model_type][hidden_size]['test_dh_error_95']:<7.4f} | "
            table_string["dw 95 error"] += f"{model_performance[model_type][hidden_size]['test_dw_error_95']:<7.4f} | "
            table_string["dh max error"] += f"{model_performance[model_type][hidden_size]['test_dh_error_max']:<7.4f} | "
            table_string["dw max error"] += f"{model_performance[model_type][hidden_size]['test_dw_error_max']:<7.4f} | "
            table_string["dh argmax error"] += f"({model_performance[model_type][hidden_size]['test_dh_error_argmax'][0]:<4},{model_performance[model_type][hidden_size]['test_dh_error_argmax'][1]:<3}) | "
            table_string["dw argmax error"] += f"({model_performance[model_type][hidden_size]['test_dw_error_argmax'][0]:<4},{model_performance[model_type][hidden_size]['test_dw_error_argmax'][1]:<3}) | "
            table_string["dh cmd_v_feedrate max error"] += f"({model_performance[model_type][hidden_size]['cmd_v_feedrate_dh_max_error'][0]:<3},{model_performance[model_type][hidden_size]['cmd_v_feedrate_dh_max_error'][1]:<4}) | "
            table_string["dw cmd_v_feedrate max error"] += f"({model_performance[model_type][hidden_size]['cmd_v_feedrate_dw_max_error'][0]:<3},{model_performance[model_type][hidden_size]['cmd_v_feedrate_dw_max_error'][1]:<4}) | "
            table_string["model dir"] += f"{model_dir_names[model_type][hidden_size][6:]:<15} | "
            table_string["Training time"] += f"{model_performance[model_type][hidden_size]['training_time_elapsed']:<7.2f} | "
        else:
            table_string["Testing loss"] += "N/A       | "
            table_string["dh mean error"] += "N/A       | "
            table_string["dw mean error"] += "N/A       | "
            table_string["dh 95 error"] += "N/A       | "
            table_string["dw 95 error"] += "N/A       | "
            table_string["dh max error"] += "N/A       | "
            table_string["dw max error"] += "N/A       | "
            table_string["dh argmax error"] += "N/A       | "
            table_string["dw argmax error"] += "N/A       | "
            table_string["dh cmd_v_feedrate max error"] += "N/A       | "
            table_string["dw cmd_v_feedrate max error"] += "N/A       | "
            table_string["model dir"] += "N/A                   | "
            table_string["Training time"] += "N/A       | "
    for stat_key in table_string.keys():
        table_string[stat_key] += "\n"

# print the table
for stat_key, content in table_string.items():
    print(f"### {stat_key}")
    print(content)

# save the performance table as a csv file so I can open with excel
import pandas as pd
# the columns are the model types and the rows are the hidden sizes, hidden sizes sorted from 3,8,16,64
model_performance = {k: dict(sorted(v.items())) for k, v in model_performance.items()}

for stats_key in ['testing_loss', 'training_time_elapsed', 'test_dh_error', 'test_dw_error', 'test_dh_error_95', 'test_dw_error_95', 'test_dh_error_max', 'test_dw_error_max']:
    this_closed_loop_table = {}
    this_open_loop_table = {}
    for model_type in model_performance.keys():
        this_closed_loop_table[model_type] = {}
        this_open_loop_table[model_type] = {}
        for hidden_size in sorted(model_performance[model_type].keys()):
            this_closed_loop_table[model_type][hidden_size] = model_performance[model_type][hidden_size][stats_key]

    # the numbers are rounded to 4 decimal places
    performance_df = pd.DataFrame(this_closed_loop_table)
    performance_df.index.name = stats_key
    performance_df.to_csv(f"stats_{stats_key}.csv", float_format='%.4f')

# Load both CSV files
for geometry in ['test_dh', 'test_dw']:
    error_file = pd.read_csv(f"stats_{geometry}_error.csv", index_col=0)
    error_95_file = pd.read_csv(f"stats_{geometry}_error_95.csv", index_col=0)
    error_max_file = pd.read_csv(f"stats_{geometry}_error_max.csv", index_col=0)

    # Combine the two dataframes into one with tuple strings
    combined = error_file.combine(error_95_file, lambda x, y: x.map(str) + ", " + y.map(str))

    # Combine with max error
    combined = combined.combine(error_max_file, lambda x, y: x + ", " + y.map(str))

    # Add parentheses around the combined values
    combined = combined.map(lambda x: f"({x})")

    # Save to Excel
    excel_path = f"stats_{geometry}_error_combined.xlsx"
    combined.to_excel(excel_path)

    # delete the csv files
    os.remove(f"stats_{geometry}_error.csv")
    os.remove(f"stats_{geometry}_error_95.csv")
    os.remove(f"stats_{geometry}_error_max.csv")