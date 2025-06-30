import numpy as np
from scipy import stats
import yaml
import os, glob

# find all directories in the current directory starting with "model_*"
model_dirs = glob.glob("model_*")

model_performance = {"RNN": {}, 'DTRNN': {}, "GRU": {}, "LSTM": {}, "NARMA": {}}
model_dir_names = {"RNN":{}, "DTRNN": {}, "GRU":{}, "LSTM":{}, "NARMA":{}}
model_close_loop_inputsize = {"RNN": 4, "DTRNN": 4, "GRU": 4, "LSTM": 4, "NARMA": 18}
model_performance_openloop = {"RNN": {}, "DTRNN": {}, "GRU": {}, "LSTM": {}, "NARMA": {}}
model_dir_names_openloop = {"RNN":{}, "DTRNN": {}, "GRU":{}, "LSTM":{}, "NARMA":{}}
model_open_loop_inputsize = {"RNN": 2, "DTRNN": 2, "GRU": 2, "LSTM": 2, "NARMA": 12}
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
    # load test_dh_error
    test_dh_error = np.loadtxt(os.path.join(model_dir, "test_dh_error.csv"), delimiter=',')
    # load test_dw_error
    test_dw_error = np.loadtxt(os.path.join(model_dir, "test_dw_error.csv"), delimiter=',')
    # get the model type
    model_type = params['model_type']
    # get the input size
    model_input_size = params['model_input_size']
    # get the hidden size
    model_hidden_size = params['model_hidden_size']
    # get closed loop or open loop
    if 'open_loop' in params and params['open_loop']:
        model_open_loop = True
    else:
        model_open_loop = False
    if type(model_hidden_size) is list:
        model_hidden_size = model_hidden_size[0]

    if model_input_size==model_close_loop_inputsize[model_type]:
        if model_hidden_size not in model_performance[model_type]:
            model_performance[model_type][model_hidden_size] = {}
        model_performance[model_type][model_hidden_size]['testing_loss'] = np.min(testing_loss)
        model_performance[model_type][model_hidden_size]['test_dh_error'] = np.mean(np.abs(test_dh_error))
        model_performance[model_type][model_hidden_size]['test_dw_error'] = np.mean(np.abs(test_dw_error))
        model_performance[model_type][model_hidden_size]['test_dh_error_95'] = stats.expon(scale=np.std(test_dh_error)).interval(0.95)[1]
        model_performance[model_type][model_hidden_size]['test_dw_error_95'] = stats.expon(scale=np.std(test_dw_error)).interval(0.95)[1]
        model_dir_names[model_type][model_hidden_size] = model_dir
    elif model_input_size==model_open_loop_inputsize[model_type] or model_open_loop:
        if model_hidden_size not in model_performance_openloop[model_type]:
            model_performance_openloop[model_type][model_hidden_size] = {}
        model_performance_openloop[model_type][model_hidden_size]['testing_loss'] = np.min(testing_loss)
        model_performance_openloop[model_type][model_hidden_size]['test_dh_error'] = np.mean(np.abs(test_dh_error))
        model_performance_openloop[model_type][model_hidden_size]['test_dw_error'] = np.mean(np.abs(test_dw_error))
        model_performance_openloop[model_type][model_hidden_size]['test_dh_error_95'] = stats.expon(scale=np.std(test_dh_error)).interval(0.95)[1]
        model_performance_openloop[model_type][model_hidden_size]['test_dw_error_95'] = stats.expon(scale=np.std(test_dw_error)).interval(0.95)[1]
        model_dir_names_openloop[model_type][model_hidden_size] = model_dir
    else:
        continue

# print out the table in a markdown format
# the columns are the model types and the rows are the hidden sizes, hidden sizes sorted from 3,8,16,64
table_string = {}
table_string["Closed loop testing loss"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Closed loop testing loss"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["Closed loop dh mean error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Closed loop dh mean error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["Closed loop dw mean error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Closed loop dw mean error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["Closed loop dh 95 error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Closed loop dh 95 error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["Closed loop dw 95 error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Closed loop dw 95 error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["Closed loop dir"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Closed loop dir"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["Open loop testing loss"] = "| Hidden Size | " + " | ".join(model_performance_openloop.keys()) + " |\n"
table_string["Open loop testing loss"] += "|-------------|" + " | ".join(["---"] * len(model_performance_openloop.keys())) + " |\n"
table_string["Open loop dh mean error"] = "| Hidden Size | " + " | ".join(model_performance_openloop.keys()) + " |\n"
table_string["Open loop dh mean error"] += "|-------------|" + " | ".join(["---"] * len(model_performance_openloop.keys())) + " |\n"
table_string["Open loop dw mean error"] = "| Hidden Size | " + " | ".join(model_performance_openloop.keys()) + " |\n"
table_string["Open loop dw mean error"] += "|-------------|" + " | ".join(["---"] * len(model_performance_openloop.keys())) + " |\n"
table_string["Open loop dh 95 error"] = "| Hidden Size | " + " | ".join(model_performance_openloop.keys()) + " |\n"
table_string["Open loop dh 95 error"] += "|-------------|" + " | ".join(["---"] * len(model_performance_openloop.keys())) + " |\n"
table_string["Open loop dw 95 error"] = "| Hidden Size | " + " | ".join(model_performance_openloop.keys()) + " |\n"
table_string["Open loop dw 95 error"] += "|-------------|" + " | ".join(["---"] * len(model_performance_openloop.keys())) + " |\n"
table_string["Open loop dir"] = "| Hidden Size | " + " | ".join(model_performance_openloop.keys()) + " |\n"
table_string["Open loop dir"] += "|-------------|" + " | ".join(["---"] * len(model_performance_openloop.keys())) + " |\n"

hidden_sizes = sorted(set(size for sizes in model_performance.values() for size in sizes.keys()))
for hidden_size in hidden_sizes:
    for stat_key in table_string.keys():
        table_string[stat_key] += f"| {hidden_size:<11} | "
    
    for model_type in model_performance.keys():
        if hidden_size in model_performance[model_type]:
            table_string["Closed loop testing loss"] += f"{model_performance[model_type][hidden_size]['testing_loss']:<7.4f} | "
            table_string["Closed loop dh mean error"] += f"{model_performance[model_type][hidden_size]['test_dh_error']:<7.4f} | "
            table_string["Closed loop dw mean error"] += f"{model_performance[model_type][hidden_size]['test_dw_error']:<7.4f} | "
            table_string["Closed loop dh 95 error"] += f"{model_performance[model_type][hidden_size]['test_dh_error_95']:<7.4f} | "
            table_string["Closed loop dw 95 error"] += f"{model_performance[model_type][hidden_size]['test_dw_error_95']:<7.4f} | "
            table_string["Closed loop dir"] += f"{model_dir_names[model_type][hidden_size][6:]:<15} | "
            table_string["Open loop testing loss"] += f"{model_performance_openloop[model_type][hidden_size]['testing_loss']:<7.4f} | "
            table_string["Open loop dh mean error"] += f"{model_performance_openloop[model_type][hidden_size]['test_dh_error']:<7.4f} | "
            table_string["Open loop dw mean error"] += f"{model_performance_openloop[model_type][hidden_size]['test_dw_error']:<7.4f} | "
            table_string["Open loop dh 95 error"] += f"{model_performance_openloop[model_type][hidden_size]['test_dh_error_95']:<7.4f} | "
            table_string["Open loop dw 95 error"] += f"{model_performance_openloop[model_type][hidden_size]['test_dw_error_95']:<7.4f} | "
            table_string["Open loop dir"] += f"{model_dir_names_openloop[model_type][hidden_size][6:]:<15} | "
        else:
            table_string["Closed loop testing loss"] += "N/A       | "
            table_string["Closed loop dh mean error"] += "N/A       | "
            table_string["Closed loop dw mean error"] += "N/A       | "
            table_string["Closed loop dh 95 error"] += "N/A       | "
            table_string["Closed loop dw 95 error"] += "N/A       | "
            table_string["Closed loop dir"] += "N/A                   | "
            table_string["Open loop testing loss"] += "N/A       | "
            table_string["Open loop dh mean error"] += "N/A       | "
            table_string["Open loop dw mean error"] += "N/A       | "
            table_string["Open loop dh 95 error"] += "N/A       | "
            table_string["Open loop dw 95 error"] += "N/A       | "
            table_string["Open loop dir"] += "N/A                   | "
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
model_performance_openloop = {k: dict(sorted(v.items())) for k, v in model_performance_openloop.items()}

for stats_key in ['testing_loss', 'test_dh_error', 'test_dw_error', 'test_dh_error_95', 'test_dw_error_95']:
    
    this_closed_loop_table = {}
    this_open_loop_table = {}
    for model_type in model_performance.keys():
        this_closed_loop_table[model_type] = {}
        this_open_loop_table[model_type] = {}
        for hidden_size in sorted(model_performance[model_type].keys()):
            this_closed_loop_table[model_type][hidden_size] = model_performance[model_type][hidden_size][stats_key]
            this_open_loop_table[model_type][hidden_size] = model_performance_openloop[model_type][hidden_size][stats_key]

    # the numbers are rounded to 4 decimal places
    performance_df = pd.DataFrame(this_closed_loop_table)
    performance_df.index.name = stats_key
    performance_df.to_csv(f"close_loop_{stats_key}.csv", float_format='%.4f')

    open_loop_performance_df = pd.DataFrame(this_open_loop_table)
    open_loop_performance_df.index.name = stats_key
    open_loop_performance_df.to_csv(f"open_loop_{stats_key}.csv", float_format='%.4f')

# Load both CSV files
for training_type in ['close_loop', 'open_loop']:
    for geometry in ['test_dh', 'test_dw']:
        error_file = pd.read_csv(f"{training_type}_{geometry}_error.csv", index_col=0)
        error_95_file = pd.read_csv(f"{training_type}_{geometry}_error_95.csv", index_col=0)

        # Combine the two dataframes into one with tuple strings
        combined = error_file.combine(error_95_file, lambda x, y: x.map(str) + ", " + y.map(str))

        # Add parentheses around the combined values
        combined = combined.map(lambda x: f"({x})")

        # Save to Excel
        excel_path = f"{training_type}_{geometry}_error_stats.xlsx"
        combined.to_excel(excel_path)

        # delete the csv files
        os.remove(f"{training_type}_{geometry}_error.csv")
        os.remove(f"{training_type}_{geometry}_error_95.csv")