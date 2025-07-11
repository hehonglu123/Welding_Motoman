import numpy as np
from scipy import stats
import yaml
import os, glob
from matplotlib import pyplot as plt

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
    # load test_cmd_v_feedrate
    test_cmd_v_feedrate = np.loadtxt("test_cmd_v_feedrate.csv", delimiter=',', skiprows=1)
    test_cmd_v_feedrate[:,0]=np.round(test_cmd_v_feedrate[:,0], 2)  # round cmd_v to 2 decimal places
    test_cmd_v_feedrate[:,1]=np.round(test_cmd_v_feedrate[:,1])  # round cmd_feedrate to integer
    # load training time elapsed
    training_time_elapsed = np.loadtxt(os.path.join(model_dir, "training_time.csv"), delimiter=',')
    training_time_elapsed = float(training_time_elapsed)
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
        model_performance[model_type][model_hidden_size]['test_dh_error_max'] = np.max(np.abs(test_dh_error))
        model_performance[model_type][model_hidden_size]['test_dw_error_max'] = np.max(np.abs(test_dw_error))
        arg_max_error_dh = np.argmax(np.abs(test_dh_error))
        arg_max_error_dw = np.argmax(np.abs(test_dw_error))
        model_performance[model_type][model_hidden_size]['test_dh_error_argmax'] = (arg_max_error_dh//40, arg_max_error_dh%40)
        model_performance[model_type][model_hidden_size]['test_dw_error_argmax'] = (arg_max_error_dw//40, arg_max_error_dw%40)
        model_performance[model_type][model_hidden_size]['cmd_v_feedrate_dh_max_error'] = test_cmd_v_feedrate[arg_max_error_dh]
        model_performance[model_type][model_hidden_size]['cmd_v_feedrate_dw_max_error'] = test_cmd_v_feedrate[arg_max_error_dw]
        model_performance[model_type][model_hidden_size]['training_time_elapsed'] = round(training_time_elapsed)
        model_dir_names[model_type][model_hidden_size] = model_dir
        # test_dh_error = np.reshape(test_dh_error, (40,-1))  # ensure it's a 2D array
        # plt.plot(np.abs(test_dh_error), '-o')
        # plt.title(f"{model_type} - {model_hidden_size} hidden size - Closed Loop DH Error")
        # plt.show()
    elif model_input_size==model_open_loop_inputsize[model_type] or model_open_loop:
        if model_hidden_size not in model_performance_openloop[model_type]:
            model_performance_openloop[model_type][model_hidden_size] = {}
        model_performance_openloop[model_type][model_hidden_size]['testing_loss'] = np.min(testing_loss)
        model_performance_openloop[model_type][model_hidden_size]['test_dh_error'] = np.mean(np.abs(test_dh_error))
        model_performance_openloop[model_type][model_hidden_size]['test_dw_error'] = np.mean(np.abs(test_dw_error))
        model_performance_openloop[model_type][model_hidden_size]['test_dh_error_95'] = stats.expon(scale=np.std(test_dh_error)).interval(0.95)[1]
        model_performance_openloop[model_type][model_hidden_size]['test_dw_error_95'] = stats.expon(scale=np.std(test_dw_error)).interval(0.95)[1]
        model_performance_openloop[model_type][model_hidden_size]['test_dh_error_max'] = np.max(np.abs(test_dh_error))
        model_performance_openloop[model_type][model_hidden_size]['test_dw_error_max'] = np.max(np.abs(test_dw_error))
        arg_max_error_dh = np.argmax(np.abs(test_dh_error))
        arg_max_error_dw = np.argmax(np.abs(test_dw_error))
        model_performance_openloop[model_type][model_hidden_size]['test_dh_error_argmax'] = (arg_max_error_dh//40, arg_max_error_dh%40)
        model_performance_openloop[model_type][model_hidden_size]['test_dw_error_argmax'] = (arg_max_error_dw//40, arg_max_error_dw%40)
        model_performance_openloop[model_type][model_hidden_size]['cmd_v_feedrate_dh_max_error'] = test_cmd_v_feedrate[arg_max_error_dh]
        model_performance_openloop[model_type][model_hidden_size]['cmd_v_feedrate_dw_max_error'] = test_cmd_v_feedrate[arg_max_error_dw]
        model_performance_openloop[model_type][model_hidden_size]['training_time_elapsed'] = round(training_time_elapsed)
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
table_string["Closed loop dh max error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Closed loop dh max error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["Closed loop dw max error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Closed loop dw max error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["Closed loop dh argmax error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Closed loop dh argmax error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["Closed loop dw argmax error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Closed loop dw argmax error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["Closed loop dh cmd_v_feedrate max error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Closed loop dh cmd_v_feedrate max error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["Closed loop dw cmd_v_feedrate max error"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Closed loop dw cmd_v_feedrate max error"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["Closed loop dir"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Closed loop dir"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
table_string["Closed loop training time"] = "| Hidden Size | " + " | ".join(model_performance.keys()) + " |\n"
table_string["Closed loop training time"] += "|-------------|" + " | ".join(["---"] * len(model_performance.keys())) + " |\n"
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
table_string["Open loop dh max error"] = "| Hidden Size | " + " | ".join(model_performance_openloop.keys()) + " |\n"
table_string["Open loop dh max error"] += "|-------------|" + " | ".join(["---"] * len(model_performance_openloop.keys())) + " |\n"
table_string["Open loop dw max error"] = "| Hidden Size | " + " | ".join(model_performance_openloop.keys()) + " |\n"
table_string["Open loop dw max error"] += "|-------------|" + " | ".join(["---"] * len(model_performance_openloop.keys())) + " |\n"
table_string["Open loop dh argmax error"] = "| Hidden Size | " + " | ".join(model_performance_openloop.keys()) + " |\n"
table_string["Open loop dh argmax error"] += "|-------------|" + " | ".join(["---"] * len(model_performance_openloop.keys())) + " |\n"
table_string["Open loop dw argmax error"] = "| Hidden Size | " + " | ".join(model_performance_openloop.keys()) + " |\n"
table_string["Open loop dw argmax error"] += "|-------------|" + " | ".join(["---"] * len(model_performance_openloop.keys())) + " |\n"
table_string["Open loop dh cmd_v_feedrate max error"] = "| Hidden Size | " + " | ".join(model_performance_openloop.keys()) + " |\n"
table_string["Open loop dh cmd_v_feedrate max error"] += "|-------------|" + " | ".join(["---"] * len(model_performance_openloop.keys())) + " |\n"
table_string["Open loop dw cmd_v_feedrate max error"] = "| Hidden Size | " + " | ".join(model_performance_openloop.keys()) + " |\n"
table_string["Open loop dw cmd_v_feedrate max error"] += "|-------------|" + " | ".join(["---"] * len(model_performance_openloop.keys())) + " |\n"
table_string["Open loop training time"] = "| Hidden Size | " + " | ".join(model_performance_openloop.keys()) + " |\n"
table_string["Open loop training time"] += "|-------------|" + " | ".join(["---"] * len(model_performance_openloop.keys())) + " |\n"
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
            table_string["Closed loop dh max error"] += f"{model_performance[model_type][hidden_size]['test_dh_error_max']:<7.4f} | "
            table_string["Closed loop dw max error"] += f"{model_performance[model_type][hidden_size]['test_dw_error_max']:<7.4f} | "
            table_string["Closed loop dh argmax error"] += f"({model_performance[model_type][hidden_size]['test_dh_error_argmax'][0]:<4},{model_performance[model_type][hidden_size]['test_dh_error_argmax'][1]:<3}) | "
            table_string["Closed loop dw argmax error"] += f"({model_performance[model_type][hidden_size]['test_dw_error_argmax'][0]:<4},{model_performance[model_type][hidden_size]['test_dw_error_argmax'][1]:<3}) | "
            table_string["Closed loop dh cmd_v_feedrate max error"] += f"({model_performance[model_type][hidden_size]['cmd_v_feedrate_dh_max_error'][0]:<3},{model_performance[model_type][hidden_size]['cmd_v_feedrate_dh_max_error'][1]:<4}) | "
            table_string["Closed loop dw cmd_v_feedrate max error"] += f"({model_performance[model_type][hidden_size]['cmd_v_feedrate_dw_max_error'][0]:<3},{model_performance[model_type][hidden_size]['cmd_v_feedrate_dw_max_error'][1]:<4}) | "
            table_string["Closed loop dir"] += f"{model_dir_names[model_type][hidden_size][6:]:<15} | "
            table_string["Closed loop training time"] += f"{model_performance[model_type][hidden_size]['training_time_elapsed']:<7.2f} | "
            table_string["Open loop testing loss"] += f"{model_performance_openloop[model_type][hidden_size]['testing_loss']:<7.4f} | "
            table_string["Open loop dh mean error"] += f"{model_performance_openloop[model_type][hidden_size]['test_dh_error']:<7.4f} | "
            table_string["Open loop dw mean error"] += f"{model_performance_openloop[model_type][hidden_size]['test_dw_error']:<7.4f} | "
            table_string["Open loop dh 95 error"] += f"{model_performance_openloop[model_type][hidden_size]['test_dh_error_95']:<7.4f} | "
            table_string["Open loop dw 95 error"] += f"{model_performance_openloop[model_type][hidden_size]['test_dw_error_95']:<7.4f} | "
            table_string["Open loop dh max error"] += f"{model_performance_openloop[model_type][hidden_size]['test_dh_error_max']:<7.4f} | "
            table_string["Open loop dw max error"] += f"{model_performance_openloop[model_type][hidden_size]['test_dw_error_max']:<7.4f} | "
            table_string["Open loop dh argmax error"] += f"({model_performance_openloop[model_type][hidden_size]['test_dh_error_argmax'][0]:<4},{model_performance_openloop[model_type][hidden_size]['test_dh_error_argmax'][1]:<3}) | "
            table_string["Open loop dw argmax error"] += f"({model_performance_openloop[model_type][hidden_size]['test_dw_error_argmax'][0]:<4},{model_performance_openloop[model_type][hidden_size]['test_dw_error_argmax'][1]:<3}) | "
            table_string["Open loop dh cmd_v_feedrate max error"] += f"({model_performance_openloop[model_type][hidden_size]['cmd_v_feedrate_dh_max_error'][0]:<3},{model_performance_openloop[model_type][hidden_size]['cmd_v_feedrate_dh_max_error'][1]:<4}) | "
            table_string["Open loop dw cmd_v_feedrate max error"] += f"({model_performance_openloop[model_type][hidden_size]['cmd_v_feedrate_dw_max_error'][0]:<3},{model_performance_openloop[model_type][hidden_size]['cmd_v_feedrate_dw_max_error'][1]:<4}) | "
            table_string["Open loop dir"] += f"{model_dir_names_openloop[model_type][hidden_size][6:]:<15} | "
            table_string["Open loop training time"] += f"{model_performance_openloop[model_type][hidden_size]['training_time_elapsed']:<7.2f} | "
        else:
            table_string["Closed loop testing loss"] += "N/A       | "
            table_string["Closed loop dh mean error"] += "N/A       | "
            table_string["Closed loop dw mean error"] += "N/A       | "
            table_string["Closed loop dh 95 error"] += "N/A       | "
            table_string["Closed loop dw 95 error"] += "N/A       | "
            table_string["Closed loop dh max error"] += "N/A       | "
            table_string["Closed loop dw max error"] += "N/A       | "
            table_string["Closed loop dir"] += "N/A                   | "
            table_string["Closed loop training time"] += "N/A       | "
            table_string["Open loop testing loss"] += "N/A       | "
            table_string["Open loop dh mean error"] += "N/A       | "
            table_string["Open loop dw mean error"] += "N/A       | "
            table_string["Open loop dh 95 error"] += "N/A       | "
            table_string["Open loop dw 95 error"] += "N/A       | "
            table_string["Open loop dh max error"] += "N/A       | "
            table_string["Open loop dw max error"] += "N/A       | "
            table_string["Open loop dir"] += "N/A                   | "
            table_string["Open loop training time"] += "N/A       | "
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

for stats_key in ['testing_loss', 'training_time_elapsed', 'test_dh_error', 'test_dw_error', 'test_dh_error_95', 'test_dw_error_95', 'test_dh_error_max', 'test_dw_error_max']:
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
        error_max_file = pd.read_csv(f"{training_type}_{geometry}_error_max.csv", index_col=0)

        # Combine the two dataframes into one with tuple strings
        combined = error_file.combine(error_95_file, lambda x, y: x.map(str) + ", " + y.map(str))

        # Combine with max error
        combined = combined.combine(error_max_file, lambda x, y: x + ", " + y.map(str))

        # Add parentheses around the combined values
        combined = combined.map(lambda x: f"({x})")

        # Save to Excel
        excel_path = f"{training_type}_{geometry}_error_stats.xlsx"
        combined.to_excel(excel_path)

        # delete the csv files
        os.remove(f"{training_type}_{geometry}_error.csv")
        os.remove(f"{training_type}_{geometry}_error_95.csv")
        os.remove(f"{training_type}_{geometry}_error_max.csv")