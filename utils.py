import subprocess
import json
import os
import re 

defects4j_path = "~/Work/defects4j/framework/bin/defects4j"
fine_tune_train_data_dir = './fine-tune-train-data'
train_data_dir = './train-data'
cloned_dir = './cloned-projects'
prediction_token = "<extra_id_0>"
bug_token = "[BUG]"
context_token = "[CONTEXT]"

txt_field = "txt"
ctxs_field = "ctxs"
bug_field = "bug"
fix_field = "fix"

# prediction info fields 
prediction_field = "prediction"
delete_add_field = "delete_added"
from_prev_prediction_field = "from_prev_prediction"
context_distance_field = "context_distance"
prediction_length_field = "prediction_length"

# context info field 
context_field = "context"
distance_field = "distance"

def create_prediction_info():
    return {
        prediction_field: "",
        delete_add_field: False,
        from_prev_prediction_field: False,
        context_distance_field: 0,
        prediction_length_field: -1
    }

def get_context_info(context, distance):
    return {
        context_field: context,
        distance_field: distance
    }

def replace_prediction_token(input_str, replacement):
    return input_str.replace(prediction_token, replacement)

def replace_placeholder(input_str, placeholder, replacement):
    return input_str.replace(placeholder, replacement)

def extract_content_between(text, start_phrase, end_phrase):
    pattern = re.escape(start_phrase) + r".*?" + re.escape(end_phrase)
    match = re.search(pattern, text)
    if match:
        context = match.group(0)
        context = context.replace(start_phrase, "").replace(end_phrase, "").strip()
        return context
    else:
        return None

def replace_prediction_token(context, replace_text):
    return context.replace(prediction_token, replace_text)

def get_available_bug_count_for_projects_in(dir_path):
    result = {}
    available_projects = get_list_of_dirs_in(dir_path)
    for project in available_projects:
        files = get_files_inside_dir(os.path.join(fine_tune_train_data_dir, project))
        json_files = [file for file in files if file.endswith('.json')]
        result[project] = len(json_files)
    return result

def get_available_bug_count_for_projects_in_train_data():
    return get_available_bug_count_for_projects_in(train_data_dir)

def get_available_bug_count_for_projects_in_fine_tune_train_data():
    return get_available_bug_count_for_projects_in(fine_tune_train_data_dir)

def get_available_bug_count_for_projects_in_cloned_data():
    result = {}
    dirs = get_list_of_dirs_in(cloned_dir)
    for dir in dirs:
        if dir in ['.git', '.svn']:
            continue
        bug_dirs = get_list_of_dirs_in(os.path.join(cloned_dir, dir))
        result[dir] = len(bug_dirs) 
    return result

def get_pids_list():
    get_pids_command = f"{defects4j_path} pids"
    output = subprocess.check_output(get_pids_command, shell=True, text=True)
    return output.strip().split('\n')

def get_bids_list(pid):
    get_bids_command = f"{defects4j_path} bids -p {pid}"
    output = subprocess.check_output(get_bids_command, shell=True, text=True)
    return output.strip().split('\n')

def write_json(file_path, json_data):
    with open(file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)

def write_to_file(file_path, text_to_write):
    try:
        with open(file_path, 'w') as file:
            file.write(text_to_write)
        print(f"Successfully wrote to {file_path}")
    except IOError as e:
        print(f"Error: {e}")

def append_to_file(file_path, text_to_append):
    try:
        with open(file_path, 'a') as file:
            file.write(text_to_append + '\n')
        # print(f"Appended '{text_to_append}' to {file_path}")
    except IOError as e:
        print(f"Error: {e}")

def get_json_data(file_path):
    with open(file_path, 'r') as json_file:
        return json.load(json_file)

def file_path_exists(file_path):
    return os.path.exists(file_path)

def get_list_of_dirs_in(current_dir):
    directories = [dir_path for dir_path in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, dir_path))]
    return directories

def get_files_inside_dir(dir_path):
    return os.listdir(dir_path)

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        print(f"already exist. dir: {dir_path}")

