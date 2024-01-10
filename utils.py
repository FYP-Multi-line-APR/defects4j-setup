import subprocess
import json
import os

# defects4j = "~/Work/defects4j/framework/bin/defects4j"
fine_tune_train_data_dir = './fine-tune-train-data'
train_data_dir = './train-data'
cloned_dir = './cloned-projects'

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
    get_pids_command = f"defects4j pids"
    output = subprocess.check_output(get_pids_command, shell=True, text=True)
    return output.strip().split('\n')

def get_bids_list(pid):
    get_bids_command = f"defects4j bids -p {pid}"
    output = subprocess.check_output(get_bids_command, shell=True, text=True)
    return output.strip().split('\n')

def write_json(file_path, json_data):
    with open(file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)

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

