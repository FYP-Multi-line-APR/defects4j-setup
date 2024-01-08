# must execute 2_generate_train_data.py script before executing this

import re
import os
from utils import make_dir, get_list_of_dirs_in, get_files_inside_dir, get_json_data, write_json

train_data_dir_path = "./train-data"
fine_tune_train_data_dir_path = "./fine-tune-train-data"


def replace_bug_text(original_text, replacement_text):
    pattern = re.compile(r'<BUG>.*?</BUG>', re.DOTALL)
    try:
        modified_text = re.sub(pattern, replacement_text, original_text)
        return modified_text
    except Exception as ex:
        print(f"original text: {original_text}")
        print(ex)
        return None

def generate_fine_tune_data_point(json_data):
    id = json_data['id']
    buggy = json_data['ctxs'][0]['txt']
    fixed = replace_bug_text(buggy, json_data['fix'])

    if fixed is not None:
        return {
            "id": id,
            "buggy": buggy,
            "fixed": fixed
        }
    else:
        return None

def handle_project_dirs(project_dirs):
    for project_dir in project_dirs:
        make_dir(f"{fine_tune_train_data_dir_path}/{project_dir}")
        files = get_files_inside_dir(os.path.join(train_data_dir_path, project_dir))
        for file in files:
            fine_tuned_data = []
            file_json_data = get_json_data(f'{train_data_dir_path}/{project_dir}/{file}')
            for data_element in file_json_data:
                fine_tuned_data_point = generate_fine_tune_data_point(data_element)
                if fine_tuned_data_point is not None:
                    fine_tuned_data.append(fine_tuned_data_point)
            write_json(f"{fine_tune_train_data_dir_path}/{project_dir}/{file}", fine_tuned_data)

if __name__ == "__main__":
    make_dir(fine_tune_train_data_dir_path)
    project_dirs = get_list_of_dirs_in(train_data_dir_path)
    handle_project_dirs(project_dirs)

    
    # sample_file_path = f"{train_data_dir_path}/Chart/2.json"
    # json_data = get_json_data(sample_file_path)
    # print(len(json_data))

    # # select one 
    # print(json_data[0])
    

