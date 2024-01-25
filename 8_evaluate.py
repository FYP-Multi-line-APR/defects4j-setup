import os
import json
import re
import sys

from utils import get_list_of_dirs_in, get_files_inside_dir, get_json_data, write_json
from utils import append_to_file
from transformers import RobertaTokenizer, T5ForConditionalGeneration

def do_model_prediction(text):
    global tokenizer, model
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=20)
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return prediction

def do_model_beam_predictions(text):
    global tokenizer, model 
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    generate_ids = model.generate(input_ids, max_length=20)
    

def log_prediction(project, bug, original_fix, prediction):
    log_str = f"{project} {bug} FIX: {original_fix} PREDICTION: {prediction}"
    append_to_file(prediction_log_file_path, log_str)

def process_line_for_compare(code_str):
    code_str = code_str.strip()
    code_str = re.sub(r'\s+', ' ', code_str)
    code_str = code_str.replace('\n', '')
    return code_str

def is_bug_fixed(original_fix, prediction):
    processed_original_fix = process_line_for_compare(original_fix)
    processed_prediction = process_line_for_compare(prediction)
    if processed_original_fix == processed_prediction:
        return True
    else:
        print(f"{processed_original_fix} != {processed_prediction}")
        return False

def is_prediction_close(original_fix, prediction):
    processed_original_fix = process_line_for_compare(original_fix)
    processed_prediction = process_line_for_compare(prediction)
    if processed_original_fix in processed_prediction or processed_prediction in processed_original_fix:
    # if processed_original_fix in processed_prediction:
        return True
    return False

def predict_for_bug_file(result, project, bug, bug_file_path):
    file_content = get_json_data(bug_file_path)
    total_bug_count = len(file_content)
    fix_bug_count = 0
    prediction_close_count = 0
    for bug_content in file_content:
        bug_contexts = bug_content['ctxs']
        selected_context = bug_contexts[0]['txt']
        prediction = do_model_prediction(selected_context)
        original_fix = bug_content['fix']
        log_prediction(project, bug, original_fix, prediction)
        is_fixed = is_bug_fixed(original_fix, prediction)
        is_close = is_prediction_close(original_fix, prediction)
        if is_fixed:
            fix_bug_count += 1
            prediction_close_count += 1 
        elif is_close:
            prediction_close_count += 1

    result[project][bug] = f"{fix_bug_count} / {prediction_close_count} / {total_bug_count}"

def get_select_model():
    print("Available models. Select one.")
    for i in range(len(models)): 
        print(f"{i+1}. {models[i]}")
    choice = input("Enter choice: ")
    return models[choice-1]


models = ['Salesforce/codet5p-220m', 'ayeshgk/codet5-small-ft-v8-cpatd-ft-v8-cpat_dv5']

tokenizer = RobertaTokenizer.from_pretrained('ayeshgk/codet5-small-ft-v8-cpatd-ft-v8-cpat_dv5')
model = T5ForConditionalGeneration.from_pretrained('ayeshgk/codet5-small-ft-v8-cpatd-ft-v8-cpat_dv5')

# tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5p-220m')
# model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5p-220m')

cloned_projects_dir_path = "./cloned-projects"
train_data_dir_path = "./fine-tune-train-data/set5-prediction-token-context-5"
prediction_file_path = "./results/prediction-token-added-context-5-only-fix-in.json"
prediction_log_file_path = "./results/prediction.txt"

if __name__ == "__main__":


    pid = None 
    bid = None 
    try:
        pid = sys.argv[1]
        bid = sys.argv[2]
    except:
        pass 
    bug_projects = get_list_of_dirs_in(train_data_dir_path)
    result = {}
    if pid is None and bid is None:
        for project in bug_projects:
            project_path = os.path.join(train_data_dir_path, project)
            bug_files = get_files_inside_dir(project_path)
            result[project] = {}
            for file in bug_files:
                file_path = os.path.join(project_path, file)
                bug = file.split('.')[0]
                predict_for_bug_file(result, project, bug, file_path)
    elif pid is not None and bid is None:
        project_path = os.path.join(train_data_dir_path, pid)
        bug_files = get_files_inside_dir(project_path)
        result[pid] = {}
        for file in bug_files:
            file_path = os.path.join(project_path, file)
            bug = file.split('.')[0]
            predict_for_bug_file(result, pid, bug, file_path)
    print(result)
    write_json(prediction_file_path, result)
