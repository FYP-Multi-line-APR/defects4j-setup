import os
import json
import re
import sys

from utils import get_list_of_dirs_in, get_files_inside_dir, get_json_data, write_json
from utils import append_to_file, write_to_file
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from utils import extract_content_between, bug_token, context_token
from utils import replace_prediction_token

from generate_result_summary import get_file_to_write_path, generate_summary

fim_middle_token = "<fim_middle>"
prediction_end_token = "<|endoftext|>"

def do_model_prediction(text):
    global tokenizer, model
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=predict_max_length)
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return prediction

# def do_model_beam_prediction(text):
#     global tokenizer, model 
#     input_ids = tokenizer(text, return_tensors="pt").input_ids
#     generated_ids = model.generate(input_ids, max_length=predict_max_length, num_beams=beam_size, num_return_sequences=beam_size)
#     prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#     return prediction 

def get_prediction_only_from_starcoder_output(output):
    start_index = output.find(fim_middle_token)
    end_index = output.find(prediction_end_token)
    if start_index != -1 and end_index != -1:
        prediction_only = output[start_index + len(fim_middle_token) : end_index].strip()
        return prediction_only
    elif start_index != -1:
        prediction_only = output[start_index + len(fim_middle_token) :].strip()
        return prediction_only
    return output

def do_model_beam_predictions(text):
    global tokenizer, model, add_delete
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=predict_max_length, num_beams=beam_size, num_return_sequences=beam_size)
    predictions = []
    if add_delete == "yes":
        predictions.append("[Delete]")
    for i, generated_id in enumerate(generated_ids):
        prediction = tokenizer.decode(generated_id, skip_special_tokens=True)
        prediction = get_prediction_only_from_starcoder_output(prediction)
        predictions.append(prediction)
    return predictions 

def replace_bug_context(original_text, replacement_text):
    start_index = original_text.find("[BUG]")
    end_index = original_text.find("[CONTEXT]")
    if start_index != -1 and end_index != -1:
        modified_text = original_text[:start_index + 5] + replacement_text + original_text[end_index:]
        return modified_text
    else:
        print("Error: [BUG] and/or [CONTEXT] markers not found.")
        return original_text

def log_prediction(project, bug, original_fix, prediction, prompt):
    log_str = f"{project} {bug}    PREDICTION: {prediction}    FIX: {original_fix}    PROMPT: {prompt}"
    append_to_file(prediction_log_file_path, log_str)

def log_iterative_prediction(project, bug, original_fix, prediction, prompt, iteration):
    log_str = f"{project} {bug}    ITERATION: {iteration}    PREDICTION: {prediction}    FIX: {original_fix}    PROMPT: {prompt}"
    append_to_file(prediction_log_file_path, log_str)

def process_line_for_compare(code_str):
    code_str = code_str.strip()
    code_str = re.sub(r'\s+', '', code_str)
    code_str = code_str.replace('\n', '')
    paranthesis = ['(',')','[',']','{','}']
    for p in paranthesis:
        code_str = code_str.replace(p+' ', p)
        code_str = code_str.replace(' '+p, p)
    return code_str

def is_bug_fixed(original_fix, prediction):
    processed_original_fix = process_line_for_compare(original_fix)
    processed_prediction = process_line_for_compare(prediction)
    if processed_original_fix == processed_prediction:
        return True
    else:
        # print(f"{processed_original_fix} != {processed_prediction}")
        return False

def is_prediction_close(original_fix, prediction):
    processed_original_fix = process_line_for_compare(original_fix)
    processed_prediction = process_line_for_compare(prediction)
    if processed_original_fix in processed_prediction or processed_prediction in processed_original_fix:
    # if processed_original_fix in processed_prediction:
        return True
    return False

def initiate_list_of_queues_for_iterations():
    global iterations 
    result = []
    for i in range(iterations):
        result.append([])
    return result

def add_predictions_to_queue(queue, predictions):
    for prediction in predictions:
        queue.append(prediction)

def single_iteration_beam_predict_for_bug_file(result, project, bug, bug_file_path):
    global iterations, bug_count_to_check, number_of_improvements_after_first_iteration
    set_prediction_files_for_iterative_beam_approach()
    file_content = get_json_data(bug_file_path)
    total_bug_count = len(file_content)
    if total_bug_count != bug_count_to_check:
        return
    correct_predictions_with_context = []
    fix_bug_count = 0 
    take_first_prediction = False

    try_fix_bug_contents = file_content
    not_fix_bug_contents = []
    fix_bug_contents = []

    while try_fix_bug_contents:
        print(f"project: {project}, bug: {bug}")
        print(f"try_fix_bug_contents: {len(try_fix_bug_contents)}")
        print(f"fix_bug_contents: {len(fix_bug_contents)}")
        print(f"not_fix_bug_contents: {len(not_fix_bug_contents)}")
        bug_content = try_fix_bug_contents.pop(0)
        bug_contexts = bug_content['ctxs']
        next_context = bug_contexts[0]['txt']
        bug_part = extract_content_between(next_context, bug_token, context_token)
        if take_first_prediction:
            first_predictions = [bug_content['fix']]
            take_first_prediction = False
        else:
            first_predictions = do_model_beam_predictions(next_context)
        # take contexts with prediction to provide as input
        for correct_prediction_with_context in correct_predictions_with_context:
            new_input_text = f"{bug_token} {bug_part} {context_token} {correct_prediction_with_context}"
            new_input_predictions = do_model_beam_predictions(new_input_text)
            first_predictions.extend(new_input_predictions)

        list_of_queues = initiate_list_of_queues_for_iterations()
        original_fix = bug_content['fix']
        list_of_queues[0] = first_predictions
        is_fixed = False

        curr_queue = list_of_queues[0]
        if is_fixed:
            break
        while curr_queue:
            curr_prediction = curr_queue.pop(0)
            log_iterative_prediction(project, bug, original_fix, curr_prediction, next_context, 0)
            is_fixed = is_bug_fixed(original_fix, curr_prediction)
            if is_fixed:
                fix_bug_count += 1
                correct_prediction_with_context = replace_prediction_token(bug_part, curr_prediction)
                correct_predictions_with_context.append(correct_prediction_with_context)
                break
        if is_fixed:
            fix_bug_contents.append(bug_content)
            try_fix_bug_contents.extend(not_fix_bug_contents)
            not_fix_bug_contents = []
        else:
            not_fix_bug_contents.append(bug_content)    
    result[project][bug] = f"{fix_bug_count} / {total_bug_count}"
        



def iterative_beam_predict_for_bug_file(result, project, bug, bug_file_path):
    global iterations, bug_count_to_check, number_of_improvements_after_first_iteration
    set_prediction_files_for_iterative_beam_approach()
    file_content = get_json_data(bug_file_path)
    total_bug_count = len(file_content)
    if total_bug_count != bug_count_to_check:
        return
    correct_predictions_with_context = []
    fix_bug_count = 0 
    take_first_prediction = False
    for bug_content in file_content: 
        bug_contexts = bug_content['ctxs']
        next_context = bug_contexts[0]['txt']
        bug_part = extract_content_between(next_context, bug_token, context_token)
        if take_first_prediction:
            first_predictions = [bug_content['fix']]
            take_first_prediction = False
        else:
            first_predictions = do_model_beam_predictions(next_context)
        # take contexts with prediction to provide as input
        for correct_prediction_with_context in correct_predictions_with_context:
            new_input_text = f"{bug_token} {bug_part} {context_token} {correct_prediction_with_context}"
            new_input_predictions = do_model_beam_predictions(new_input_text)
            first_predictions.extend(new_input_predictions)

        list_of_queues = initiate_list_of_queues_for_iterations()
        original_fix = bug_content['fix']
        list_of_queues[0] = first_predictions
        is_fixed = False
        for iteration in range(iterations):
            curr_queue = list_of_queues[iteration]
            if is_fixed:
                break
            while curr_queue:
                curr_prediction = curr_queue.pop(0)
                log_iterative_prediction(project, bug, original_fix, curr_prediction, next_context, iteration)
                is_fixed = is_bug_fixed(original_fix, curr_prediction)
                if is_fixed:
                    fix_bug_count += 1
                    correct_prediction_with_context = replace_prediction_token(bug_part, curr_prediction)
                    correct_predictions_with_context.append(correct_prediction_with_context)
                    break
                else:
                    if iteration < iterations - 1:
                        next_context = replace_bug_context(next_context, curr_prediction)
                        next_predictions = do_model_beam_predictions(next_context)
                        next_predictions = next_predictions[:number_of_improvements_after_first_iteration]
                        add_predictions_to_queue(list_of_queues[iteration + 1], next_predictions)
    result[project][bug] = f"{fix_bug_count} / {total_bug_count}"

def iterative_predict_for_bug_file(result, project, bug, bug_file_path):
    set_prediction_files_for_iterative_approach()
    file_content = get_json_data(bug_file_path)
    total_bug_count = len(file_content)
    fix_bug_count = 0
    for bug_content in file_content:
        bug_contexts = bug_content['ctxs']
        next_context = bug_contexts[0]['txt']
        for i in range(iterations):
            prediction = do_model_prediction(next_context)
            original_fix = bug_content['fix']
            log_prediction(project, bug, original_fix, prediction, next_context)
            is_fixed = is_bug_fixed(original_fix, prediction)
            if is_fixed:
                fix_bug_count += 1
                break
            else:
                next_context = replace_bug_context(next_context, prediction)
    result[project][bug] = f"{fix_bug_count} / {total_bug_count}"

def beam_predict_for_bug_file(result, project, bug, bug_file_path):
    set_prediction_files_for_multile_prediction_approach()
    file_content = get_json_data(bug_file_path)
    total_bug_count = len(file_content)
    fix_bug_count = 0
    for bug_content in file_content:
        bug_contexts = bug_content['ctxs']
        selected_context = bug_contexts[0]['txt']
        original_fix = bug_content['fix']
        predictions = do_model_beam_predictions(selected_context)
        predictions_set = set()
        for prediction in predictions:
            processed_prediction = process_line_for_compare(prediction)
            if processed_prediction not in predictions_set:
                predictions_set.add(processed_prediction)
                log_prediction(project, bug, original_fix, prediction, selected_context)
                is_fixed = is_bug_fixed(original_fix, prediction)
                if is_fixed:
                    fix_bug_count += 1
                    break 
    result[project][bug] = f"{fix_bug_count} / {total_bug_count}"

def predict_for_bug_file(result, project, bug, bug_file_path):
    set_prediction_files_for_normal_approach()
    file_content = get_json_data(bug_file_path)
    total_bug_count = len(file_content)
    fix_bug_count = 0
    prediction_close_count = 0
    for bug_content in file_content:
        bug_contexts = bug_content['ctxs']
        selected_context = bug_contexts[0]['txt']
        prediction = do_model_prediction(selected_context)
        original_fix = bug_content['fix']
        log_prediction(project, bug, original_fix, prediction, selected_context)
        is_fixed = is_bug_fixed(original_fix, prediction)
        is_close = is_prediction_close(original_fix, prediction)
        if is_fixed:
            fix_bug_count += 1
            prediction_close_count += 1 
        elif is_close:
            prediction_close_count += 1
    result[project][bug] = f"{fix_bug_count} / {prediction_close_count} / {total_bug_count}"

def set_prediction_files_for_iterative_beam_approach():
    global version, beam_size, iterations, model_name, bug_count_to_check, number_of_improvements_after_first_iteration, add_delete
    global prediction_file_path, prediction_log_file_path , summary_file_path
    prediction_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}-iter-{iterations}-improv-{number_of_improvements_after_first_iteration}-checked-bug-count-{bug_count_to_check}.json"
    prediction_log_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}-iter-{iterations}-improv-{number_of_improvements_after_first_iteration}-checked-bug-count-{bug_count_to_check}.txt"
    summary_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}-iter-{iterations}-improv-{number_of_improvements_after_first_iteration}-checked-bug-count-{bug_count_to_check}-summary.txt"

def set_prediction_files_for_iterative_approach():
    global version, beam_size, iterations, model_name, bug_count_to_check
    global prediction_file_path, prediction_log_file_path, summary_file_path
    prediction_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}-iter-{iterations}.json"
    prediction_log_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}-iter-{iterations}.txt"
    summary_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}-iter-{iterations}-summary.txt"

def set_prediction_files_for_multile_prediction_approach():
    global version, beam_size, model_name, pid
    global prediction_file_path, prediction_log_file_path, summary_file_path
    prediction_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}-multiple-predictions-{pid}.json"
    prediction_log_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}-multiple-predictions-{pid}.txt"
    summary_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}-multiple-predictions-{pid}-summary.txt"

def set_prediction_files_for_normal_approach():
    global version, beam_size, model_name
    global prediction_file_path, prediction_log_file_path, summary_file_path
    prediction_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}.json"
    prediction_log_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}.txt"
    summary_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}-summary.txt"

version = "v31"
# model_name = "codet5p-220m"
model_name = "codet5-small"
tokenizer = RobertaTokenizer.from_pretrained(f'chathuranga-jayanath/{model_name}-' + version)
model = T5ForConditionalGeneration.from_pretrained(f'chathuranga-jayanath/{model_name}-' + version)

# tokenizer = RobertaTokenizer.from_pretrained(f'Salesforce/{model_name}')
# model = T5ForConditionalGeneration.from_pretrained(f'Salesforce/{model_name}')


# beam_size = 2
# iterations = 1
# number_of_improvements_after_first_iteration = 1
# bug_count_to_check = 1
predict_max_length = 20

cloned_projects_dir_path = "./cloned-projects"
train_data_dir_path = "./fine-tune-train-data/set8-context-5-prompt-3"
# train_data_dir_path = "./train-data/set8-prediction-token-context-5-prompt-1"
categorized_bugs_dir_path = "./categorize-bugs"

json_ext = ".json"

prediction_file_path = None
prediction_log_file_path = None
summary_file_path = None

add_delete = "no"

pid = None

defects4j_v2_bugs_path = "defects4j-v2-bugs.json"

if __name__ == "__main__":
    pid = sys.argv[1]
    beam_size = int(sys.argv[2])
    iterations = int(sys.argv[3])
    number_of_improvements_after_first_iteration = int(sys.argv[4])
    bug_count_to_check = int(sys.argv[5])
    add_delete = sys.argv[6]


    print(f"running for bugs in {pid} \nbeam size: {beam_size} \niterations: {iterations} \nimprovements after 1st iteration: {number_of_improvements_after_first_iteration} \nbug count checking: {bug_count_to_check}")
    # bug_projects = get_list_of_dirs_in(train_data_dir_path)

    categorized_bugs_file_path = os.path.join(categorized_bugs_dir_path, str(bug_count_to_check) + json_ext)
    categorized_bugs_file_data = get_json_data(categorized_bugs_file_path)
    bug_projects = list(categorized_bugs_file_data.keys())

    # defects4j_v2_bugs_data = get_json_data(defects4j_v2_bugs_path)
    # bug_projects = list(defects4j_v2_bugs_data.keys())

    result = {}
    if pid == "all":
        for project in bug_projects:
            # project_path = os.path.join(train_data_dir_path, project)
            # bug_files = get_files_inside_dir(project_path)

            bug_files = categorized_bugs_file_data[project]
            # bug_files = defects4j_v2_bugs_data[project]
            project_path = os.path.join(train_data_dir_path, project)

            result[project] = {}
            for file in bug_files:
                if not file.endswith(".json"):
                    file = file + ".json"
                file_path = os.path.join(project_path, file)
                bug = file.split('.')[0]
                single_iteration_beam_predict_for_bug_file(result, project, bug, file_path)
                # iterative_beam_predict_for_bug_file(result, project, bug, file_path)
                # iterative_predict_for_bug_file(result, project, bug, file_path)
                # predict_for_bug_file(result, project, bug, file_path)
                # beam_predict_for_bug_file(result, project, bug, file_path)
    elif pid is not None:
        project_path = os.path.join(train_data_dir_path, pid)
        bug_files = get_files_inside_dir(project_path)
        result[pid] = {}
        for file in bug_files:
            file_path = os.path.join(project_path, file)
            bug = file.split('.')[0]
            iterative_beam_predict_for_bug_file(result, pid, bug, file_path)
            # iterative_predict_for_bug_file(result, pid, bug, file_path)
            # predict_for_bug_file(result, pid, bug, file_path)
            # beam_predict_for_bug_file(result, pid, bug, file_path)
    print(result)
    write_json(prediction_file_path, result)
    summary = generate_summary(result)
    write_to_file(summary_file_path, summary)
   
