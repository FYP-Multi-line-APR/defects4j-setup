import os
import json
import re
import sys
import math

from sctokenizer import JavaTokenizer

from utils import get_list_of_dirs_in, get_files_inside_dir, get_json_data, write_json
from utils import append_to_file, write_to_file
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from utils import extract_content_between
from utils import replace_prediction_token
from utils import bug_token, context_token, prediction_token
from utils import txt_field, bug_field
from utils import prediction_field, delete_add_field, from_prev_prediction_field, context_distance_field, prediction_length_field
from utils import context_field, distance_field
from utils import create_prediction_info, get_context_info

from generate_result_summary import get_file_to_write_path, generate_summary

fim_middle_token = "<fim_middle>"
prediction_end_token = "<|endoftext|>"

# def do_model_prediction(text):
#     global tokenizer, model
#     input_ids = tokenizer(text, return_tensors="pt").input_ids
#     generated_ids = model.generate(input_ids, max_length=predict_max_length)
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
    global tokenizer, model, add_delete, predict_max_lengths
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    predictions = []
    if add_delete == "yes":
        prediction_info = create_prediction_info()

        prediction_info[prediction_field] = "[Delete]"
        # prediction_info[prediction_length_field] = 
        prediction_info[delete_add_field] = True

        predictions.append(prediction_info)
    for predict_max_length in predict_max_lengths:
        generated_ids = model.generate(input_ids, max_length=predict_max_length, num_beams=beam_size, num_return_sequences=beam_size)
        for i, generated_id in enumerate(generated_ids):
            prediction_info = create_prediction_info()
            prediction = tokenizer.decode(generated_id, skip_special_tokens=True)

            prediction_info[prediction_field] = prediction 
            prediction_info[prediction_length_field] = predict_max_length

            predictions.append(prediction_info)
    return predictions 

def update_input_with_curr_prediction(input_text, curr_prediction):
    bug_part = extract_content_between(input_text, bug_token, context_token)
    new_context = replace_prediction_token(bug_part, curr_prediction)
    return f"{bug_token} {bug_part} {context_token} {new_context}"

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
    global java_tokenizer
    processed_original_fix = process_line_for_compare(original_fix)
    processed_prediction = process_line_for_compare(prediction)

    tokens_fix = java_tokenizer.tokenize(original_fix)
    tokens_prediction = java_tokenizer.tokenize(prediction)

    comparison_fix_tokens = [token.token_value for token in tokens_fix]
    comparison_prediction_tokens = [token.token_value for token in tokens_prediction]

    return comparison_fix_tokens == comparison_prediction_tokens

    # if processed_original_fix == processed_prediction:
    #     return True
    # else:
    #     # print(f"{processed_original_fix} != {processed_prediction}")
    #     return False

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

def count_prediction_token_in(text):
    occurrences_count = len(re.findall(prediction_token, text, re.IGNORECASE))
    return occurrences_count

def get_index_of_buggy_context(contexts):
    for i in range(len(contexts)):
        context = contexts[i]
        if count_prediction_token_in(context[txt_field]) == 1:
            return i 
    return -1 

def get_needed_contexts(contexts):
    global context_limit 
    buggy_context_index = get_index_of_buggy_context(contexts)
    needed_contexts = [get_context_info(contexts[buggy_context_index], 0)]
    for i in range(1, math.ceil(context_limit / 2)):
        above_surrounding_context_index = buggy_context_index - i 
        below_surrounding_context_index = buggy_context_index + i 
        if -1 < above_surrounding_context_index:
            # print(f"above context: {contexts[above_surrounding_context_index]}")
            needed_contexts.append(get_context_info(contexts[above_surrounding_context_index], -i))
        if below_surrounding_context_index < len(contexts):
            # print(f"below context: {contexts[below_surrounding_context_index]}")
            needed_contexts.append(get_context_info(contexts[below_surrounding_context_index], i))
    return needed_contexts

def format_to_prompt(bug_context_text, bug, current_context_text):
    if prediction_token in current_context_text:
        context_with_bug = replace_prediction_token(current_context_text, bug)
        return f"{bug_token} {bug_context_text} {context_token} {context_with_bug}"
    return f"{bug_token} {bug_context_text} {context_token} {current_context_text}"

# def single_iteration_beam_predict_for_bug_file(result, project, bug, bug_file_path):
#     global iterations, bug_count_to_check, number_of_improvements_after_first_iteration
#     set_prediction_files_for_iterative_beam_approach()
#     file_content = get_json_data(bug_file_path)
#     total_bug_count = len(file_content)
#     if total_bug_count != bug_count_to_check:
#         return
#     correct_predictions_with_context = []
#     fix_bug_count = 0 
#     take_first_prediction = False

#     try_fix_bug_contents = file_content
#     not_fix_bug_contents = []
#     fix_bug_contents = []

#     while try_fix_bug_contents:
#         # print(f"project: {project}, bug: {bug}")
#         # print(f"try_fix_bug_contents: {len(try_fix_bug_contents)}")
#         # print(f"fix_bug_contents: {len(fix_bug_contents)}")
#         # print(f"not_fix_bug_contents: {len(not_fix_bug_contents)}")
#         bug_content = try_fix_bug_contents.pop(0)
#         bug_contexts = bug_content['ctxs']
#         bug_text = bug_content[bug_field]
#         needed_contexts = get_needed_contexts(bug_contexts)
#         needed_bug_context_text = needed_contexts[0][context_field][txt_field]
#         bug_part = needed_bug_context_text
#         # bug_part = extract_content_between(needed_bug_context_text, bug_token, context_token)
#         if take_first_prediction:
#             first_predictions = [bug_content['fix']]
#             take_first_prediction = False
#         else:
#             first_predictions = []
#             # print(f"needed context length: {len(needed_contexts)}")
#             for context in needed_contexts:
#                 # formatted_context_text = format_to_prompt(needed_bug_context_text, bug_text, context[txt_field])
#                 # first_predictions.extend(do_model_beam_predictions(formatted_context_text))
#                 prediction_infos = do_model_beam_predictions(context[context_field][txt_field])
#                 for prediction_info in prediction_infos:
#                     prediction_info[context_distance_field]=context[distance_field] 
#                 first_predictions.extend(prediction_infos)
        
#         # take contexts with prediction to provide as input
#         for correct_prediction_with_context in correct_predictions_with_context:
#             new_input_text = f"{bug_token} {bug_part} {context_token} {correct_prediction_with_context}"
#             new_input_prediction_infos = do_model_beam_predictions(new_input_text)
#             for new_input_prediction_info in new_input_prediction_infos:
#                 new_input_prediction_info[from_prev_prediction_field]=True
#             first_predictions.extend(new_input_prediction_infos)

#         list_of_queues = initiate_list_of_queues_for_iterations()
#         original_fix = bug_content['fix']
#         list_of_queues[0] = first_predictions
#         is_fixed = False

#         curr_queue = list_of_queues[0]
#         if is_fixed:
#             break
#         while curr_queue:
#             curr_prediction = curr_queue.pop(0)
#             # log_iterative_prediction(project, bug, original_fix, curr_prediction, needed_bug_context_text, 0)
#             is_fixed = is_bug_fixed(original_fix, curr_prediction[prediction_field])
#             if is_fixed:
#                 fix_bug_count += 1
#                 correct_prediction_with_context = replace_prediction_token(bug_part, curr_prediction[prediction_field])
#                 correct_predictions_with_context.append(correct_prediction_with_context)

#                 technique_result_summary[fix_parts_count_field] += 1
#                 technique_result_summary[deleted_parts_count_field] += 1
#                 technique_result_summary[prev_prediction_used_count_field] += 1
#                 technique_result_summary[context_distance_count_field][curr_prediction[context_distance_field]] += 1
#                 technique_result_summary[prediction_length_field][curr_prediction[prediction_length_field]] += 1

#                 break
#         if is_fixed:
#             fix_bug_contents.append(bug_content)
#             try_fix_bug_contents.extend(not_fix_bug_contents)
#             not_fix_bug_contents = []
#         else:
#             not_fix_bug_contents.append(bug_content)    
#     result[project][bug] = f"{fix_bug_count} / {total_bug_count}"
        
def iterative_beam_predict_for_bug_file(result, project, bug, bug_file_path):
    global iterations, bug_count_to_check, number_of_improvements_after_first_iteration
    global allow_prediction_forward
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
        bug_content = try_fix_bug_contents.pop(0)
        bug_contexts = bug_content['ctxs']
        bug_text = bug_content[bug_field]
        needed_contexts = get_needed_contexts(bug_contexts)
        needed_bug_context_text = needed_contexts[0][context_field][txt_field]
        bug_part = needed_bug_context_text

        if take_first_prediction:
            first_predictions = [bug_content['fix']]
            take_first_prediction = False
        else:
            first_predictions = []
            for context in needed_contexts:
                formatted_context_text = format_to_prompt(needed_bug_context_text, bug_text, context[context_field][txt_field])
                # first_predictions.extend(do_model_beam_predictions(formatted_context_text))
                prediction_infos = do_model_beam_predictions(formatted_context_text)
                for prediction_info in prediction_infos:
                    prediction_info[context_distance_field] = context[distance_field] 
                first_predictions.extend(prediction_infos)
        # take contexts with prediction to provide as input
        
        if allow_prediction_forward == "yes":
            for correct_prediction_with_context in correct_predictions_with_context:
                new_input_text = f"{bug_token} {bug_part} {context_token} {correct_prediction_with_context}"              
                new_input_prediction_infos = do_model_beam_predictions(new_input_text)
                for new_input_prediction_info in new_input_prediction_infos:
                    new_input_prediction_info[from_prev_prediction_field]=True
                first_predictions.extend(new_input_prediction_infos)

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
                # log_iterative_prediction(project, bug, original_fix, curr_prediction, next_context, iteration)
                is_fixed = is_bug_fixed(original_fix, curr_prediction[prediction_field])
                if is_fixed:
                    fix_bug_count += 1
                    correct_prediction_with_context = replace_prediction_token(bug_part, curr_prediction[prediction_field])
                    correct_predictions_with_context.append(correct_prediction_with_context)

                    technique_result_summary[fix_parts_count_field] += 1
                    if curr_prediction[delete_add_field]:
                        technique_result_summary[deleted_parts_count_field] += 1 
                    if curr_prediction[from_prev_prediction_field]:
                        technique_result_summary[prev_prediction_used_count_field] += 1
                    technique_result_summary[context_distance_count_field][curr_prediction[context_distance_field]] += 1
                    if curr_prediction[prediction_length_field] > 0:
                        technique_result_summary[prediction_length_count_field][curr_prediction[prediction_length_field]] += 1
                    break
                else:
                    if iteration < iterations - 1:
                        new_input_context = update_input_with_curr_prediction(needed_bug_context_text, curr_prediction[prediction_field])
                        next_predictions = do_model_beam_predictions(new_input_context)
                        next_predictions = next_predictions[:number_of_improvements_after_first_iteration]
                        add_predictions_to_queue(list_of_queues[iteration + 1], next_predictions)
            if is_fixed:
                fix_bug_contents.append(bug_content)
                try_fix_bug_contents.extend(not_fix_bug_contents)
                not_fix_bug_contents = []
            else:
                not_fix_bug_contents.append(bug_content)     
    result[project][bug] = f"{fix_bug_count} / {total_bug_count}"

# def iterative_predict_for_bug_file(result, project, bug, bug_file_path):
#     set_prediction_files_for_iterative_approach()
#     file_content = get_json_data(bug_file_path)
#     total_bug_count = len(file_content)
#     fix_bug_count = 0
#     for bug_content in file_content:
#         bug_contexts = bug_content['ctxs']
#         next_context = bug_contexts[0]['txt']
#         for i in range(iterations):
#             prediction = do_model_prediction(next_context)
#             original_fix = bug_content['fix']
#             log_prediction(project, bug, original_fix, prediction, next_context)
#             is_fixed = is_bug_fixed(original_fix, prediction)
#             if is_fixed:
#                 fix_bug_count += 1
#                 break
#             else:
#                 next_context = replace_bug_context(next_context, prediction)
#     result[project][bug] = f"{fix_bug_count} / {total_bug_count}"

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

# def predict_for_bug_file(result, project, bug, bug_file_path):
#     set_prediction_files_for_normal_approach()
#     file_content = get_json_data(bug_file_path)
#     total_bug_count = len(file_content)
#     fix_bug_count = 0
#     prediction_close_count = 0
#     for bug_content in file_content:
#         bug_contexts = bug_content['ctxs']
#         selected_context = bug_contexts[0]['txt']
#         prediction = do_model_prediction(selected_context)
#         original_fix = bug_content['fix']
#         log_prediction(project, bug, original_fix, prediction, selected_context)
#         is_fixed = is_bug_fixed(original_fix, prediction)
#         is_close = is_prediction_close(original_fix, prediction)
#         if is_fixed:
#             fix_bug_count += 1
#             prediction_close_count += 1 
#         elif is_close:
#             prediction_close_count += 1
#     result[project][bug] = f"{fix_bug_count} / {prediction_close_count} / {total_bug_count}"

def set_prediction_files_for_iterative_beam_approach():
    global version, beam_size, iterations, model_name, bug_count_to_check, number_of_improvements_after_first_iteration, add_delete, allow_prediction_forward
    global context_limit, predict_max_lengths
    global prediction_file_path, prediction_log_file_path , summary_file_path, technique_summary_file_path
    predict_max_lengths_str = "-".join([str(length) for length in predict_max_lengths])
    prediction_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}-iter-{iterations}-improv-{number_of_improvements_after_first_iteration}-checked-bug-count-{bug_count_to_check}-context-limit-{context_limit}-max-length-{predict_max_lengths_str}-delete-{add_delete}-semantic-{allow_prediction_forward}.json"
    prediction_log_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}-iter-{iterations}-improv-{number_of_improvements_after_first_iteration}-checked-bug-count-{bug_count_to_check}-context-limit-{context_limit}-max-length-{predict_max_lengths_str}-delete-{add_delete}-semantic-{allow_prediction_forward}.txt"
    summary_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}-iter-{iterations}-improv-{number_of_improvements_after_first_iteration}-checked-bug-count-{bug_count_to_check}-context-limit-{context_limit}-max-length-{predict_max_lengths_str}-delete-{add_delete}-semantic-{allow_prediction_forward}-summary.txt"
    technique_summary_file_path = f"./results/prediction-{model_name}-{version}-chathuranga-beam-{beam_size}-iter-{iterations}-improv-{number_of_improvements_after_first_iteration}-checked-bug-count-{bug_count_to_check}-context-limit-{context_limit}-max-length-{predict_max_lengths_str}-delete-{add_delete}-semantic-{allow_prediction_forward}-technique-summary.json"

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

predict_max_lengths = None


cloned_projects_dir_path = "./cloned-projects"

# train_data_dir_path = "./fine-tune-train-data/set12-context-all-prompt-3"
train_data_dir_path = "./train-data/set16-context-all-width-5"
# train_data_dir_path = "./fine-tune-train-data/set10-context-10-prompt-3"
# train_data_dir_path = "./train-data/set8-prediction-token-context-5-prompt-1"
categorized_bugs_dir_path = "./categorize-bugs"

# context limit should vary in odd numbers 
context_limit = None

fix_parts_count_field = "fix_parts_count"
deleted_parts_count_field = "deleted_parts_count"
prev_prediction_used_count_field = "prev_prediction_used_count"
context_distance_count_field = "context_distance_count"
prediction_length_count_field = "prediction_length_count"

json_ext = ".json"

prediction_file_path = None
prediction_log_file_path = None
summary_file_path = None
technique_summary_file_path = None

add_delete = "no"
allow_prediction_forward = "no"

pid = None

defects4j_v2_bugs_path = "defects4j-v2-bugs.json"

java_tokenizer = JavaTokenizer()

if __name__ == "__main__":
    pid = sys.argv[1]
    beam_size = int(sys.argv[2])
    iterations = int(sys.argv[3])
    number_of_improvements_after_first_iteration = int(sys.argv[4])
    bug_count_to_check = int(sys.argv[5])
    add_delete = sys.argv[6]
    allow_prediction_forward = sys.argv[7]
    context_limit = int(sys.argv[8])
    predict_max_lengths = [int(length) for length in sys.argv[9:]]

    print(f"running for bugs in {pid} \nbeam size: {beam_size} \niterations: {iterations} \nimprovements after 1st iteration: {number_of_improvements_after_first_iteration} \nbug count checking: {bug_count_to_check}")
    print(f"context limit: {context_limit}\npredict max lengths: {predict_max_lengths}")
    # bug_projects = get_list_of_dirs_in(train_data_dir_path)

    technique_result_summary = {
        fix_parts_count_field: 0,
        deleted_parts_count_field: 0,
        prev_prediction_used_count_field: 0,
        context_distance_count_field: {distance: 0 for distance in list(range(-context_limit//2, context_limit//2 + 1))},
        prediction_length_count_field: {predict_length: 0 for predict_length in predict_max_lengths}
    }

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
                # single_iteration_beam_predict_for_bug_file(result, project, bug, file_path)
                iterative_beam_predict_for_bug_file(result, project, bug, file_path)
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
            
    print(result)
    write_json(prediction_file_path, result)
    summary = generate_summary(result)
    write_to_file(summary_file_path, summary)
    write_json(technique_summary_file_path, technique_result_summary)
   
    