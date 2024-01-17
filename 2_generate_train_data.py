import os 
import subprocess
import re
from utils import make_dir, write_json, get_pids_list, get_bids_list, file_path_exists

cloned_projects_dir_path = './cloned-projects'
train_data_dir_path = "./train-data"

file_git_diff = "diff --git "
java_file_ext = ".java"

generated_train_data_id = 1

context_width = 5

def get_git_show_result():
    return subprocess.check_output("git show", shell=True, text=True)

def get_git_show_result_lines(file_path):
    print(f"file path: {file_path}")
    os.chdir(file_path)
    result = subprocess.check_output("git show", shell=True)
    result = result.decode('utf-8', errors='replace').splitlines()
    os.chdir('./../../../')
    return result

def extract_file_path_from(diff_line):
    pattern = r'diff --git a/(.*?) b/\1'
    match = re.search(pattern, diff_line)
    if match:
        return match.group(1)
    return None

def extract_starting_line_number(line):
    pattern = r'@@ -\d+,\d+ \+(\d+),\d+ @@'
    match = re.match(pattern, line)
    if match:
        return int(match.group(1))
    return None

def collect_context(bug_id, bug_idx, file_path, lines_range):
    # print(f"bug id: {bug_id}, bug idx: {bug_idx}")
    os.chdir(f"{cloned_projects_dir_path}/{bug_id}/{bug_idx}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file: 
            lines = file.readlines()
        selected_lines = lines[lines_range[0]-1 : lines_range[1]-1]
        selected_lines = [process_line(line) for line in selected_lines]
        return " ".join(selected_lines)
    except Exception as ex:
        print(ex)
    finally:
        os.chdir(f"./../../../")

def generate_context(context_front, context_end, bug):
    return context_front + ' <BUG> ' + bug + ' </BUG> ' + context_end

def generate_train_data_dict(id, bug, fix, context, file_path, buggy_line_no):
    return {
        "id": id,
        "file_path": file_path,
        "buggy_line_no": buggy_line_no,
        "bug": bug,
        "fix": fix,
        "fixes": [],
        "err": "",
        "ctxs": [
            {
                "txt": context
            }
        ]
    }

def process_line(line):
    line = line.lstrip('-')
    line = line.lstrip('+')
    line = line.rstrip('\n')
    line = line.strip()
    return line

def generate_train_data_for_chunk(bug_id, bug_idx, bug_chunk_start_line_no, bug_chunk_lines, file_path, buggy_end_line):
    buggy_line_no = bug_chunk_start_line_no 
    bug_lines = []
    fix_lines = []
    context_front = [buggy_line_no - context_width, buggy_line_no]
    context_end = [buggy_line_no, buggy_line_no + context_width]
    
    shift_context_end = 0
    for line in bug_chunk_lines:
        if line.startswith("- "):
            line = process_line(line)
            fix_lines.append(line)
        elif line.startswith("+ "):
            line = process_line(line)
            bug_lines.append(line)
            shift_context_end += 1
    context_end = [number + shift_context_end for number in context_end]

    if len(fix_lines) == 0:
        fix_lines.append('[DELETED]')

    bug = " ".join(bug_lines)
    fix = " ".join(fix_lines)
    collected_context_front = collect_context(bug_id, bug_idx, file_path, context_front)
    collected_context_end = collect_context(bug_id, bug_idx, file_path, context_end)
    context = generate_context(collected_context_front, collected_context_end, bug)
    
    global generated_train_data_id
    result = generate_train_data_dict(generated_train_data_id, bug, fix, context, file_path, (buggy_line_no,buggy_end_line))
    generated_train_data_id += 1
    return result

def generate_train_data(bug_id, bug_idx):
    buggy_dir = f"{cloned_projects_dir_path}/{bug_id}/{bug_idx}"
    git_show_result_lines = get_git_show_result_lines(buggy_dir)
    
    train_data = []

    file_path = ""
    bug_chunk_lines = []
    bug_chunk_start_line = -1
    curr_line_number = -1

    for line_idx in range(len(git_show_result_lines)):
        # print(curr_line_number)
        line = git_show_result_lines[line_idx]
        if line.startswith("diff --git ") and line.endswith(".java"): 
            file_path = extract_file_path_from(line)
        elif line.startswith("@@ "):    
            
            if not extract_starting_line_number(line):
                print(line,curr_line_number)
                continue
            curr_line_number = extract_starting_line_number(line)
            
        elif line.startswith("- "):
            bug_chunk_lines.append(line)
            bug_chunk_start_line = curr_line_number if bug_chunk_start_line == -1 else bug_chunk_start_line
        elif line.startswith("+ "):
            bug_chunk_lines.append(line)
            bug_chunk_start_line = curr_line_number if bug_chunk_start_line == -1 else bug_chunk_start_line
            curr_line_number += 1
        else:
            modified_file_path = f"{buggy_dir}/{file_path}"
            is_modified_file_exist = file_path_exists(modified_file_path)
            if file_path != "" and is_modified_file_exist and bug_chunk_lines != [] and bug_chunk_start_line != -1:
                generated_train_data = generate_train_data_for_chunk(bug_id, bug_idx, bug_chunk_start_line, bug_chunk_lines, file_path,curr_line_number)
                train_data.append(generated_train_data)
            curr_line_number += 1
            bug_chunk_lines = []
            bug_chunk_start_line = -1
    return train_data

def make_train_data_dir(bug_id):
    bug_id_dir = f"{train_data_dir_path}/{bug_id}"
    make_dir(bug_id_dir)

def write_train_data(bug_id, bug_idx, train_data):
    write_file_path = f"{train_data_dir_path}/{bug_id}/{bug_idx}.json"
    write_json(write_file_path, train_data)

if __name__ == "__main__":
    pids = get_pids_list()
    for pid in pids:
        bids = get_bids_list(pid)
        for bid in bids:
            train_data = generate_train_data(pid, bid)
            make_train_data_dir(pid)
            write_train_data(pid, bid, train_data)


