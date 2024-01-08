import subprocess
import os 
import re
from utils import get_pids_list, get_bids_list, write_json, make_dir

defects4j_path = "~/Work/defects4j/framework/bin/defects4j"
cloned_projects_dir_path = './cloned-projects'

def get_multi_line_chunk_change_count_from_git_show_result(text):
    pattern = re.compile(r'(?:(?:^\s*[+-][^\+\-]*$\n)+)', re.MULTILINE)
    matches = pattern.findall(text)
    return len(matches)

def get_multi_line_chunk_count(pid, bid):
    bug_proj_path = f"{cloned_projects_dir_path}/{pid}/{bid}"
    os.chdir(bug_proj_path)
    multi_line_chunk_count = 0
    try: 
        output = subprocess.check_output("git show", shell=True, text=True)
        multi_line_chunk_count = get_multi_line_chunk_change_count_from_git_show_result(output) - 1
    except Exception as ex:
        print(f"error while checking pid:{pid}    bid:{bid}")
        print(ex)
    finally:
        os.chdir('./../../../')
    return multi_line_chunk_count

def generate_dict_with_empty_array_for_each(pids):
    result = {}
    for pid in pids:
        result[pid] = []
    return result

def categorize_bugs():
    result = {}
    bugs_by_count = {}
    pids_list = get_pids_list()
    for pid in pids_list:
        result[pid] = []
        bids_list = get_bids_list(pid)
        for bid in bids_list:
            multi_line_chunk_count = get_multi_line_chunk_count(pid, bid)
            result[pid].append((bid, multi_line_chunk_count))
            if bugs_by_count.get(multi_line_chunk_count) is not None:
                bugs_by_count[multi_line_chunk_count][pid].append(bid)
            else:
                bugs_by_count[multi_line_chunk_count] = generate_dict_with_empty_array_for_each(pids_list)
                bugs_by_count[multi_line_chunk_count][pid].append(bid)

    write_json('./categorize-bugs/bugs_by_count.json', bugs_by_count)
    return result 


def find_multi_line_bugs():
    result = {}
    pids_list = get_pids_list()
    for pid in pids_list:
        result[pid] = []
        bids_list = get_bids_list(pid)
        for bid in bids_list:
            multi_line_chunk_count = get_multi_line_chunk_count(pid, bid)
            
            if (multi_line_chunk_count > 1):
                result[pid].append((bid, multi_line_chunk_count))
    return result 

if __name__ == "__main__":
    categorize_bugs_dir = './categorize-bugs'
    bugs_chunk_count_path = '/bugs-chunk-count.json'
    # result_file_path = "multi-line-bugs.json"
    result = categorize_bugs()
    write_json(categorize_bugs_dir + bugs_chunk_count_path, result)
    print("finish categorizing bugs")