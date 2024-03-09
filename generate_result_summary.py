import sys 

from utils import get_json_data, write_to_file

def extract_bug_result(bug_result_str):
    result = bug_result_str.split("/")
    return [int(x) for x in result]

def generate_summary_message(project, fix_count, prediction_close_count, bug_count):
    if prediction_close_count is None:
        return f"{project}: {fix_count} / {bug_count}\n"
    else:
        return f"{project}: {fix_count} / {prediction_close_count} / {bug_count}\n"

def generate_summary(result_data):
    projects = result_data.keys()
    summary = ""
    complete_fix_count = 0
    partial_fix_count = 0
    multiline_fix_count = 0
    fixed_multiline_bugs = []
    fixed_singleline_bugs = []
    partially_fixed_multiline_bugs = []
    for project in projects:
        results_info = result_data[project]
        if not results_info:
            continue
        # check first one to identify the pattern
        sample_result_str = list(results_info.items())[0][1]
        if len(extract_bug_result(sample_result_str)) == 3:
            fix_count, prediction_close_count, bug_count = 0, 0, 0
            for bug, bug_result_str in results_info.items():
                fix_amount, close_amount, bug_amount = extract_bug_result(bug_result_str)
                fix_count += fix_amount
                prediction_close_count += close_amount 
                bug_count += bug_amount
                if fix_amount == bug_amount:
                    complete_fix_count += 1
                    if bug_amount > 1:
                        multiline_fix_count += 1
                        fixed_multiline_bugs.append(f"{project}{bug}")
                    else:
                        fixed_singleline_bugs.append(f"{project}{bug}")
            summary += generate_summary_message(project, fix_count, prediction_close_count, bug_count)
        elif len(extract_bug_result(sample_result_str)) == 2:
            fix_count, bug_count = 0, 0
            for bug, bug_result_str in results_info.items():
                fix_amount, bug_amount = extract_bug_result(bug_result_str)
                fix_count += fix_amount
                bug_count += bug_amount
                if fix_amount == bug_amount:
                    complete_fix_count += 1 
                    if bug_amount > 1:
                        multiline_fix_count += 1 
                        fixed_multiline_bugs.append(f"{project}{bug}")
                    else:
                        fixed_singleline_bugs.append(f"{project}{bug}")
                elif 0 < fix_amount and fix_amount < bug_amount:
                    partial_fix_count += 1
                    partially_fixed_multiline_bugs.append(f"{project}{bug}:{fix_amount}/{bug_amount}")
            summary += generate_summary_message(project, fix_count, None, bug_count)
    fix_message = f"completely fix: {complete_fix_count}, multiline fix: {multiline_fix_count}\n"
    multiline_bugs_name = "fixed multilines: " + str(fixed_multiline_bugs) + "\n"
    singleline_bugs_name = "fixed singlelines: " + str(fixed_singleline_bugs) + "\n"
    partial_fix_message = f"partial fix multilines: {partial_fix_count}\n"
    partial_fix_bugs_info = f"partial fix mulines: {str(partially_fixed_multiline_bugs)}\n"
    return fix_message + multiline_bugs_name + singleline_bugs_name + partial_fix_message + partial_fix_bugs_info + summary

def get_file_to_write_path(input_file_path):
    global txt_ext
    file_name = input_file_path.split('.')[0]
    file_name = file_name + "-summary-new"
    return file_name + txt_ext

json_ext = ".json"
txt_ext = ".txt"

if __name__ == "__main__":
    result_file_path = sys.argv[1]
    file_to_write = get_file_to_write_path(result_file_path)
    result_data = get_json_data(result_file_path) 
    summary = generate_summary(result_data)
    write_to_file(file_to_write, summary)

