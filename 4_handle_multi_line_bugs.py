from utils import get_json_data

if __name__ == "__main__":
    multi_line_bugs_json_file_path = 'multi-line-bugs.json'
    json_data = get_json_data(multi_line_bugs_json_file_path)
    print(json_data['Chart'])