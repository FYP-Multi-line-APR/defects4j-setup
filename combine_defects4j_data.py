import os 
import sys 
from utils import get_list_of_dirs_in, get_files_inside_dir, get_json_data, write_json

if __name__ == "__main__":
    data_dir_path = "train-data/set16-context-all-width-5"
    # data_dir_path = sys.argv[1]
    result = []
    result_file_path = "train-data/result.json"
    dirs = get_list_of_dirs_in(data_dir_path)
    for dir in dirs:
        project_dir_path = os.path.join(data_dir_path, dir)
        files_inside = get_files_inside_dir(project_dir_path)
        for file in files_inside:
            file_path = os.path.join(project_dir_path, file)
            file_data = get_json_data(file_path)
            result.extend(file_data)
    write_json(result_file_path, result)
    print(dirs)