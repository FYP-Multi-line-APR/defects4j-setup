from utils import get_available_bug_count_for_projects_in_fine_tune_train_data, get_available_bug_count_for_projects_in_train_data
from utils import get_available_bug_count_for_projects_in_cloned_data

if __name__ == "__main__":
    print("print bug counts in cloned data")
    print(get_available_bug_count_for_projects_in_cloned_data())

    print()
    print()
    print("print bug counts in train data")
    print(get_available_bug_count_for_projects_in_train_data())

    print()
    print()
    print("print bug counts in fine tune train data")
    print(get_available_bug_count_for_projects_in_fine_tune_train_data())