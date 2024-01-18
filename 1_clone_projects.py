import os 
import subprocess
from utils import get_pids_list, get_bids_list

defects4j_path = "defects4j"
cloned_path = "./cloned-projects"
os.environ['PATH'] = defects4j_path + ":" + os.environ['PATH']

def make_dir(clone_pid_path):
    if not os.path.exists(clone_pid_path) or not os.path.isdir(clone_pid_path):
        os.makedirs(clone_pid_path) 

def checkout_project(pid, bid):
    checkout_command = f"{defects4j_path} checkout -p {pid} -v {bid}b -w {cloned_path}/{pid}/{bid}"
    os.system(checkout_command)

def handle_pids(pids_list):
    for pid in pids_list:
        bids_list = get_bids_list(pid)
        make_dir(f"{cloned_path}/{pid}")
        for bid in bids_list:
            checkout_project(pid, bid)


if __name__ == "__main__":
    pid = "Closure"
    bid = "169"
    
    # checkout_project(pid, bid)
    pids_list = get_pids_list()
    handle_pids(pids_list)
    print(pids_list)
    print("finish")