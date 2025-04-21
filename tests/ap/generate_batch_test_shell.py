import shutil
import os
import os.path
import random
import sys
import hashlib
import string
from absl import app, flags
import time

FLAGS = flags.FLAGS

flags.DEFINE_string("test_dir", "/all_case_v2/", "")
flags.DEFINE_string("log_root", "/work/Paddle/Athena/tests/ap/ap_matmul_graphs/", "")
flags.DEFINE_string("dir_name", "/work/PaddleTest/", "")
flags.DEFINE_string("shell_script", "/work/Paddle/Athena/tests/ap/test_single_matmul_file.sh", "")


def test_all_graphs(log_dir):
    print(f"mkdir -p {log_dir}")
    for file_name in os.listdir(FLAGS.dir_name + FLAGS.test_dir): 
        print(f"bash {FLAGS.shell_script} {file_name} {log_dir}")
        print("sleep 5")
        print("")

def hash_code(*args, **kwargs):
    return time.strftime("%Y-%m-%d_%H_%M_%S")


def main(argv):
    log_dir = f"{FLAGS.log_root}{hash_code()}"
    test_all_graphs(log_dir)

if __name__ == "__main__":
    app.run(main)