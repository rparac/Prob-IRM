"""
Important - should be run from a main directory (python hyperparameter_search/start.py)
"""

import subprocess

import optuna

# The outputs folder should exist
study_name = "optuna_study_1"
db_path = f"sqlite:///outputs/{study_name}.db"
study = optuna.create_study(study_name=study_name, storage=db_path, direction="minimize")

num_experiments = 10
for i in range(num_experiments):
    directory = "test_dqrm"
    name = f"{directory}_run_{i}"
    subprocess.run(["python", "submit_rcs_script.py", directory, name, "hyperparameter_search.single_run.py", name, db_path])
