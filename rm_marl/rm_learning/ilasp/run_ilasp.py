import argparse
import json

from rm_marl.rm_learning.ilasp.ilasp_example_representation import lift_goal_example, lift_inc_example, \
    lift_dend_example
from .task_generator.ilasp_task_generator import generate_ilasp_task
from .task_solver.ilasp_solver import solve_ilasp_task
from .task_parser.ilasp_solution_parser import parse_ilasp_solutions


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_config", help="json file containing number of states, observables and examples")
    parser.add_argument("task_filename", help="filename of the ILASP task")
    parser.add_argument("solution_filename", help="filename of the ILASP task solution")
    parser.add_argument("plot_filename", help="filename of the automaton plot")
    parser.add_argument("--symmetry_breaking_method", "-s", default=None,
                        help="method for symmetry breaking (bfs, increasing_path)")
    return parser


if __name__ == "__main__":
    args = get_argparser().parse_args()
    with open(args.task_config) as f:
        config = json.load(f)

    binary_folder_name = "../bin"
    output_folder = "."

    goal_examples = [lift_goal_example(ex, f"ex_goal_{i}") for i, ex in enumerate(config["goal_examples"])]
    dend_examples = [lift_dend_example(ex, f"ex_dend_{i}") for i, ex in enumerate(config["dend_examples"])]
    inc_examples = [lift_inc_example(ex, f"ex_inc_{i}") for i, ex in enumerate(config["inc_examples"])]

    generate_ilasp_task(config["num_states"], "u_acc", "u_rej", config["observables"], goal_examples,
                        dend_examples, inc_examples, ".", args.task_filename,
                        args.symmetry_breaking_method, config["max_disjunction_size"], config["learn_acyclic"],
                        config["use_compressed_traces"], config["avoid_learning_only_negative"],
                        config["prioritize_optimal_solutions"], use_state_id_restrictions=False,
                        binary_folder_name="../bin")

    solve_ilasp_task(args.task_filename, args.solution_filename, binary_folder_name="../bin")
    automaton = parse_ilasp_solutions(args.solution_filename)
    automaton.plot(".", args.plot_filename)

'''
Configuration File Example:

{
    "num_states": 6,
    "max_disjunction_size": 1,
    "learn_acyclic": true,
    "use_compressed_traces": true,
    "avoid_learning_only_negative": false,
    "prioritize_optimal_solutions": false,
    "observables": ["a", "b", "c", "d", "f", "g", "m", "n"],
    "goal_examples": [
        [["f"], ["m"], ["g"]],
        [["m"], ["f"], ["g"]],
        [["m", "f"], ["g"]],
        [["m", "f", "g"]],
        [["m"], ["f", "g"]],
        [["f"], ["m", "g"]]
    ],
    "deadend_examples": [
        [["n"]],
        [["f"], ["n"]],
        [["m"], ["n"]],
        [["f"], ["m"], ["n"]],
        [["m"], ["f"], ["n"]]
    ],
    "inc_examples": [
        [["f"]],
        [["g"]],
        [["m"]],
        [[], []],
        [[], ["g"]],
        [[], ["m"]],
        [[], ["f"]],
        [["m"], []],
        [["f"], []],
        [["m"], ["g"]],
        [["m"], [], ["g"]],
        [["f"], ["g"]],
        [["f"], [], ["g"]],
        [[], ["f"], ["g"]],
        [[], ["m"], ["g"]],
        [[], [], []],
        [[], ["f"], []],
        [[], ["m"], []],
        [["f"], [], []],
        [["m"], [], []],
        [[], [], ["g"]],
        [["f"], ["m"]],
        [["f"], ["m"], []],
        [["m"], ["f"], []],
        [["m", "f"]],
        [["g", "f"]],
        [["g", "m"]]
    ]
}
'''
