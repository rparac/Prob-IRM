import os

from rm_marl.rm_learning.ilasp.task_solver import solve_ilasp_task


def _pylasp_example_penalty_string():
    return """
#ilasp_script
ilasp.cdilp.initialise()

ilasp.cdilp.add_to_meta_program(ilasp.ids_to_meta_representation(ilasp.s_m()))
ilasp.cdilp.add_to_meta_program(ilasp.background_meta_representation())
ilasp.cdilp.add_to_meta_program(ilasp.meta_encoding())
ilasp.cdilp.add_to_meta_program("active(X) :- nge_HYP(X).")
ilasp.cdilp.add_to_meta_program("priority(X) :- level(_, X).")

all_examples = ilasp.all_examples()
for eg_id in all_examples:
    eg = ilasp.get_example(eg_id)
    if eg['type'] == 'positive' or eg['type'] == 'brave-order':
        ilasp.cdilp.add_to_meta_program(ilasp.meta_representation(eg_id))

solve_result = ilasp.cdilp.solve()

print(solve_result['example_penalty'])
#end.
"""


def get_ilasp_solution_penalty(output_folder, solution, ilasp_task):
    if solution is None or ilasp_task is None:
        return 100000000000

    # Generate pylasp script
    output_filename = 'pylasp_example_penalty.py'
    pylasp_filename = os.path.join(output_folder, output_filename)
    with open(pylasp_filename, 'w') as f:
        f.write(_pylasp_example_penalty_string())

    result_file = f'{output_folder}/total_penalty.tmp'
    solve_ilasp_task([ilasp_task, solution], result_file, pylasp_script_name=pylasp_filename)

    result = open(result_file).read()
    return int(result)
