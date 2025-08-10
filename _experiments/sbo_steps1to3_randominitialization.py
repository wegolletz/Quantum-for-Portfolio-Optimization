####################################
### THIS CODE USES sbo_steps1to3.py 
### FROM PROVIDED  GitHub repository
####################################


import pickle as pkl
import time
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# from pathlib import Path
from qiskit import qpy
import dataclasses
# from qiskit_aer import AerSimulator
# import docplex.mp.model_reader
# from qiskit_ibm_catalog import QiskitServerless

from pathlib import Path
ROOT = Path(__file__).parent.parent
print(ROOT)

import sys
sys.path.append(str(ROOT))

from src.step_1 import get_cplex_sol, problem_mapping
from src.experiment import Experiment
from src.sbo.src.patterns.building_blocks.step_3 import HardwareExecutor
from doe import doe_random


def execute_multiple_runs(lp_file: str, experiment_id: str, num_exec: int, ansatz: str, ansatz_params: dict, 
                          theta_initial: str, device: str, instance: str, optimizer: str, max_epoch: int, alpha: float, shots: int,
                          run_on_serverless: bool, theta_threshold: float, penalty_val: float = 1.1):

    lp_file = lp_file.replace('.lp','-nocplexvars.lp')

    # step 1 PROBLEM MAPPING
    qubo_problem, ansatz_, theta_initial_, backend, initial_layout = problem_mapping(lp_file, ansatz, ansatz_params, theta_initial, device, instance,experiment_id, penalty_val)

    # create refval and check consistency of the obj fn
    refx, refval = get_cplex_sol(lp_file, qubo_problem)
    print(f"REFVAL = {refval}")

    (Path(lp_file).parent / experiment_id).mkdir(exist_ok=True)
    print(initial_layout, backend)
    # step_2 CIRCUIT OPTIMIZATION
    isa_ansatz_file = Path(lp_file).parent / f'{experiment_id}/isa_ansatz.qpy'
    if isa_ansatz_file.is_file():
        with open(isa_ansatz_file, 'rb') as f:
            isa_ansatz = qpy.load(f)
            isa_ansatz = isa_ansatz[0]
    else:
        isa_ansatz = generate_preset_pass_manager(target=backend.target, optimization_level=1, initial_layout=initial_layout).run(ansatz_)
        with open(isa_ansatz_file, 'wb') as f:
            qpy.dump(isa_ansatz, f)



    # step_3 HW EXECUTION
    for exec in range(num_exec):
        print(f'Experiment {exec}\n')
        experiment_id_with_exec = f'{experiment_id}/{exec}'
        if not run_on_serverless:
            out_file = Path(lp_file).parent / f'{experiment_id}/exp{exec}.pkl'
            if out_file.is_file():
                print(f'File {out_file} exists. Skipped.')
                continue
            
            t = time.time()
            he = HardwareExecutor(
                objective_fun=qubo_problem,
                backend=backend,
                isa_ansatz=isa_ansatz,
                optimizer_theta0=theta_initial_,
                optimizer_method=optimizer,
                refvalue=refval,
                sampler_options={'default_shots':shots, 'dynamical_decoupling':{'enable': True}},
                use_session=False,
                iter_file_path_prefix=str(Path(lp_file).parent / experiment_id_with_exec),
                # verbose="iteration_all",
                store_all_x=True,
                solver_options={"max_epoch": max_epoch, "alpha": alpha, 'theta_threshold': theta_threshold},
            )
            result = he.run()
            step3_time = time.time() - t

            out = Experiment.from_step3(
                experiment_id_with_exec,
                ansatz, ansatz_params, theta_initial, device, optimizer, alpha, theta_threshold, lp_file, shots, refx, refval,
                Experiment.get_current_classical_hw(), step3_time, he.job_ids,
                result, he.optimization_monitor
            )
            with open(out_file, 'bw') as f:
                pkl.dump(dataclasses.asdict(out), f)

        # serverless execution removed - no needed


        # step 4 POSTPROCESSING
        # step 4 is deferred to another script


if __name__ == '__main__':
    # for _, exp in doe.items():
    #     if exp['device'] != 'AerSimulator':
    #         continue
    #     execute_multiple_runs(**exp, instance='', run_on_serverless=False)
    
    ###########################################
    ### EXPERIMENTS WITH RANDOM INITIALIZATIONS
    ### 31 bonds
    names = [
        '1/31bonds/RandomInit1_bfcd2rep_piby3_AerSimulator_0p1',
        '1/31bonds/RandomInit2_bfcd2rep_piby3_AerSimulator_0p1',
        '1/31bonds/RandomInit3_bfcd2rep_piby3_AerSimulator_0p1',
        '1/31bonds/RandomInit1_TwoLocal2rep_piby3_AerSimulator_0p1',
        '1/31bonds/RandomInit2_TwoLocal2rep_piby3_AerSimulator_0p1',
        '1/31bonds/RandomInit3_TwoLocal2rep_piby3_AerSimulator_0p1'
    ]
    
    for iname in names:    
        execute_multiple_runs(**doe_random[iname], instance='', run_on_serverless=False)