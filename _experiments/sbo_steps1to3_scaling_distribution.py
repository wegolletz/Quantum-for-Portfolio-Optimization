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
from doe import doe_1run, doe_100runs


def execute_multiple_runs(lp_file: str, experiment_id: str, num_exec: int, ansatz: str, ansatz_params: dict, 
                          theta_initial: str, device: str, instance: str, optimizer: str, max_epoch: int, alpha: float, shots: int,
                          run_on_serverless: bool, theta_threshold: float, penalty_val: float = 1.1):

    # lp_file = lp_file.replace('.lp','-nocplexvars.lp')

    # step 1 PROBLEM MAPPING
    qubo_problem, ansatz_, theta_initial_, backend, initial_layout = problem_mapping(lp_file, ansatz, ansatz_params, theta_initial, device, instance, penalty_val)

    # create refval and check consistency of the obj fn
    refx, refval, _ = get_cplex_sol(lp_file, qubo_problem)
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
    
    ##################################################
    ### EXPERIMENT: 1 run of 100 different problems
    names = [
        '1/31bonds/Bonds30_TwoLocal2rep_piby3_AerSimulator_0p1'
    ]
    
    for iname in names:
        for id in range(100):
            print(id)
            config = doe_1run[iname].copy()
            
            # Modify the 'lp_file' entry
            config['lp_file'] = f'{ROOT}/data/1/30bonds_100problems_1run/normalized_problem_30bonds_v3_{id}.lp'
            config['experiment_id'] = f'Bonds30_TwoLocal2rep_piby3_AerSimulator_{id}_0p1'
            # Call the function with the modified config
            execute_multiple_runs(**config, instance='', run_on_serverless=False)

    ##################################################
    ### EXPERIMENT: 100 runs of the same problem
    # names = [
    #     '1/31bonds/Bonds30_TwoLocal2rep_piby3_AerSimulator_0p1'
    # ]
    
    # for iname in names:
    #     execute_multiple_runs(**doe_100runs[iname], instance='', run_on_serverless=False)

