####################################
### THIS CODE USES doe.py 
### FROM PROVIDED  GitHub repository
####################################



from pathlib import Path
ROOT = Path(__file__).parent.parent

### EXPERIMENTS WITH DIFFERENT PENALTIES
### 31 bonds
# doe_penalty_twolocal_common = {
#     'lp_file': f'{ROOT}/data/1/31bonds/docplex-bin-avgonly.lp',
#     'num_exec': 10, # was 10
#     'ansatz': 'TwoLocal',
#     'theta_initial': 'piby3',
#     'optimizer': 'nft',
#     'device':'AerSimulator',
#     'max_epoch': 4,
#     'shots': 2**13,
#     'theta_threshold': 0.,
#     }

# doe_penalty_bfcd_common = {
#     'lp_file': f'{ROOT}/data/1/31bonds/docplex-bin-avgonly.lp',
#     'num_exec': 10, 
#     'ansatz': 'bfcd',
#     'theta_initial': 'piby3',
#     'optimizer': 'nft',
#     'device':'AerSimulator',
#     'max_epoch': 4,
#     'shots': 2**13,
#     'theta_threshold': 0.,
#     }

# doe_penalty = {
#     # Twolocal bilinear entanglement
#     '1/31bonds/Penalty1p1_TwoLocal2rep_piby3_AerSimulator_0p1':
#         doe_penalty_twolocal_common | {
#             'experiment_id': 'Penalty1p1_TwoLocal2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             'penalty_val': 1.1,
#             },
#     '1/31bonds/Penalty1p5_TwoLocal2rep_piby3_AerSimulator_0p1':
#         doe_penalty_twolocal_common | {
#             'experiment_id': 'Penalty1p5_TwoLocal2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             'penalty_val': 1.5,
#             },
#     '1/31bonds/Penalty2p0_TwoLocal2rep_piby3_AerSimulator_0p1':
#         doe_penalty_twolocal_common | {
#             'experiment_id': 'Penalty2p0_TwoLocal2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             'penalty_val': 2.0,
#             },
#     '1/31bonds/Penalty2p5_TwoLocal2rep_piby3_AerSimulator_0p1':
#         doe_penalty_twolocal_common | {
#             'experiment_id': 'Penalty2p5_TwoLocal2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             'penalty_val': 2.5,
#             },
#     # BFCD
#     '1/31bonds/Penalty1p1_bfcd2rep_piby3_AerSimulator_0p1':
#         doe_penalty_bfcd_common | {
#             'experiment_id': 'Penalty1p1_bfcd2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             'penalty_val': 1.1,
#             },
#     '1/31bonds/Penalty1p5_bfcd2rep_piby3_AerSimulator_0p1':
#         doe_penalty_bfcd_common | {
#             'experiment_id': 'Penalty1p5_bfcd2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             'penalty_val': 1.5,
#             },
#     '1/31bonds/Penalty2p0_bfcd2rep_piby3_AerSimulator_0p1':
#         doe_penalty_bfcd_common | {
#             'experiment_id': 'Penalty2p0_bfcd2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             'penalty_val': 2.0,
#             },
#     '1/31bonds/Penalty2p5_bfcd2rep_piby3_AerSimulator_0p1':
#         doe_penalty_bfcd_common | {
#             'experiment_id': 'Penalty2p5_bfcd2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             'penalty_val': 2.5,
#             },
# }

###########################################
### EXPERIMENTS WITH RANDOM INITIALIZATIONS
### 31 bonds
# doe_initialization_twolocal_common = {
#     'lp_file': f'{ROOT}/data/1/31bonds/docplex-bin-avgonly.lp',
#     'num_exec': 10, # was 10
#     'ansatz': 'TwoLocal',
#     'theta_initial': 'random',
#     'optimizer': 'nft',
#     'device':'AerSimulator',
#     'max_epoch': 4,
#     'shots': 2**13,
#     'theta_threshold': 0.,
#     }

# doe_initialization_bfcd_common = {
#     'lp_file': f'{ROOT}/data/1/31bonds/docplex-bin-avgonly.lp',
#     'num_exec': 10, 
#     'ansatz': 'bfcd',
#     'theta_initial': 'random',
#     'optimizer': 'nft',
#     'device':'AerSimulator',
#     'max_epoch': 4,
#     'shots': 2**13,
#     'theta_threshold': 0.,
#     }

# doe_penalty = {
#     # Twolocal bilinear entanglement
#     '1/31bonds/RandomInit1_TwoLocal2rep_piby3_AerSimulator_0p1':
#         doe_initialization_twolocal_common | {
#             'experiment_id': 'RandomInit1_TwoLocal2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             'penalty_val': 1.1,
#             },
#     '1/31bonds/RandomInit1_TwoLocal2rep_piby3_AerSimulator_0p1':
#         doe_initialization_twolocal_common | {
#             'experiment_id': 'RandomInit1_TwoLocal2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },
#     '1/31bonds/RandomInit2_TwoLocal2rep_piby3_AerSimulator_0p1':
#         doe_initialization_twolocal_common | {
#             'experiment_id': 'RandomInit2_TwoLocal2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },
#     '1/31bonds/RandomInit3_TwoLocal2rep_piby3_AerSimulator_0p1':
#         doe_initialization_twolocal_common | {
#             'experiment_id': 'RandomInit3_TwoLocal2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },

#     # BFCD
#     '1/31bonds/RandomInit1_bfcd2rep_piby3_AerSimulator_0p1':
#         doe_initialization_bfcd_common | {
#             'experiment_id': 'RandomInit1_bfcd2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },
#     '1/31bonds/RandomInit2_bfcd2rep_piby3_AerSimulator_0p1':
#         doe_initialization_bfcd_common | {
#             'experiment_id': 'RandomInit2_bfcd2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },
#     '1/31bonds/RandomInit3_bfcd2rep_piby3_AerSimulator_0p1':
#         doe_initialization_bfcd_common | {
#             'experiment_id': 'RandomInit3_bfcd2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             }
# }

##################################################
### EXPERIMENT: 100 runs of the same problem
# doe_100runs_twolocal_common = {
#     'num_exec': 100, # was 10
#     'ansatz': 'TwoLocal',
#     'theta_initial': 'piby3',
#     'optimizer': 'nft',
#     'device':'AerSimulator',
#     'max_epoch': 4,
#     'shots': 2**13,
#     'theta_threshold': 0.,
#     }

# doe_100runs = {
#     '1/31bonds/Bonds30_TwoLocal2rep_piby3_AerSimulator_0p1':
#         doe_100runs_twolocal_common | {
#             'lp_file': f'{ROOT}/data/1/30bonds_100problems_1run/normalized_problem_30bonds_v3.lp',
#             'experiment_id': 'Bonds30_TwoLocal2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },
# }

##################################################
### EXPERIMENT: 1 run of 100 different problems
# doe_1run_twolocal_common = {
#     'num_exec': 1, # was 10
#     'ansatz': 'TwoLocal',
#     'theta_initial': 'piby3',
#     'optimizer': 'nft',
#     'device':'AerSimulator',
#     'max_epoch': 4,
#     'shots': 2**13,
#     'theta_threshold': 0.,
#     }

# doe_1run = {
#     # Twolocal bilinear entanglement
#     '1/31bonds/Bonds30_TwoLocal2rep_piby3_AerSimulator_0p1':
#         doe_1run_twolocal_common | {
#             'lp_file': f'{ROOT}/data/1/30bonds_100problems_1run/normalized_problem_30bonds_v3_0.lp',
#             'experiment_id': 'Bonds30_TwoLocal2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },
# }

##################################################
### EXPERIMENT: different number of bonds
# doe_bonds_twolocal_common = {
#     'num_exec': 10, # was 10
#     'ansatz': 'TwoLocal',
#     'theta_initial': 'piby3',
#     'optimizer': 'nft',
#     'device':'AerSimulator',
#     'max_epoch': 4,
#     'shots': 2**13,
#     'theta_threshold': 0.,
#     }

# doe_bonds_bfcd_common = {
#     'num_exec': 10, 
#     'ansatz': 'bfcd',
#     'theta_initial': 'piby3',
#     'optimizer': 'nft',
#     'device':'AerSimulator',
#     'max_epoch': 4,
#     'shots': 2**13,
#     'theta_threshold': 0.,
#     }

# doe_bonds = {
#     # Twolocal bilinear entanglement
#     '1/31bonds/Bonds15_TwoLocal2rep_piby3_AerSimulator_0p1':
#         doe_bonds_twolocal_common | {
#             'lp_file': f'{ROOT}/data/1/scaling/normalized_problem_15bonds_v3.lp',
#             'experiment_id': 'Bonds15_TwoLocal2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },
#     '1/31bonds/Bonds20_TwoLocal2rep_piby3_AerSimulator_0p1':
#         doe_bonds_twolocal_common | {
#             'lp_file': f'{ROOT}/data/1/scaling/normalized_problem_20bonds_v3.lp',
#             'experiment_id': 'Bonds20_TwoLocal2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },
#     '1/31bonds/Bonds25_TwoLocal2rep_piby3_AerSimulator_0p1':
#         doe_bonds_twolocal_common | {
#             'lp_file': f'{ROOT}/data/1/scaling/normalized_problem_25bonds_v3.lp',
#             'experiment_id': 'Bonds25_TwoLocal2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },
#     '1/31bonds/Bonds30_TwoLocal2rep_piby3_AerSimulator_0p1':
#         doe_bonds_twolocal_common | {
#             'lp_file': f'{ROOT}/data/1/scaling/normalized_problem_30bonds_v3.lp',
#             'experiment_id': 'Bonds30_TwoLocal2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },
#     # BFCD
#     '1/31bonds/Bonds15_bfcd2rep_piby3_AerSimulator_0p1':
#         doe_bonds_bfcd_common | {
#             'lp_file': f'{ROOT}/data/1/scaling/normalized_problem_15bonds_v3.lp',
#             'experiment_id': 'Bonds15_bfcd2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },
#     '1/31bonds/Bonds20_bfcd2rep_piby3_AerSimulator_0p1':
#         doe_bonds_bfcd_common | {
#             'lp_file': f'{ROOT}/data/1/scaling/normalized_problem_20bonds_v3.lp',
#             'experiment_id': 'Bonds20_bfcd2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },
#     '1/31bonds/Bonds25_bfcd2rep_piby3_AerSimulator_0p1':
#         doe_bonds_bfcd_common | {
#             'lp_file': f'{ROOT}/data/1/scaling/normalized_problem_25bonds_v3.lp',
#             'experiment_id': 'Bonds25_bfcd2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },
#     '1/31bonds/Bonds30_bfcd2rep_piby3_AerSimulator_0p1':
#         doe_bonds_bfcd_common | {
#             'lp_file': f'{ROOT}/data/1/scaling/normalized_problem_30bonds_v3.lp',
#             'experiment_id': 'Bonds30_bfcd2rep_piby3_AerSimulator_0p1',
#             'ansatz_params': {'reps': 2, 'entanglement': 'bilinear'},
#             'alpha': 0.1,
#             },
# }


############################
## HEAT EXCHANGE ANSATZ
doe_he_common = {
    'lp_file': f'{ROOT}/data/1/scaling/normalized_problem_15bonds_v3.lp',
    'num_exec': 1, # was 10
    'ansatz': 'he',
    'theta_initial': 'piby3',
    'optimizer': 'nft',
    'device':'AerSimulator',
    'max_epoch': 4,
    'shots': 2**13,
    'theta_threshold': 0.,
    }


doe_he = {
    # he -- new
    '1/scaling/he1rep_piby3_AerSimulator_0.1':
        doe_he_common | {
            'experiment_id': 'he1rep_piby3_AerSimulator_0.1',
            'ansatz_params': {'reps': 1},
            'alpha': 0.1,
            },
    }
