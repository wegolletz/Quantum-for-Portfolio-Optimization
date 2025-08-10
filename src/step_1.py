import numpy as np
import docplex.mp
import docplex.mp.model
import docplex.mp.constants
import numpy as np
import docplex.mp
import docplex.mp.model
import docplex.mp.model_reader
import docplex.mp.solution
from qiskit.circuit.library import TwoLocal, NLocal, RYGate
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit_aer import AerSimulator
# from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.backend import BackendV2
from qiskit.transpiler import CouplingMap
from qiskit.circuit.library import n_local
from qiskit import ClassicalRegister

import json
import os

from pathlib import Path

import numba
from numba import jit

# from line_profiler import profile

from scipy.sparse import csr_array

def model_to_obj_dense(model: docplex.mp.model.Model, penalty_val: float = 1.1):
    num_vars = model.number_of_binary_variables
    num_ctr = model.number_of_constraints
    print(f"Number of variables = {num_vars}, Number of constraints = {num_ctr}")

    # Parsing objective under assumption:
    # - the objective is in the form quadratic + linear + c
    # Output: Q (as a dense matrix) and c such that      objective = x^T Q x + c

    Q = np.zeros((num_vars, num_vars))
    c = model.objective_expr.get_constant()

    for i, dvari in enumerate(model.iter_variables()):
        for j, dvarj in enumerate(model.iter_variables()):
            Q[i,j] = model.objective_expr.get_quadratic_coefficient(dvari, dvarj) / 2
            if i == j:
                Q[i,i] = model.objective_expr.get_quadratic_coefficient(dvari, dvari) + model.objective_expr.linear_part.get_coef(dvari)

    # Parsing constraints under the assumption:
    # - they are all linear inequalities *** it could be generalized to equalities!
    # Retrieving A and b such that constraints write as    A x - b ≥ 0.

    A = np.zeros((num_ctr, num_vars))
    b = np.zeros(num_ctr)

    for i, ctr in enumerate(model.iter_constraints()):
        sense = 1 if ctr.sense == docplex.mp.constants.ComparisonType.GE else -1
        for j, dvarj in enumerate(model.iter_variables()):
            A[i,j] = sense * ctr.lhs.get_coef(dvarj)
        b[i] = sense * ctr.rhs.get_constant()

    # Rescale constraints so that the minimum coefficient of a variable in each constraint is 1 (in abs)

    min_A_by_row = np.zeros(num_ctr)

    for i,row in enumerate(A):
        min_A_by_row[i] = np.min(np.abs(row[np.nonzero(row)]))

    A = A / min_A_by_row.reshape(num_ctr, 1)
    b = b / min_A_by_row

    # Translate constraints into obj terms under the following assumptions:
    # - minimization problem
    # - all vars are bin (or integer)
    # - each coef in constraints is ≥ 1 (in abs)
    # Remark: the resulting unconstr_obj_fn is not polynomial (contains maximum), and it's designed to be used in sampling VQE

    max_obj = np.sum(Q, where=Q>0)
    min_obj = np.sum(Q, where=Q<0)
    penalty = (max_obj-min_obj) * penalty_val

    @jit
    def obj_fn_embedding_constraints(x):
        return x @ Q @ x + c + penalty * np.sum(np.maximum(b - A @ x, 0)**2)

    return obj_fn_embedding_constraints


def model_to_obj_sparse(model: docplex.mp.model.Model, penalty_val: float = 1.1):
    num_vars = model.number_of_binary_variables
    num_ctr = model.number_of_constraints
    print(f"Number of variables = {num_vars}, Number of constraints = {num_ctr}")

    # Parsing objective under assumption:
    # - the objective is in the form quadratic + linear + c
    # Output: Q (as a dense matrix) and c such that      objective = x^T Q x + c

    Q = np.zeros((num_vars, num_vars))
    c = model.objective_expr.get_constant()

    for i, dvari in enumerate(model.iter_variables()):
        for j, dvarj in enumerate(model.iter_variables()):
            if i > j: continue
            Q[i,j] = model.objective_expr.get_quadratic_coefficient(dvari, dvarj)
            if i == j:
                Q[i,i] += model.objective_expr.linear_part.get_coef(dvari)

    # Parsing constraints under the assumption:
    # - they are all linear inequalities *** it could be generalized to equalities!
    # Retrieving A and b such that constraints write as    A x - b ≥ 0.

    A = np.zeros((num_ctr, num_vars))
    b = np.zeros(num_ctr)

    for i, ctr in enumerate(model.iter_constraints()):
        sense = 1 if ctr.sense == docplex.mp.constants.ComparisonType.GE else -1
        for j, dvarj in enumerate(model.iter_variables()):
            A[i,j] = sense * ctr.lhs.get_coef(dvarj)
        b[i] = sense * ctr.rhs.get_constant()

    # Rescale constraints so that the minimum coefficient of a variable in each constraint is 1 (in abs)

    min_A_by_row = np.zeros(num_ctr)

    for i,row in enumerate(A):
        min_A_by_row[i] = np.min(np.abs(row[np.nonzero(row)]))

    A = A / min_A_by_row.reshape(num_ctr, 1)
    b = b / min_A_by_row

    # Translate constraints into obj terms under the following assumptions:
    # - minimization problem
    # - all vars are bin (or integer)
    # - each coef in constraints is ≥ 1 (in abs)
    # Remark: the resulting unconstr_obj_fn is not polynomial (contains maximum), and it's designed to be used in sampling VQE

    max_obj = np.sum(Q, where=Q>0)
    min_obj = np.sum(Q, where=Q<0)
    penalty = (max_obj-min_obj) * penalty_val

    sparseQ = csr_array(Q)
    sparseA = csr_array(A)

    @jit
    def obj_fn_embedding_constraints(x):
        return x @ (sparseQ @ x) + c + penalty * np.sum(np.maximum(b - sparseA @ x, 0)**2)

    return obj_fn_embedding_constraints


def model_to_obj_sparse_numba(model: docplex.mp.model.Model, penalty_val: float = 1.1):
    num_vars = model.number_of_binary_variables
    num_ctr = model.number_of_constraints
    print(f"Number of variables = {num_vars}\nNumber of constraints = {num_ctr}\n")

    # Parsing objective under assumption:
    # - the objective is in the form quadratic + linear + c
    # Output: Q (as a dense matrix) and c such that      objective = x^T Q x + c

    Q = np.zeros((num_vars, num_vars))
    c = model.objective_expr.get_constant()

    for i, dvari in enumerate(model.iter_variables()):
        for j, dvarj in enumerate(model.iter_variables()):
            if i > j: continue
            Q[i,j] = model.objective_expr.get_quadratic_coefficient(dvari, dvarj)
            if i == j:
                Q[i,i] += model.objective_expr.linear_part.get_coef(dvari)

    # Parsing constraints under the assumption:
    # - they are all linear inequalities *** it could be generalized to equalities!
    # Retrieving A and b such that constraints write as    A x - b ≥ 0.

    A = np.zeros((num_ctr, num_vars))
    b = np.zeros(num_ctr)

    for i, ctr in enumerate(model.iter_constraints()):
        sense = 1 if ctr.sense == docplex.mp.constants.ComparisonType.GE else -1
        for j, dvarj in enumerate(model.iter_variables()):
            A[i,j] = sense * ctr.lhs.get_coef(dvarj)
        b[i] = sense * ctr.rhs.get_constant()

    # Rescale constraints so that the minimum coefficient of a variable in each constraint is 1 (in abs)

    min_A_by_row = np.zeros(num_ctr)

    for i,row in enumerate(A):
        min_A_by_row[i] = np.min(np.abs(row[np.nonzero(row)]))

    A = A / min_A_by_row.reshape(num_ctr, 1)
    b = b / min_A_by_row

    # Translate constraints into obj terms under the following assumptions:
    # - minimization problem
    # - all vars are bin (or integer)
    # - each coef in constraints is ≥ 1 (in abs)
    # Remark: the resulting unconstr_obj_fn is not polynomial (contains maximum), and it's designed to be used in sampling VQE

    max_obj = np.sum(Q, where=Q>0)
    min_obj = np.sum(Q, where=Q<0)
    penalty = (max_obj-min_obj) * penalty_val

    sparseQ = csr_array(Q)
    sparseQ_data = sparseQ.data
    sparseQ_indices = sparseQ.indices
    sparseQ_indptr = sparseQ.indptr
    
    sparseA = csr_array(A)
    sparseA_data = sparseA.data
    sparseA_indices = sparseA.indices
    sparseA_indptr = sparseA.indptr

    @jit
    def matrix_vector_sparse(As_data, As_indices, As_indptr, b):
        res = np.zeros(len(As_indptr)-1, dtype=float)
        for i in range(len(As_indptr)-1):
            for j in range(As_indptr[i], As_indptr[i+1]):
                x = b[As_indices[j]]
                y = As_data[j]
                res[i] += x*y
        return res

    @jit
    def obj_fn_embedding_constraints(x):
        y = matrix_vector_sparse(sparseQ_data, sparseQ_indices, sparseQ_indptr, x)
        z= x @y
        w = np.sum(np.maximum(b - matrix_vector_sparse(sparseA_data, sparseA_indices, sparseA_indptr, x), 0)**2)
        return z + c + penalty * w

    return obj_fn_embedding_constraints

model_to_obj = model_to_obj_sparse_numba


def get_cplex_sol(lp_file: str, obj_fn, threshold: float =1e-8):
    model: docplex.mp.model.Model = docplex.mp.model_reader.ModelReader.read(lp_file)

    sol: docplex.mp.solution.SolveSolution = model.solve()
    x_cplex = [v.solution_value for v in model.iter_binary_vars()]

    # check consistency with obj_fn
    fx_cplex = obj_fn(np.array(x_cplex, dtype=float))
    error = np.abs(fx_cplex - sol.objective_value) 
    assert error < threshold
    
    # print(f"Cplex solution: {x_cplex}")
    print(f"CPLEX objective value: \n{fx_cplex}")
    print(f"QUBO problem evaluated on CPLEX solution: \n{sol.objective_value}")
    print(f"Error = {error}\n")
    return x_cplex, fx_cplex, model.number_of_binary_variables

def make_iswap_like_block():
    """Creates a 2-qubit iSWAP-like parameterized gate."""

    qc_params = ParameterVector(name='p', length=1)
    qc = QuantumCircuit(2)

    qc.rxx(-qc_params[0]/2, 0, 1)
    qc.ryy(-qc_params[0]/2, 0, 1)

    return qc

def build_ansatz(ansatz: str, ansatz_params: dict, num_qubits: int, backend: BackendV2) -> tuple[QuantumCircuit, dict | None]:

    def apply_entanglement_map(entanglement):
        # custom-constructed
        if entanglement == 'bilinear':
            return [[i, i+1] for i in range(0, num_qubits-1, 2)] + [[i, i+1] for i in range(1, num_qubits-1, 2)], None

        if entanglement == 'color':
            assert backend.coupling_map.is_symmetric, 'Non-sym coupling map. Do you mean `di-color`?'
            coupling_map = [(i,j) for i,j in backend.coupling_map if i<j]
        elif entanglement == 'di-color':
            coupling_map = list(backend.coupling_map)
            assert not backend.coupling_map.is_symmetric, 'Sym coupling map. Do you mean `color`?'
        if entanglement in ['color', 'di-color']:
            nodes = list(range(backend.num_qubits))
            # TODO: this could be smarter
            for _ in range(num_qubits, backend.num_qubits):
                node_degrees = {n: sum(i == n or j == n for i,j in coupling_map) for n in nodes}
                remove_node = min(node_degrees.keys(), key=(lambda k: node_degrees[k]))
                nodes.remove(remove_node)
                coupling_map = [(i,j) for i,j in coupling_map if i != remove_node and j != remove_node]

            from qiskit_addon_utils.coloring import auto_color_edges
            coloring: dict = auto_color_edges(coupling_map)
            num_colors = max(v for _,v in coloring.items())+1

            out_map = [k for c1 in range(num_colors) for k, c in coloring.items() if c==c1]
            initial_layout = {idx: node for idx, node in enumerate(nodes)}
            reverse_layout = {node: idx for idx, node in enumerate(nodes)}
            return [(reverse_layout[i], reverse_layout[j]) for i,j in out_map], initial_layout
        
        # default
        return entanglement, None
    
    print(f'The ansatz from doe file = {ansatz}')
    if ansatz == 'TwoLocal':
        print('Using TwoLocal ansatz\n')
        ansatz_params_ = {'rotation_blocks':'ry', 'entanglement_blocks':'cz', 'entanglement': 'bilinear', 'reps': 1}
        ansatz_params_.update(ansatz_params)
        ansatz_params_['entanglement'], initial_layout = apply_entanglement_map(ansatz_params_['entanglement'])
        ansatz_ = TwoLocal(num_qubits, **ansatz_params_)
        ansatz_.measure_all()
    
    elif (ansatz == 'bfcd') or (ansatz == 'bfcdR'):
        print('Using bfcd/bfcdR ansatz\n')
        # bfcd -> uses 2 parameters (for 2 RZZ gates)
        # bfcdR -> uses 1 parameter (shared by both RZZ gates)
        # ParameterVector creates symbolic variables p[0], p[1] to be optimized later.
        qc_params = ParameterVector(name='p',length=2 if ansatz == 'bfcd' else 1)
        entanglement_block = QuantumCircuit(2)
        # Rzy
        entanglement_block.rx(np.pi / 2, 0)
        entanglement_block.rzz(qc_params[0], 0, 1)
        entanglement_block.rx(- np.pi / 2, 0)
        # Ryz
        entanglement_block.rx(np.pi / 2, 1)
        entanglement_block.rzz(qc_params[1] if ansatz == 'bfcd' else qc_params[0], 0, 1)
        entanglement_block.rx(- np.pi / 2, 1)

        ansatz_params_ = {'rotation_blocks': 'ry',
                          'entanglement_blocks':entanglement_block,
                          'overwrite_block_parameters': True,
                          'flatten': True,
                          'entanglement': 'bilinear',
                          'reps': 1,
                          'skip_final_rotation_layer': True,
                        #   'insert_barriers': True,
                          }
        ansatz_params_.update(ansatz_params)
        if isinstance(ansatz_params_['rotation_blocks'], str):
            ansatz_params_['rotation_blocks'] = get_standard_gate_name_mapping()[ansatz_params_['rotation_blocks']]
        ansatz_params_['entanglement'], initial_layout = apply_entanglement_map(ansatz_params_['entanglement'])
        ansatz_ = NLocal(num_qubits, **ansatz_params_)
        ansatz_.measure_all()

    elif ansatz == 'he':
        print('Using HE ansatz\n')
        print(f"Num qubits = {num_qubits}\n")
        num_qubits = int(num_qubits/2)
        total_qubits = 2*num_qubits  # problem + bath
        iswap_block = make_iswap_like_block()

        # Entangler map: connect problem[i] to bath[i]
        entangler_map = [[i, num_qubits + i] for i in range(num_qubits)]

        # # Initial bath state |1⟩
        # init = QuantumCircuit(total_qubits)
        # for i in range(num_qubits, total_qubits):
        #     init.x(i)

        # === Step 1: Prepare Initial State Circuit ===
        init_circ = QuantumCircuit(total_qubits)

        # Set bath qubits (second half) to |1⟩
        for i in range(num_qubits, total_qubits):
            init_circ.x(i)

        # Build HE ansatz
        heansatz_ = n_local(
            num_qubits=total_qubits,
            rotation_blocks=[],
            entanglement_blocks=iswap_block.to_gate(),
            entanglement=entangler_map,
            reps=1,
            skip_final_rotation_layer=True,
            overwrite_block_parameters=True,
            parameter_prefix='theta'
        )
        ansatz_ = init_circ.compose(heansatz_)
        # ansatz_.measure_all()

        creg = ClassicalRegister(num_qubits)
        ansatz_.add_register(creg)
        ansatz_.measure(list(range(num_qubits)),list(range(num_qubits)))
        

        initial_layout = None

    else:
        raise ValueError('unknown ansatz')
    
    if initial_layout is not None:
        initial_layout = {ansatz_.qubits[k]: v for k,v in initial_layout.items()}
    
    return ansatz_, initial_layout

def get_or_create_theta(ansatz_name: str, num_qubits: int, num_params: int, seed: int = 42):
    save_dir = f"../data/1/{num_qubits}bonds/theta_init"
                        
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = f"theta_init_{ansatz_name}_numbonds{num_qubits}_numparams{num_params}_seed{seed}.json"
    save_path = save_dir / file_name

    if save_path.exists():
        with open(save_path, "r") as f:
            theta = np.array(json.load(f)["theta"])
        print(f"Loaded theta from: {save_path}")
    else:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 2*np.pi, size=num_params)
        with open(save_path, "w") as f:
            json.dump({"seed": seed, "theta": theta.tolist()}, f)
        print(f"Created new theta and saved to: {save_path}")

    return theta

def get_backend(device: str, instance: str, num_vars: int) -> BackendV2:
    if device == 'AerSimulator':
        aer_options={'method' : 'matrix_product_state', 'n_qubits': num_vars}
        backend = AerSimulator(**aer_options)
    elif device[:4] == 'ibm_':
        service = QiskitRuntimeService()
        backend = service.backend(device, instance)
    else:
        raise ValueError('unknown device')
    
    return backend


def problem_mapping(lp_file: str, ansatz: str, ansatz_params: dict, theta_initial: str, device: str, instance: str, penalty_val: float = 1.1):
    model: docplex.mp.model.Model = docplex.mp.model_reader.ModelReader.read(lp_file)

    print(f"Penalty value = {penalty_val}\n")
    obj_fn = model_to_obj(model, penalty_val)
    num_vars = 2 * model.number_of_binary_variables if ansatz == 'he' else model.number_of_binary_variables

    backend = get_backend(device, instance, num_vars)
    if 'from_backend' in ansatz_params:
        build_backend = get_backend(ansatz_params['from_backend'], instance, num_vars)
    else:
        build_backend = backend
    ansatz_params_ = ansatz_params.copy()
    ansatz_params_.pop('from_backend', None)
    ansatz_params_.pop('discard_initial_layout', None)
    ansatz_, initial_layout = build_ansatz(ansatz, ansatz_params_, num_vars, build_backend)
    if ansatz_params.get('discard_initial_layout', False):
        initial_layout = None

    if theta_initial == 'piby3':
        print(f"Theta initial: piby3\n")
        theta_initial_ = np.pi/3 * np.ones(ansatz_.num_parameters)
        print(len(theta_initial_),ansatz_.num_parameters, num_vars)
    elif theta_initial == 'random':
        theta_initial_ = get_or_create_theta(ansatz, num_vars, ansatz_.num_parameters, seed=42)
    else:
        raise ValueError('unknown theta_initial')

    return obj_fn, ansatz_, theta_initial_, backend, initial_layout


### TESTS
def build_ansatz_he(ansatz: str, num_qubits: int, backend: BackendV2) -> tuple[QuantumCircuit, dict | None]:

    def apply_entanglement_map(entanglement):
        # custom-constructed
        if entanglement == 'bilinear':
            return [[i, i+1] for i in range(0, num_qubits-1, 2)] + [[i, i+1] for i in range(1, num_qubits-1, 2)], None

        if entanglement == 'color':
            assert backend.coupling_map.is_symmetric, 'Non-sym coupling map. Do you mean `di-color`?'
            coupling_map = [(i,j) for i,j in backend.coupling_map if i<j]
        elif entanglement == 'di-color':
            coupling_map = list(backend.coupling_map)
            assert not backend.coupling_map.is_symmetric, 'Sym coupling map. Do you mean `color`?'
        if entanglement in ['color', 'di-color']:
            nodes = list(range(backend.num_qubits))
            # TODO: this could be smarter
            for _ in range(num_qubits, backend.num_qubits):
                node_degrees = {n: sum(i == n or j == n for i,j in coupling_map) for n in nodes}
                remove_node = min(node_degrees.keys(), key=(lambda k: node_degrees[k]))
                nodes.remove(remove_node)
                coupling_map = [(i,j) for i,j in coupling_map if i != remove_node and j != remove_node]

            from qiskit_addon_utils.coloring import auto_color_edges
            coloring: dict = auto_color_edges(coupling_map)
            num_colors = max(v for _,v in coloring.items())+1

            out_map = [k for c1 in range(num_colors) for k, c in coloring.items() if c==c1]
            initial_layout = {idx: node for idx, node in enumerate(nodes)}
            reverse_layout = {node: idx for idx, node in enumerate(nodes)}
            return [(reverse_layout[i], reverse_layout[j]) for i,j in out_map], initial_layout
        
        # default
        return entanglement, None
    
    print(f'The ansatz from doe file = {ansatz}')
    if ansatz == 'TwoLocal':
        print('Using TwoLocal ansatz\n')
        ansatz_params_ = {'rotation_blocks':'ry', 'entanglement_blocks':'cz', 'entanglement': 'bilinear', 'reps': 1}
        ansatz_params_.update(ansatz_params)
        ansatz_params_['entanglement'], initial_layout = apply_entanglement_map(ansatz_params_['entanglement'])
        ansatz_ = TwoLocal(num_qubits, **ansatz_params_)
        ansatz_.measure_all()
    
    elif (ansatz == 'bfcd') or (ansatz == 'bfcdR'):
        print('Using bfcd/bfcdR ansatz\n')
        # bfcd -> uses 2 parameters (for 2 RZZ gates)
        # bfcdR -> uses 1 parameter (shared by both RZZ gates)
        # ParameterVector creates symbolic variables p[0], p[1] to be optimized later.
        qc_params = ParameterVector(name='p',length=2 if ansatz == 'bfcd' else 1)
        entanglement_block = QuantumCircuit(2)
        # Rzy
        entanglement_block.rx(np.pi / 2, 0)
        entanglement_block.rzz(qc_params[0], 0, 1)
        entanglement_block.rx(- np.pi / 2, 0)
        # Ryz
        entanglement_block.rx(np.pi / 2, 1)
        entanglement_block.rzz(qc_params[1] if ansatz == 'bfcd' else qc_params[0], 0, 1)
        entanglement_block.rx(- np.pi / 2, 1)

        ansatz_params_ = {'rotation_blocks': 'ry',
                          'entanglement_blocks':entanglement_block,
                          'overwrite_block_parameters': True,
                          'flatten': True,
                          'entanglement': 'bilinear',
                          'reps': 1,
                          'skip_final_rotation_layer': True,
                        #   'insert_barriers': True,
                          }
        ansatz_params_.update(ansatz_params)
        if isinstance(ansatz_params_['rotation_blocks'], str):
            ansatz_params_['rotation_blocks'] = get_standard_gate_name_mapping()[ansatz_params_['rotation_blocks']]
        ansatz_params_['entanglement'], initial_layout = apply_entanglement_map(ansatz_params_['entanglement'])
        ansatz_ = NLocal(num_qubits, **ansatz_params_)
        ansatz_.measure_all()

    elif ansatz == 'he':
        print('Using HE ansatz\n')
        print(f"Num qubits = {num_qubits}\n")
        total_qubits = 2 * num_qubits  # problem + bath
        iswap_block = make_iswap_like_block()

        # Entangler map: connect problem[i] to bath[i]
        entangler_map = [[i, num_qubits + i] for i in range(num_qubits)]

        # # Initial bath state |1⟩
        # init = QuantumCircuit(total_qubits)
        # for i in range(num_qubits, total_qubits):
        #     init.x(i)

        # === Step 1: Prepare Initial State Circuit ===
        init_circ = QuantumCircuit(total_qubits)

        # Set bath qubits (second half) to |1⟩
        for i in range(num_qubits, total_qubits):
            init_circ.x(i)

        # Build HE ansatz
        heansatz_ = n_local(
            num_qubits=total_qubits,
            rotation_blocks=[],
            entanglement_blocks=iswap_block.to_gate(),
            entanglement=entangler_map,
            reps=1,
            skip_final_rotation_layer=True,
            overwrite_block_parameters=True,
            parameter_prefix='theta'
        )
        ansatz_ = init_circ.compose(heansatz_)
        # ansatz_.measure_all()

        initial_layout = None

    else:
        raise ValueError('unknown ansatz')
    
    if initial_layout is not None:
        initial_layout = {ansatz_.qubits[k]: v for k,v in initial_layout.items()}
    
    return ansatz_, initial_layout

def problem_mapping_he(ansatz: str, ansatz_params: dict, theta_initial: str, device: str, instance: str):

    ### TODO
    num_vars = 15

    backend = get_backend(device, instance, num_vars)
    if 'from_backend' in ansatz_params:
        build_backend = get_backend(ansatz_params['from_backend'], instance, num_vars)
    else:
        build_backend = backend
    # ansatz_params_ = ansatz_params.copy()
    # ansatz_params_.pop('from_backend', None)
    # ansatz_params_.pop('discard_initial_layout', None)
    ansatz_, initial_layout = build_ansatz_he(ansatz, num_vars, build_backend)
    # if ansatz_params.get('discard_initial_layout', False):
        # initial_layout = None

    if theta_initial == 'piby3':
        print(f"Theta initial: piby3\n")
        theta_initial_ = np.pi/3 * np.ones(ansatz_.num_parameters)
        print(len(theta_initial_),ansatz_.num_parameters, num_vars)
    elif theta_initial == 'random':
        theta_initial_ = get_or_create_theta(ansatz, num_vars, ansatz_.num_parameters, seed=42)
    else:
        raise ValueError('unknown theta_initial')

    return ansatz_, theta_initial_, backend, initial_layout
