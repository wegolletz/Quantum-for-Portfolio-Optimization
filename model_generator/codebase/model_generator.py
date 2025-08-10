import numpy as np
import hashlib
import random
import pandas as pd
from .solvers import generate_optimal_solution
from .domain import BondData, ProblemInstance
from typing import List, Tuple, Dict
from .converter import save_to_docplex

def set_random_seed(hash_code: str):
    """Set random seed based on hash code for reproducibility"""
    seed = int(hashlib.md5(hash_code.encode()).hexdigest()[:8], 16) % (2**31)
    np.random.seed(seed)
    random.seed(seed)

def generate_problem(n_bonds: int, hash_code: str, excel: bool = False, cplex: bool = False) -> ProblemInstance:
    """Generate a normalized bond portfolio optimization problem"""
    set_random_seed(hash_code)
    
    # Constants - all normalized to lot-based units
    characteristics = ['duration', 'credit_risk', 'liquidity']
    buckets = ['GOVT', 'CORP']
    mv_basket = 100.0  # 100 lots as base portfolio size
    
    # Generate bonds with normalized values
    bonds = []
    for i in range(n_bonds):
        bucket = random.choice(buckets)
        
        # Normalized prices around par (1.0 = par)
        price = np.random.uniform(0.95, 1.05)
        
        # Trade sizes in lot units (small, manageable values)
        min_trade = np.random.uniform(0.5, 2.0)  # 0.5-2.0 lots
        max_trade = min_trade * np.random.uniform(1.5, 4.0)  # 0.75-8.0 lots max
        inventory = max_trade * np.random.uniform(0.5, 1.5)  # Available inventory
        increment = np.random.uniform(0.1, 0.5)  # Small lot increments
        
        # Generate normalized characteristics per lot
        char_values = {}
        for char in characteristics:
            if char == 'duration':
                # Duration contribution per lot (years per lot)
                char_values[char] = np.random.uniform(0.1, 1.5)
            elif char == 'credit_risk':
                # Risk contribution per lot (risk units per lot)
                char_values[char] = np.random.uniform(0.0, 0.1)
            else:  # liquidity
                # Liquidity contribution per lot (liquidity units per lot)
                char_values[char] = np.random.uniform(0.01, 0.1)
        
        bonds.append(BondData(
            price=price,
            min_trade=min_trade,
            max_trade=max_trade,
            inventory=inventory,
            increment=increment,
            bucket=bucket,
            characteristics=char_values
        ))
    
    # Generate normalized weights (dimensionless scaling factors)
    weights = {char: np.random.uniform(0.1, 2.0) for char in characteristics}
    
    # Generate normalized targets per lot
    targets = {}
    for bucket in buckets:
        for char in characteristics:
            if char == 'duration':
                # Target duration per lot (reasonable portfolio average)
                targets[(bucket, char)] = np.random.uniform(0.3, 1.0)
            elif char == 'credit_risk':
                # Target risk per lot
                targets[(bucket, char)] = np.random.uniform(0.02, 0.08)
            else:  # liquidity
                # Target liquidity per lot
                targets[(bucket, char)] = np.random.uniform(0.03, 0.07)
    
    # Generate normalized cash flow constraints (in lot units)
    min_rc = mv_basket * np.random.uniform(0.02, 0.05)  # 2-5 lots
    max_rc = mv_basket * np.random.uniform(0.08, 0.12)  # 8-12 lots
    
    # Maximum number of bonds constraint (scales with problem size)
    max_bonds = max(3, int(n_bonds * np.random.uniform(0.3, 0.8)))
    
    # Generate a feasible solution by solving a simplified version
    solution = generate_optimal_solution(bonds, targets, weights, mv_basket, 
                                        min_rc, max_rc, max_bonds)
    
    problem = ProblemInstance(
        bonds=bonds,
        targets=targets,
        weights=weights,
        mv_basket=mv_basket,
        min_rc=min_rc,
        max_rc=max_rc,
        max_bonds=max_bonds,
        solution=solution
    )
    
    # Save to files if requested
    if excel:
        save_to_excel(problem, hash_code)
    
    if cplex:
        save_to_docplex(problem, hash_code)
    
    return problem

def calculate_cash_flow(bonds: List[BondData], solution: List[int], mv_basket: float) -> float:
    """Calculate total cash flow for given solution - normalized to lot units"""
    total = 0.0
    for i, bond in enumerate(bonds):
        if solution[i] == 1:
            # x_c normalized to lot units
            x_c = (bond.min_trade + min(bond.max_trade, bond.inventory)) / (2 * bond.increment)
            # Cash flow per lot: price * increment * x_c (all in lot units)
            total += bond.price * bond.increment * x_c
    return total

def print_problem_summary(problem: ProblemInstance):
    """Print a summary of the generated problem"""
    print(f"Generated normalized problem with {len(problem.bonds)} bonds")
    print(f"Solution: {problem.solution}")
    print(f"Number of selected bonds: {sum(problem.solution)}")
    print(f"Max allowed bonds: {problem.max_bonds}")
    print(f"Basket market value: {problem.mv_basket:.1f} lots")
    print(f"Cash flow range: {problem.min_rc:.2f} - {problem.max_rc:.2f} lots")
    
    # Show bucket distribution
    govt_bonds = sum(1 for bond in problem.bonds if bond.bucket == 'GOVT')
    corp_bonds = len(problem.bonds) - govt_bonds
    print(f"Bond distribution: {govt_bonds} GOVT, {corp_bonds} CORP")
    
    # Show selected bonds by bucket
    selected_govt = sum(1 for i, bond in enumerate(problem.bonds) 
                       if problem.solution[i] == 1 and bond.bucket == 'GOVT')
    selected_corp = sum(problem.solution) - selected_govt
    print(f"Selected distribution: {selected_govt} GOVT, {selected_corp} CORP")

def save_to_excel(problem: ProblemInstance, hash_code: str):
    """Save the normalized problem instance to an Excel file"""
    filename = f"model_{hash_code}.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Bond data
        bond_data = []
        for i, bond in enumerate(problem.bonds):
            row = {
                'bond_id': i,
                'price': bond.price,
                'min_trade': bond.min_trade,
                'max_trade': bond.max_trade,
                'inventory': bond.inventory,
                'increment': bond.increment,
                'bucket': bond.bucket,
                'solution': problem.solution[i]
            }
            # Add characteristics
            for char, value in bond.characteristics.items():
                row[f'beta_{char}'] = value
            bond_data.append(row)
        
        df_bonds = pd.DataFrame(bond_data)
        df_bonds.to_excel(writer, sheet_name='Bonds', index=False)
        
        # Targets
        target_data = []
        for (bucket, char), target in problem.targets.items():
            target_data.append({
                'bucket': bucket,
                'characteristic': char,
                'target': target
            })
        df_targets = pd.DataFrame(target_data)
        df_targets.to_excel(writer, sheet_name='Targets', index=False)
        
        # Weights
        weight_data = [{'characteristic': char, 'weight': weight} 
                      for char, weight in problem.weights.items()]
        df_weights = pd.DataFrame(weight_data)
        df_weights.to_excel(writer, sheet_name='Weights', index=False)
        
        # Global parameters
        global_params = pd.DataFrame([
            {'parameter': 'mv_basket', 'value': problem.mv_basket},
            {'parameter': 'min_rc', 'value': problem.min_rc},
            {'parameter': 'max_rc', 'value': problem.max_rc},
            {'parameter': 'max_bonds', 'value': problem.max_bonds},
            {'parameter': 'hash_code', 'value': hash_code}
        ])
        global_params.to_excel(writer, sheet_name='Parameters', index=False)
    
    print(f"Problem saved to {filename}")

def calculate_x_value(bond) -> float:
    """Calculate the normalized x_c value for a bond when selected (in lot units)"""
    return (bond.min_trade + min(bond.max_trade, bond.inventory)) / (2 * bond.increment)

def evaluate_solution(problem: ProblemInstance, solution: List[int]) -> Tuple[float, bool, Dict]:
    """
    Evaluate a proposed solution using normalized values
    
    Returns:
        - objective_value: The quadratic objective value (dimensionless)
        - is_feasible: Whether all constraints are satisfied
        - diagnostics: Dict with detailed constraint checking
    """
    n_bonds = len(problem.bonds)
    characteristics = list(problem.weights.keys())
    buckets = ['GOVT', 'CORP']
    
    # Evaluate objective function (all normalized)
    objective_value = 0.0
    obj_details = {}
    
    for bucket in buckets:
        bucket_bonds = [i for i, bond in enumerate(problem.bonds) if bond.bucket == bucket]
        if not bucket_bonds:
            continue
            
        for char in characteristics:
            weight = problem.weights[char]  # dimensionless
            target = problem.targets.get((bucket, char), 0)  # per lot
            
            # Calculate actual value for this bucket-characteristic (per lot)
            actual_value = 0.0
            for i in bucket_bonds:
                if solution[i] == 1:
                    beta = problem.bonds[i].characteristics[char]  # per lot
                    x_val = calculate_x_value(problem.bonds[i])  # lot units
                    actual_value += beta * x_val  # (per lot) * (lot units) = dimensionless
            
            # Calculate weighted squared deviation (dimensionless)
            deviation = actual_value - target
            weighted_sq_dev = weight * deviation * deviation
            objective_value += weighted_sq_dev
            
            obj_details[f"{bucket}_{char}"] = {
                'actual': actual_value,
                'target': target, 
                'deviation': deviation,
                'weight': weight,
                'contribution': weighted_sq_dev
            }
    
    # Check constraints
    diagnostics = {
        'objective_details': obj_details,
        'constraints': {}
    }
    
    # 1. Max bonds constraint (dimensionless)
    num_selected = sum(solution)
    max_bonds_ok = num_selected <= problem.max_bonds
    diagnostics['constraints']['max_bonds'] = {
        'satisfied': max_bonds_ok,
        'actual': num_selected,
        'limit': problem.max_bonds
    }
    
    # 2. Cash flow constraints (in lot units)
    total_cash_flow = 0.0
    for i, bond in enumerate(problem.bonds):
        if solution[i] == 1:
            x_val = calculate_x_value(bond)  # lot units
            # Cash contribution: price * increment * x_val (all normalized)
            cash_contribution = bond.price * bond.increment * x_val
            total_cash_flow += cash_contribution
    
    min_cash_ok = total_cash_flow >= problem.min_rc
    max_cash_ok = total_cash_flow <= problem.max_rc
    
    diagnostics['constraints']['cash_flow'] = {
        'satisfied': min_cash_ok and max_cash_ok,
        'actual': total_cash_flow,
        'min_required': problem.min_rc,
        'max_allowed': problem.max_rc,
        'min_ok': min_cash_ok,
        'max_ok': max_cash_ok
    }
    
    # Overall feasibility
    is_feasible = max_bonds_ok and min_cash_ok and max_cash_ok
    
    return objective_value, is_feasible, diagnostics

def evaluate_solution_with_guardrails(problem, solution: List[int]) -> Tuple[float, bool, Dict]:
    """
    Evaluate a proposed solution including normalized guardrail constraints
    
    Returns:
        - objective_value: The quadratic objective value (dimensionless)
        - is_feasible: Whether all constraints are satisfied
        - diagnostics: Dict with detailed constraint checking
    """
    n_bonds = len(problem.bonds)
    characteristics = list(problem.weights.keys())
    buckets = ['GOVT', 'CORP']
    
    # Evaluate objective function (same as before, all normalized)
    objective_value = 0.0
    obj_details = {}
    
    for bucket in buckets:
        bucket_bonds = [i for i, bond in enumerate(problem.bonds) if bond.bucket == bucket]
        if not bucket_bonds:
            continue
            
        for char in characteristics:
            weight = problem.weights[char]  # dimensionless
            target = problem.targets.get((bucket, char), 0)  # per lot
            
            # Calculate actual value for this bucket-characteristic (per lot)
            actual_value = 0.0
            for i in bucket_bonds:
                if solution[i] == 1:
                    beta = problem.bonds[i].characteristics[char]  # per lot
                    x_val = calculate_x_value(problem.bonds[i])  # lot units
                    actual_value += beta * x_val
            
            # Calculate weighted squared deviation (dimensionless)
            deviation = actual_value - target
            weighted_sq_dev = weight * deviation * deviation
            objective_value += weighted_sq_dev
            
            obj_details[f"{bucket}_{char}"] = {
                'actual': actual_value,
                'target': target, 
                'deviation': deviation,
                'weight': weight,
                'contribution': weighted_sq_dev
            }
    
    # Check constraints
    diagnostics = {
        'objective_details': obj_details,
        'constraints': {}
    }
    
    # 1. Max bonds constraint
    num_selected = sum(solution)
    max_bonds_ok = num_selected <= problem.max_bonds
    diagnostics['constraints']['max_bonds'] = {
        'satisfied': max_bonds_ok,
        'actual': num_selected,
        'limit': problem.max_bonds
    }
    
    # 2. Cash flow constraints (in lot units)
    total_cash_flow = 0.0
    for i, bond in enumerate(problem.bonds):
        if solution[i] == 1:
            x_val = calculate_x_value(bond)
            cash_contribution = bond.price * bond.increment * x_val
            total_cash_flow += cash_contribution
    
    min_cash_ok = total_cash_flow >= problem.min_rc
    max_cash_ok = total_cash_flow <= problem.max_rc
    
    diagnostics['constraints']['cash_flow'] = {
        'satisfied': min_cash_ok and max_cash_ok,
        'actual': total_cash_flow,
        'min_required': problem.min_rc,
        'max_allowed': problem.max_rc,
        'min_ok': min_cash_ok,
        'max_ok': max_cash_ok
    }
    
    # 3. Normalized guardrail constraints
    guardrail_violations = 0
    guardrail_details = {}
    
    if hasattr(problem, 'guardrails') and problem.guardrails is not None:
        for bucket in buckets:
            bucket_bonds = [i for i, bond in enumerate(problem.bonds) if bond.bucket == bucket]
            if not bucket_bonds:
                continue
                
            for char in characteristics:
                key = (bucket, char)
                
                if key in problem.guardrails.lower_bounds or key in problem.guardrails.upper_bounds:
                    # Calculate actual value for this bucket-characteristic (normalized)
                    actual_value = 0.0
                    for i in bucket_bonds:
                        if solution[i] == 1:
                            beta = problem.bonds[i].characteristics[char]  # per lot
                            x_val = calculate_x_value(problem.bonds[i])  # lot units
                            actual_value += beta * x_val
                    
                    # Check normalized bounds
                    lower_bound = problem.guardrails.lower_bounds.get(key, float('-inf'))
                    upper_bound = problem.guardrails.upper_bounds.get(key, float('inf'))
                    
                    lower_ok = actual_value >= lower_bound
                    upper_ok = actual_value <= upper_bound
                    satisfied = lower_ok and upper_ok
                    
                    if not satisfied:
                        guardrail_violations += 1
                    
                    guardrail_details[f"{bucket}_{char}_guardrail"] = {
                        'satisfied': satisfied,
                        'actual': actual_value,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'lower_ok': lower_ok,
                        'upper_ok': upper_ok
                    }
    
    diagnostics['constraints']['guardrails'] = {
        'satisfied': guardrail_violations == 0,
        'violations': guardrail_violations,
        'details': guardrail_details
    }
    
    # Overall feasibility (now includes guardrails)
    is_feasible = (max_bonds_ok and min_cash_ok and max_cash_ok and 
                   guardrail_violations == 0)
    
    return objective_value, is_feasible, diagnostics

# Example usage
if __name__ == "__main__":
    # Generate a normalized problem instance
    n_bonds = 6
    hash_code = "test_hash_123"
    
    # Generate problem and save to both Excel and CPLEX formats
    problem = generate_problem(n_bonds, hash_code, excel=True, cplex=True)
    print_problem_summary(problem)
    
    # Test with large problem to ensure scaling works
    large_problem = generate_problem(1000, "large_test", excel=False, cplex=False)
    print(f"\nLarge problem test - 1000 bonds generated successfully")
    print(f"Objective function remains manageable: {evaluate_solution(large_problem, large_problem.solution)[0]:.6f}")
