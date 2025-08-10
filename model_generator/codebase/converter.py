"""
Normalized Docplex Model Converter

This module converts normalized ProblemInstance objects to docplex models
and handles saving/solving with CPLEX. All values are normalized to lot units.
"""

from docplex.mp.model import Model
from typing import List, Dict, Tuple
from .domain import ProblemInstance
import pandas as pd

def calculate_x_value(bond) -> float:
    """Calculate the normalized x_c value for a bond when selected (in lot units)"""
    return (bond.min_trade + min(bond.max_trade, bond.inventory)) / (2 * bond.increment)

def create_docplex_model(problem, hash_code: str) -> Model:
    """
    Convert normalized ProblemInstance to a docplex Model including guardrail constraints
    
    Args:
        problem: ProblemInstance object with normalized values
        hash_code: String identifier for the problem
        
    Returns:
        docplex Model object ready to solve
    """
    
    # Create the model
    model = Model(name=f"NormalizedBondPortfolio_{hash_code}")
    
    n_bonds = len(problem.bonds)
    characteristics = list(problem.weights.keys())
    buckets = ['GOVT', 'CORP']
    
    # Decision variables - binary variables for each bond
    y = model.binary_var_list(n_bonds, name="y")
    
    # Pre-calculate normalized x_values for each bond (these are constants in lot units)
    x_values = [calculate_x_value(bond) for bond in problem.bonds]
    
    # Build the quadratic objective function (all normalized)
    objective_expr = 0
    
    for bucket in buckets:
        # Get bonds in this bucket
        bucket_bonds = [i for i, bond in enumerate(problem.bonds) if bond.bucket == bucket]
        if not bucket_bonds:
            continue
            
        for char in characteristics:
            weight = problem.weights[char]  # dimensionless
            target = problem.targets.get((bucket, char), 0)  # per lot
            
            # Build the sum: Σ(beta_c * x_c * y_c) for bonds in this bucket
            # beta_c is per lot, x_c is in lot units, result is normalized
            actual_sum = model.sum(
                problem.bonds[i].characteristics[char] * x_values[i] * y[i] 
                for i in bucket_bonds
            )
            
            # Add weighted squared deviation: weight * (actual_sum - target)^2
            # All terms are normalized, so objective remains manageable
            deviation = actual_sum - target
            objective_expr += weight * deviation * deviation
    
    # Set the objective
    model.minimize(objective_expr)
    
    # Constraints
    
    # 1. Maximum number of bonds constraint (dimensionless)
    model.add_constraint(
        model.sum(y[i] for i in range(n_bonds)) <= problem.max_bonds,
        ctname="max_bonds"
    )
    
    # 2. Normalized cash flow constraints (in lot units)
    # price * increment * x_value * y (all normalized to lot units)
    cash_flow_expr = model.sum(
        problem.bonds[i].price * problem.bonds[i].increment * x_values[i] * y[i]
        for i in range(n_bonds)
    )
    
    model.add_constraint(
        cash_flow_expr >= problem.min_rc,
        ctname="min_cash"
    )
    
    model.add_constraint(
        cash_flow_expr <= problem.max_rc,
        ctname="max_cash"
    )
    
    # 3. Normalized value-based guardrail constraints
    if hasattr(problem, 'guardrails') and problem.guardrails is not None:
        guardrail_count = 0
        
        for bucket in buckets:
            bucket_bonds = [i for i, bond in enumerate(problem.bonds) if bond.bucket == bucket]
            if not bucket_bonds:
                continue
                
            for char in characteristics:
                key = (bucket, char)
                
                # Build the expression for this bucket-characteristic combination
                # All terms normalized: beta_c (per lot) * x_c (lot units) * y
                guardrail_expr = model.sum(
                    problem.bonds[i].characteristics[char] * x_values[i] * y[i]
                    for i in bucket_bonds
                )
                
                # Add lower bound constraint if it exists (normalized bounds)
                if key in problem.guardrails.lower_bounds:
                    lower_bound = problem.guardrails.lower_bounds[key]
                    model.add_constraint(
                        guardrail_expr >= lower_bound,
                        ctname=f"guardrail_lower_{bucket}_{char}"
                    )
                    guardrail_count += 1
                
                # Add upper bound constraint if it exists (normalized bounds)
                if key in problem.guardrails.upper_bounds:
                    upper_bound = problem.guardrails.upper_bounds[key]
                    model.add_constraint(
                        guardrail_expr <= upper_bound,
                        ctname=f"guardrail_upper_{bucket}_{char}"
                    )
                    guardrail_count += 1
        
        print(f"Added {guardrail_count} normalized guardrail constraints to the model")
    
    return model

def solve_with_docplex(problem, hash_code: str, save_lp: bool = True, verbose: bool = True):
    """
    Create and solve the normalized optimization problem using docplex
    
    Args:
        problem: ProblemInstance object with normalized values
        hash_code: String identifier
        save_lp: Whether to save the LP file
        verbose: Whether to print solving information
        
    Returns:
        Tuple of (solution_vector, objective_value)
    """
    
    # Create the model
    model = create_docplex_model(problem, hash_code)
    
    # Save LP file if requested
    if save_lp:
        lp_filename = f"normalized_problem_{hash_code}.lp"
        model.export_as_lp(lp_filename)
        if verbose:
            print(f"Normalized model exported to {lp_filename}")
    
    # Solve the model
    if verbose:
        print("Solving normalized problem with CPLEX...")
    
    solution = model.solve()
    
    if solution is None:
        if verbose:
            print("No solution found!")
        return None, None
    
    # Extract solution vector
    n_bonds = len(problem.bonds)
    solution_vector = [int(solution.get_value(f"y_{i}")) for i in range(n_bonds)]
    objective_value = solution.get_objective_value()
    
    if verbose:
        print(f"Optimal normalized solution found!")
        print(f"Solution: {solution_vector}")
        print(f"Objective value: {objective_value:.6f}")
        print(f"Bonds selected: {sum(solution_vector)}")
        
        # Show cash flow information (normalized)
        total_cash = sum(
            problem.bonds[i].price * problem.bonds[i].increment * calculate_x_value(problem.bonds[i])
            for i in range(n_bonds) if solution_vector[i] == 1
        )
        print(f"Total cash flow: {total_cash:.2f} lots (range: {problem.min_rc:.2f}-{problem.max_rc:.2f})")
    
    return solution_vector, objective_value

def compare_solutions(problem, hash_code: str, python_solution: List[int], verbose: bool = True):
    """
    Compare Python solution with docplex solution for normalized problems
    
    Args:
        problem: ProblemInstance object with normalized values
        hash_code: String identifier
        python_solution: Solution from Python solver
        verbose: Whether to print comparison details
        
    Returns:
        Dict with comparison results
    """
    
    # Get docplex solution
    docplex_solution, docplex_obj = solve_with_docplex(problem, hash_code, save_lp=True, verbose=False)
    
    if docplex_solution is None:
        return {"error": "Docplex could not find solution"}
    
    # Evaluate Python solution using our normalized evaluation function
    python_obj_docplex = None
    try:
        # Import the updated evaluation function
        if hasattr(problem, 'guardrails'):
            from .model_generator import evaluate_solution_with_guardrails 
            python_obj_docplex, python_feasible, _ = evaluate_solution_with_guardrails(problem, python_solution)
        else:
            from .model_generator import evaluate_solution 
            python_obj_docplex, python_feasible, _ = evaluate_solution(problem, python_solution)
    except:
        python_obj_docplex = "Could not evaluate"
        python_feasible = "Unknown"
    
    results = {
        "python_solution": python_solution,
        "docplex_solution": docplex_solution,
        "python_objective": python_obj_docplex,
        "docplex_objective": docplex_obj,
        "solutions_match": python_solution == docplex_solution,
        "objectives_match": abs(python_obj_docplex - docplex_obj) < 1e-6 if isinstance(python_obj_docplex, (int, float)) else False
    }
    
    if verbose:
        print("\n" + "="*60)
        print("NORMALIZED SOLUTION COMPARISON")
        print("="*60)
        print(f"Python solution:  {python_solution}")
        print(f"Docplex solution: {docplex_solution}")
        print(f"Solutions match:  {results['solutions_match']}")
        print(f"Python objective: {python_obj_docplex}")
        print(f"Docplex objective: {docplex_obj}")
        print(f"Objectives match: {results['objectives_match']}")
        
        if not results['solutions_match']:
            print("\nDifferences found - investigating...")
            diff_indices = [i for i in range(len(python_solution)) if python_solution[i] != docplex_solution[i]]
            print(f"Different at indices: {diff_indices}")
            
            # Show impact of differences (normalized values)
            python_cash = sum(
                problem.bonds[i].price * problem.bonds[i].increment * calculate_x_value(problem.bonds[i])
                for i in range(len(python_solution)) if python_solution[i] == 1
            )
            docplex_cash = sum(
                problem.bonds[i].price * problem.bonds[i].increment * calculate_x_value(problem.bonds[i])
                for i in range(len(docplex_solution)) if docplex_solution[i] == 1
            )
            print(f"Python cash flow: {python_cash:.2f} lots")
            print(f"Docplex cash flow: {docplex_cash:.2f} lots")
    
    return results

# Convenience function to integrate with existing code
def save_to_docplex(problem, hash_code: str):
    """
    Save normalized problem as LP file using docplex
    
    This function matches the signature of your existing save_to_cplex functions
    """
    model = create_docplex_model(problem, hash_code)
    lp_filename = f"normalized_problem_{hash_code}.lp"
    model.export_as_lp(lp_filename)
    print(f"Normalized docplex model saved to {lp_filename}")
    
    return model

def save_to_excel(problem: ProblemInstance, hash_code: str):
    """Save the normalized problem instance to an Excel file"""
    filename = f"normalized_model_{hash_code}.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Bond data (all normalized)
        bond_data = []
        for i, bond in enumerate(problem.bonds):
            row = {
                'bond_id': i,
                'price': bond.price,  # Normalized around 1.0
                'min_trade': bond.min_trade,  # In lot units
                'max_trade': bond.max_trade,  # In lot units
                'inventory': bond.inventory,  # In lot units
                'increment': bond.increment,  # In lot units
                'bucket': bond.bucket,
                'solution': problem.solution[i],
                'x_value': calculate_x_value(bond)  # Calculated lot units
            }
            # Add normalized characteristics (per lot)
            for char, value in bond.characteristics.items():
                row[f'beta_{char}_per_lot'] = value
            bond_data.append(row)
        
        df_bonds = pd.DataFrame(bond_data)
        df_bonds.to_excel(writer, sheet_name='Normalized_Bonds', index=False)
        
        # Targets (normalized per lot)
        target_data = []
        for (bucket, char), target in problem.targets.items():
            target_data.append({
                'bucket': bucket,
                'characteristic': char,
                'target_per_lot': target
            })
        df_targets = pd.DataFrame(target_data)
        df_targets.to_excel(writer, sheet_name='Normalized_Targets', index=False)
        
        # Weights (dimensionless)
        weight_data = [{'characteristic': char, 'weight_dimensionless': weight} 
                      for char, weight in problem.weights.items()]
        df_weights = pd.DataFrame(weight_data)
        df_weights.to_excel(writer, sheet_name='Weights', index=False)
        
        # Global parameters (normalized)
        global_params = pd.DataFrame([
            {'parameter': 'mv_basket_lots', 'value': problem.mv_basket},
            {'parameter': 'min_rc_lots', 'value': problem.min_rc},
            {'parameter': 'max_rc_lots', 'value': problem.max_rc},
            {'parameter': 'max_bonds', 'value': problem.max_bonds},
            {'parameter': 'hash_code', 'value': hash_code},
            {'parameter': 'normalization_note', 'value': 'All values in lot units, prices around 1.0'}
        ])
        global_params.to_excel(writer, sheet_name='Normalized_Parameters', index=False)
        
        # Guardrails if present (normalized)
        if hasattr(problem, 'guardrails') and problem.guardrails is not None:
            guardrail_data = []
            for (bucket, char), lower in problem.guardrails.lower_bounds.items():
                guardrail_data.append({
                    'bucket': bucket,
                    'characteristic': char,
                    'lower_bound_per_lot': lower,
                    'upper_bound_per_lot': problem.guardrails.upper_bounds.get((bucket, char), 'None')
                })
            df_guardrails = pd.DataFrame(guardrail_data)
            df_guardrails.to_excel(writer, sheet_name='Normalized_Guardrails', index=False)
    
    print(f"Normalized problem saved to {filename}")
    print("Note: All values are normalized to lot units for scalability")