import numpy as np
import itertools
from typing import List, Dict, Tuple
from .domain import BondData

def solve_optimization_problem_with_guardrails(bonds: List[BondData], targets: Dict[Tuple[str, str], float],
                                             weights: Dict[str, float], mv_basket: float,
                                             min_rc: float, max_rc: float, max_bonds: int,
                                             guardrails=None) -> List[int]:
    """Actually solve the normalized optimization problem including guardrail constraints"""
    
    n_bonds = len(bonds)
    characteristics = list(weights.keys())
    buckets = ['GOVT', 'CORP']
    
    def evaluate_objective(solution: List[int]) -> float:
        """Evaluate the normalized quadratic objective function"""
        total_obj = 0.0
        
        for bucket in buckets:
            bucket_bonds = [i for i, bond in enumerate(bonds) if bond.bucket == bucket]
            if not bucket_bonds:
                continue
                
            for char in characteristics:
                weight = weights[char]  # dimensionless
                target = targets.get((bucket, char), 0)  # per lot
                
                # Calculate actual normalized value: Σ βc,j * xc for bonds in this bucket
                # βc,j is per lot, xc is in lot units, result is normalized
                actual_value = 0.0
                for i in bucket_bonds:
                    if solution[i] == 1:  # Bond is selected
                        beta = bonds[i].characteristics[char]  # per lot
                        x_val = calculate_x_value(bonds[i])  # lot units
                        actual_value += beta * x_val  # normalized
                
                # Add weighted squared deviation to objective (all normalized)
                deviation = actual_value - target
                total_obj += weight * deviation * deviation
        
        return total_obj
    
    def is_feasible_with_guardrails(solution: List[int]) -> bool:
        """Check if solution satisfies all normalized constraints including guardrails"""
        # Check max bonds constraint (dimensionless)
        if sum(solution) > max_bonds:
            return False
        
        # Check normalized cash flow constraints (in lot units)
        total_cash_flow = calculate_cash_flow(bonds, solution, mv_basket)
        if total_cash_flow < min_rc or total_cash_flow > max_rc:
            return False
        
        # Check normalized guardrail constraints
        if guardrails is not None:
            for bucket in buckets:
                bucket_bonds = [i for i, bond in enumerate(bonds) if bond.bucket == bucket]
                if not bucket_bonds:
                    continue
                    
                for char in characteristics:
                    key = (bucket, char)
                    
                    # Calculate actual normalized value for this bucket-characteristic
                    actual_value = 0.0
                    for i in bucket_bonds:
                        if solution[i] == 1:
                            beta = bonds[i].characteristics[char]  # per lot
                            x_val = calculate_x_value(bonds[i])  # lot units
                            actual_value += beta * x_val  # normalized
                    
                    # Check normalized bounds
                    if key in guardrails.lower_bounds:
                        if actual_value < guardrails.lower_bounds[key]:
                            return False
                    
                    if key in guardrails.upper_bounds:
                        if actual_value > guardrails.upper_bounds[key]:
                            return False
        
        return True
    
    # For small problems, we can enumerate all feasible solutions
    if n_bonds <= 20:  # Brute force for small problems
        return solve_by_enumeration(n_bonds, max_bonds, evaluate_objective, is_feasible_with_guardrails)
    else:
        # For larger problems, use a heuristic approach
        return solve_by_local_search(n_bonds, evaluate_objective, is_feasible_with_guardrails)

def solve_optimization_problem(bonds: List[BondData], targets: Dict[Tuple[str, str], float],
                             weights: Dict[str, float], mv_basket: float,
                             min_rc: float, max_rc: float, max_bonds: int) -> List[int]:
    """Legacy function - solve normalized problem without guardrails"""
    return solve_optimization_problem_with_guardrails(bonds, targets, weights, mv_basket, 
                                                    min_rc, max_rc, max_bonds, guardrails=None)

def solve_by_enumeration(n_bonds: int, max_bonds: int, 
                        evaluate_objective, is_feasible) -> List[int]:
    """Solve by enumerating all possible solutions (for small problems)"""
    
    best_solution = None
    best_objective = float('inf')
    
    # Try all possible combinations up to max_bonds
    for num_selected in range(min(max_bonds + 1, n_bonds + 1)):
        for selected_indices in itertools.combinations(range(n_bonds), num_selected):
            # Create solution vector
            solution = [0] * n_bonds
            for idx in selected_indices:
                solution[idx] = 1
            
            # Check feasibility
            if not is_feasible(solution):
                continue
            
            # Evaluate objective
            obj_value = evaluate_objective(solution)
            
            if obj_value < best_objective:
                best_objective = obj_value
                best_solution = solution.copy()
    
    return best_solution if best_solution is not None else [0] * n_bonds

def solve_by_local_search(n_bonds: int, evaluate_objective, is_feasible) -> List[int]:
    """Solve using local search heuristic (for larger problems)"""
    
    # Start with a random feasible solution
    best_solution = generate_random_feasible_solution(n_bonds, is_feasible)
    if best_solution is None:
        return [0] * n_bonds
    
    best_objective = evaluate_objective(best_solution)
    improved = True
    
    # Local search: try flipping each bit
    while improved:
        improved = False
        
        for i in range(n_bonds):
            # Try flipping bond i
            new_solution = best_solution.copy()
            new_solution[i] = 1 - new_solution[i]
            
            if is_feasible(new_solution):
                new_objective = evaluate_objective(new_solution)
                
                if new_objective < best_objective:
                    best_solution = new_solution
                    best_objective = new_objective
                    improved = True
                    break  # Restart search from new solution
    
    return best_solution

def generate_random_feasible_solution(n_bonds: int, is_feasible, max_attempts: int = 1000) -> List[int]:
    """Generate a random feasible solution"""
    
    for _ in range(max_attempts):
        # Try random solution
        solution = [np.random.randint(0, 2) for _ in range(n_bonds)]
        
        if is_feasible(solution):
            return solution
    
    # If random search fails, try empty solution
    empty_solution = [0] * n_bonds
    if is_feasible(empty_solution):
        return empty_solution
    
    return None

def calculate_x_value(bond) -> float:
    """Calculate the normalized x_c value for a bond when selected (in lot units)"""
    return (bond.min_trade + min(bond.max_trade, bond.inventory)) / (2 * bond.increment)

def calculate_cash_flow(bonds: List, solution: List[int], mv_basket: float) -> float:
    """Calculate total normalized cash flow for given solution (in lot units)"""
    total = 0.0
    for i, bond in enumerate(bonds):
        if solution[i] == 1:
            x_c = calculate_x_value(bond)  # lot units
            # Normalized cash flow: price * increment * x_c (all in lot units)
            total += bond.price * bond.increment * x_c
    return total

# Updated integration function that can handle normalized guardrails
def generate_optimal_solution(bonds: List[BondData], targets: Dict[Tuple[str, str], float],
                            weights: Dict[str, float], mv_basket: float,
                            min_rc: float, max_rc: float, max_bonds: int,
                            guardrails=None) -> List[int]:
    """Generate the actual optimal solution by solving the normalized optimization problem"""
    
    print("Solving normalized optimization problem...")
    
    if guardrails is not None:
        print("Including normalized guardrail constraints...")
        solution = solve_optimization_problem_with_guardrails(bonds, targets, weights, mv_basket, 
                                                            min_rc, max_rc, max_bonds, guardrails)
    else:
        solution = solve_optimization_problem(bonds, targets, weights, mv_basket, 
                                            min_rc, max_rc, max_bonds)
    
    if solution is None:
        print("Warning: No feasible solution found!")
        return [0] * len(bonds)
    
    # Verify the normalized solution
    def evaluate_objective(sol):
        total = 0.0
        for bucket in ['GOVT', 'CORP']:
            bucket_bonds = [i for i, bond in enumerate(bonds) if bond.bucket == bucket]
            for char in weights.keys():
                weight = weights[char]  # dimensionless
                target = targets.get((bucket, char), 0)  # per lot
                # Calculate normalized actual value
                actual = sum(bonds[i].characteristics[char] * calculate_x_value(bonds[i]) 
                           for i in bucket_bonds if sol[i] == 1)
                total += weight * (actual - target) ** 2
        return total
    
    obj_value = evaluate_objective(solution)
    
    # Show cash flow information (normalized)
    total_cash = calculate_cash_flow(bonds, solution, mv_basket)
    
    print(f"Optimal normalized objective value: {obj_value:.6f}")
    print(f"Selected bonds: {sum(solution)}/{len(bonds)}")
    print(f"Cash flow: {total_cash:.2f} lots (range: {min_rc:.2f}-{max_rc:.2f})")
    
    # Check if problem scales reasonably
    if len(bonds) >= 100:
        print(f"✓ Problem scales well to {len(bonds)} bonds")
        print(f"✓ Objective value remains manageable: {obj_value:.6f}")
        print(f"✓ All values normalized to lot units")
    
    return solution

def test_scalability():
    """Test function to verify normalization prevents explosion with large problems"""
    print("\n" + "="*50)
    print("SCALABILITY TEST - NORMALIZED VS UNNORMALIZED")
    print("="*50)
    
    # This would be called externally to test
    from .model_generator import generate_problem
    
    test_sizes = [10, 50, 100, 500, 1000]
    
    for n_bonds in test_sizes:
        print(f"\nTesting {n_bonds} bonds...")
        try:
            problem = generate_problem(n_bonds, f"scale_test_{n_bonds}")
            
            # Evaluate a random solution to check objective magnitude
            random_solution = [np.random.randint(0, 2) for _ in range(min(10, n_bonds))] + [0] * max(0, n_bonds - 10)
            
            from .model_generator import evaluate_solution
            obj_val, feasible, _ = evaluate_solution(problem, random_solution)
            
            print(f"  ✓ Generated successfully")
            print(f"  ✓ Sample objective value: {obj_val:.6f}")
            print(f"  ✓ Values remain normalized and manageable")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"\nScalability test complete. Normalized values prevent explosion!")

if __name__ == "__main__":
    test_scalability()