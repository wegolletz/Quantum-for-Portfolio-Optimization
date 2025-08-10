import numpy as np
import random
from typing import Dict, List, Tuple
from .solvers import generate_optimal_solution
from .converter import save_to_docplex, save_to_excel
from .domain import Guardrails

def generate_realistic_targets_and_params(bonds, n_bonds: int, max_bonds: int) -> Tuple[Dict, Dict, float, float, float]:
    """
    Generate realistic normalized targets scaled to lot contributions
    
    Returns:
        - targets: Dict mapping (bucket, characteristic) to realistic targets per lot
        - weights: Dict mapping characteristic to weights (dimensionless)
        - mv_basket: Basket market value in lots
        - min_rc: Minimum residual cash in lots
        - max_rc: Maximum residual cash in lots
    """
    
    characteristics = ['duration', 'credit_risk', 'liquidity']
    buckets = ['GOVT', 'CORP']
    
    # Calculate realistic ranges by simulating portfolio contributions (normalized)
    contribution_ranges = calculate_contribution_ranges(bonds, characteristics, buckets, max_bonds)
    
    # Generate targets within realistic ranges (all per lot)
    targets = {}
    for bucket in buckets:
        for char in characteristics:
            min_contrib, max_contrib = contribution_ranges[(bucket, char)]
            
            if max_contrib > min_contrib:
                # Target somewhere in the middle 20-80% of the feasible range
                target_range_start = min_contrib + 0.2 * (max_contrib - min_contrib)
                target_range_end = min_contrib + 0.8 * (max_contrib - min_contrib)
                targets[(bucket, char)] = np.random.uniform(target_range_start, target_range_end)
            else:
                # Fallback for edge cases
                targets[(bucket, char)] = min_contrib
    
    # Generate dimensionless weights that encourage multi-bond solutions
    weights = generate_balanced_weights(characteristics)
    
    # Generate normalized cash constraints (in lots)
    mv_basket = 100.0  # Standard 100-lot portfolio base
    min_rc, max_rc = generate_realistic_cash_constraints(bonds, max_bonds, mv_basket)
    
    return targets, weights, mv_basket, min_rc, max_rc

def generate_smart_guardrails(bonds, targets: Dict, weights: Dict, max_bonds: int, solution: List[int]) -> Guardrails:
    """
    Generate normalized value-based guardrails that create meaningful constraints
    
    All bounds are normalized to per-lot contributions to prevent scaling issues
    """
    
    characteristics = list(weights.keys())
    buckets = ['GOVT', 'CORP']
    
    lower_bounds = {}
    upper_bounds = {}
    
    for bucket in buckets:
        bucket_bonds = [i for i, bond in enumerate(bonds) if bond.bucket == bucket]
        
        if not bucket_bonds:
            # No bonds in this bucket - set trivial bounds
            for char in characteristics:
                lower_bounds[(bucket, char)] = 0.0
                upper_bounds[(bucket, char)] = 10.0  # Reasonable upper bound for normalized values
            continue
        
        for char in characteristics:
            # Calculate what the optimal solution achieves (normalized)
            optimal_value = 0.0
            for i in bucket_bonds:
                if solution[i] == 1:
                    beta = bonds[i].characteristics[char]  # per lot
                    x_val = calculate_x_value(bonds[i])  # lot units
                    optimal_value += beta * x_val  # normalized contribution
            
            # Sample achievable normalized values with NON-ZERO contributions only
            meaningful_values = sample_meaningful_characteristic_values(
                bonds, bucket_bonds, char, max_bonds
            )
            
            if len(meaningful_values) == 0:
                # Fallback - use optimal value with buffer
                buffer = max(0.2 * optimal_value, 0.1)  # Small normalized buffer
                lower_bounds[(bucket, char)] = max(0.1 * optimal_value, 0.01)  # Meaningful minimum
                upper_bounds[(bucket, char)] = optimal_value + buffer
                continue
            
            # Include optimal solution in our analysis
            all_values = meaningful_values + [optimal_value]
            all_values.sort()
            
            # Calculate statistics for MEANINGFUL values (excluding zeros)
            nonzero_values = [v for v in all_values if v > 1e-6]
            
            if len(nonzero_values) < 2:
                # Not enough variation - create reasonable normalized bounds around optimal
                lower_bounds[(bucket, char)] = max(0.5 * optimal_value, 0.01)
                upper_bounds[(bucket, char)] = 1.5 * optimal_value
                continue
            
            min_nonzero = min(nonzero_values)
            max_nonzero = max(nonzero_values)
            median_nonzero = np.median(nonzero_values)
            
            # Smart lower bound: ensure some meaningful contribution is required
            lower_percentile = np.random.uniform(0.15, 0.35)  # 15-35th percentile
            lower_target = np.percentile(nonzero_values, lower_percentile * 100)
            
            # But don't make it too restrictive - ensure optimal solution has some cushion
            safety_margin = 0.1 * optimal_value
            lower_bounds[(bucket, char)] = min(lower_target, optimal_value - safety_margin)
            
            # Make absolutely sure it's positive and meaningful (normalized)
            min_meaningful = calculate_minimum_meaningful_contribution(bonds, bucket_bonds, char)
            lower_bounds[(bucket, char)] = max(lower_bounds[(bucket, char)], min_meaningful)
            
            # Smart upper bound: constrain but don't eliminate too many solutions
            upper_percentile = np.random.uniform(0.75, 0.95)  # 75-95th percentile
            upper_target = np.percentile(nonzero_values, upper_percentile * 100)
            
            # Ensure optimal solution is feasible with margin
            safety_margin_upper = 0.15 * optimal_value
            upper_bounds[(bucket, char)] = max(upper_target, optimal_value + safety_margin_upper)
            
            print(f"Normalized Guardrail {bucket}-{char}: [{lower_bounds[(bucket, char)]:.3f}, {upper_bounds[(bucket, char)]:.3f}] (optimal: {optimal_value:.3f})")
    
    return Guardrails(lower_bounds=lower_bounds, upper_bounds=upper_bounds)

def generate_balanced_weights(characteristics: List[str]) -> Dict[str, float]:
    """
    Generate dimensionless weights that encourage balanced portfolios
    """
    
    # Base weights (dimensionless scaling factors)
    base_weights = {
        'duration': np.random.uniform(0.5, 2.0),
        'credit_risk': np.random.uniform(0.5, 2.0), 
        'liquidity': np.random.uniform(0.5, 2.0)
    }
    
    # Normalize so they're in similar ranges
    total_weight = sum(base_weights.values())
    target_total = 3.0  # Average weight of 1.0
    
    weights = {char: (weight / total_weight) * target_total for char, weight in base_weights.items()}
    
    return weights

def generate_realistic_cash_constraints(bonds, max_bonds: int, mv_basket: float) -> Tuple[float, float]:
    """
    Generate normalized cash constraints that encourage multiple bond selection (in lot units)
    """
    
    # Calculate cash contributions for different portfolio sizes (all normalized)
    cash_contributions = []
    
    for _ in range(1000):  # Sample many random portfolios
        num_selected = np.random.randint(1, max_bonds + 1)  # At least 1 bond
        selected_bonds = np.random.choice(len(bonds), size=num_selected, replace=False)
        
        total_cash = 0.0
        for i in selected_bonds:
            bond = bonds[i]
            x_val = calculate_x_value(bond)  # lot units
            # Normalized cash contribution: price * increment * x_val (all in lot units)
            cash_contrib = bond.price * bond.increment * x_val
            total_cash += cash_contrib
        
        cash_contributions.append(total_cash)
    
    # Set constraints based on distribution of cash flows (normalized)
    cash_contributions.sort()
    
    # Min cash: 10th percentile (allows smaller portfolios)
    min_rc = cash_contributions[int(0.1 * len(cash_contributions))]
    
    # Max cash: 90th percentile (allows larger portfolios) 
    max_rc = cash_contributions[int(0.9 * len(cash_contributions))]
    
    return min_rc, max_rc

def generate_more_interesting_bonds(n_bonds: int) -> List:
    """
    Generate bonds with normalized, diverse characteristics to create interesting trade-offs
    """
    from dataclasses import dataclass
    from typing import Dict
    
    @dataclass 
    class BondData:
        price: float
        min_trade: float
        max_trade: float
        inventory: float
        increment: float
        bucket: str
        characteristics: Dict[str, float]
    
    bonds = []
    buckets = ['GOVT', 'CORP']
    
    for i in range(n_bonds):
        bucket = random.choice(buckets)
        
        # Normalized price range around par
        price = np.random.uniform(0.90, 1.10)
        
        # Normalized trade sizes in lot units
        min_trade = np.random.uniform(0.5, 3.0)  # 0.5-3 lots
        max_trade = min_trade * np.random.uniform(1.5, 4.0)  # up to 12 lots
        inventory = max_trade * np.random.uniform(0.3, 2.0)
        increment = np.random.uniform(0.05, 0.5)  # Small increments
        
        # Create normalized characteristic combinations per lot
        if bucket == 'GOVT':
            # Government bonds: generally lower risk, diverse duration (per lot)
            duration = np.random.uniform(0.05, 2.0)  # Duration per lot
            credit_risk = np.random.uniform(0.0, 0.03)  # Low risk per lot
            liquidity = np.random.uniform(0.06, 0.10)  # High liquidity per lot
        else:
            # Corporate bonds: higher risk, moderate duration (per lot)
            duration = np.random.uniform(0.1, 1.5)  # Duration per lot
            credit_risk = np.random.uniform(0.02, 0.09)  # Higher risk per lot
            liquidity = np.random.uniform(0.02, 0.08)  # Lower liquidity per lot
        
        # Add some correlation structure to make trade-offs interesting
        if duration > 1.0:  # Long duration bonds per lot
            credit_risk += np.random.uniform(0, 0.02)  # Slightly more risky
            liquidity -= np.random.uniform(0, 0.02)    # Slightly less liquid
        
        # Ensure bounds for normalized values
        credit_risk = np.clip(credit_risk, 0.0, 0.10)
        liquidity = np.clip(liquidity, 0.01, 0.10)
        
        char_values = {
            'duration': duration,
            'credit_risk': credit_risk,
            'liquidity': liquidity
        }
        
        bonds.append(BondData(
            price=price,
            min_trade=min_trade,
            max_trade=max_trade,
            inventory=inventory,
            increment=increment,
            bucket=bucket,
            characteristics=char_values
        ))
    
    return bonds

def calculate_x_value(bond) -> float:
    """Calculate the normalized x_c value for a bond when selected (in lot units)"""
    return (bond.min_trade + min(bond.max_trade, bond.inventory)) / (2 * bond.increment)

# Drop-in replacement for the target generation in generate_problem
def generate_problem_interesting(n_bonds: int, hash_code: str, excel: bool = False, docplex: bool = False):
    """
    Generate a more interesting normalized bond portfolio optimization problem with guardrails
    
    This is a drop-in replacement for the original generate_problem function
    """
    from dataclasses import dataclass
    from typing import List, Dict, Tuple
    
    @dataclass
    class ProblemInstance:
        bonds: List
        targets: Dict[Tuple[str, str], float]
        weights: Dict[str, float]
        mv_basket: float
        min_rc: float
        max_rc: float
        max_bonds: int
        solution: List[int]
        guardrails: Guardrails  # NEW!
    
    # Set random seed for reproducibility
    seed = int(hash(hash_code)) % (2**31)
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate more interesting normalized bonds
    bonds = generate_more_interesting_bonds(n_bonds)
    
    # Generate realistic max_bonds constraint (30-70% of universe)
    max_bonds = max(2, int(n_bonds * np.random.uniform(0.3, 0.7)))
    
    # Generate realistic normalized targets and parameters
    targets, weights, mv_basket, min_rc, max_rc = generate_realistic_targets_and_params(
        bonds, n_bonds, max_bonds
    )
    
    # Solve for optimal solution using the proper optimizer
    solution = generate_optimal_solution(bonds, targets, weights, mv_basket, min_rc, max_rc, max_bonds)
    
    # Generate smart normalized guardrails based on the optimal solution
    guardrails = generate_smart_guardrails(bonds, targets, weights, max_bonds, solution)
    
    problem = ProblemInstance(
        bonds=bonds,
        targets=targets,
        weights=weights,
        mv_basket=mv_basket,
        min_rc=min_rc,
        max_rc=max_rc,
        max_bonds=max_bonds,
        solution=solution,
        guardrails=guardrails
    )
    
    # Save to files if requested
    if excel:
        save_to_excel(problem, hash_code)  # Your existing function
    
    if docplex:
        save_to_docplex(problem, hash_code)
    
    print(f"Generated interesting normalized problem with guardrails:")
    print(f"- Bonds: {n_bonds}, Max selectable: {max_bonds}")
    print(f"- All values normalized to lot units (price ~1.0, trades ~1-10 lots)")
    print(f"- Target ranges realistic for normalized portfolio combinations")
    print(f"- Cash constraints encourage multiple bonds ({min_rc:.2f}-{max_rc:.2f} lots)")
    print(f"- Bond characteristics have interesting trade-offs (per-lot basis)")
    print(f"- Value-based guardrails add {len(guardrails.lower_bounds)} additional constraints")
    print(f"- Problem scales to 1000+ bonds without numerical explosion")
    
    return problem

def sample_meaningful_characteristic_values(bonds, bucket_bonds: List[int], char: str, max_bonds: int, num_samples: int = 300) -> List[float]:
    """
    Sample normalized portfolio combinations that actually SELECT bonds (no zero contributions)
    """
    
    values = []
    
    for _ in range(num_samples):
        # Force selection of at least 1 bond from this bucket
        num_selected = np.random.randint(1, min(len(bucket_bonds), max_bonds) + 1)
        
        selected_indices = np.random.choice(bucket_bonds, size=num_selected, replace=False)
        
        # Calculate total normalized contribution
        total_value = 0.0
        for i in selected_indices:
            beta = bonds[i].characteristics[char]  # per lot
            x_val = calculate_x_value(bonds[i])  # lot units
            total_value += beta * x_val  # normalized
        
        values.append(total_value)
    
    return values

def calculate_minimum_meaningful_contribution(bonds, bucket_bonds: List[int], char: str) -> float:
    """
    Calculate the minimum meaningful normalized contribution by taking the smallest single-bond contribution
    """
    
    if not bucket_bonds:
        return 0.01  # Small normalized fallback
    
    min_contribution = float('inf')
    
    for i in bucket_bonds:
        beta = bonds[i].characteristics[char]  # per lot
        x_val = calculate_x_value(bonds[i])  # lot units
        contribution = beta * x_val  # normalized
        min_contribution = min(min_contribution, contribution)
    
    # Return half of the minimum single-bond contribution as the meaningful minimum
    return 0.5 * min_contribution

def calculate_contribution_ranges(bonds, characteristics: List[str], buckets: List[str], max_bonds: int) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Calculate realistic contribution ranges for each (bucket, characteristic) pair
    by sampling possible portfolio combinations
    """
    
    contribution_ranges = {}
    
    for bucket in buckets:
        bucket_bonds = [i for i, bond in enumerate(bonds) if bond.bucket == bucket]
        
        if not bucket_bonds:
            # No bonds in this bucket
            for char in characteristics:
                contribution_ranges[(bucket, char)] = (0.0, 0.0)
            continue
        
        for char in characteristics:
            contributions = []
            
            # Sample different portfolio combinations
            num_samples = min(1000, 2**len(bucket_bonds))  # Don't explode with too many bonds
            
            for _ in range(num_samples):
                # Random selection of bonds from this bucket
                num_selected = np.random.randint(0, min(len(bucket_bonds), max_bonds) + 1)
                
                if num_selected == 0:
                    contributions.append(0.0)
                else:
                    selected_indices = np.random.choice(bucket_bonds, size=num_selected, replace=False)
                    
                    # Calculate total contribution
                    total_contrib = 0.0
                    for i in selected_indices:
                        beta = bonds[i].characteristics[char]
                        x_val = calculate_x_value(bonds[i])
                        total_contrib += beta * x_val
                    
                    contributions.append(total_contrib)
            
            # Set range based on observed contributions
            min_contrib = min(contributions)
            max_contrib = max(contributions)
            
            # Add some buffer to avoid targets right at the boundary
            range_buffer = 0.1 * (max_contrib - min_contrib) if max_contrib > min_contrib else 1.0
            contribution_ranges[(bucket, char)] = (
                min_contrib - range_buffer, 
                max_contrib + range_buffer
            )
    
    return contribution_ranges
