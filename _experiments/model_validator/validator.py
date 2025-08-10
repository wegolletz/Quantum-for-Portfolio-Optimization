from collections.abc import Sequence
from typing import Tuple, Optional

def is_feasible_with_objective(lp_file: str, sol: Sequence[int]) -> Tuple[bool, Optional[float]]:
    """
    Check feasibility of a given binary solution vector against a docplex Model and return objective value.

    Args:
        lp_file (str): A path to a model instance defining binary variables and linear constraints.
        sol (Sequence[int]): A list or numpy array of 0/1 integers representing the proposed assignment.
                             The order corresponds to the iteration order of model.iter_variables().

    Returns:
        Tuple[bool, Optional[float]]: (feasibility, objective_value)
            - feasibility: True if the solution satisfies all constraints, False otherwise
            - objective_value: The objective function value if feasible, None if infeasible
    """
    import docplex.mp.model_reader
    
    # Read the model from the path
    model = docplex.mp.model_reader.ModelReader.read(lp_file)
    
    # Convert any sequence (e.g., list or numpy array) to a list
    sol_list = list(sol)
    
    # Get variables in iteration order
    variables = list(model.iter_variables())
    
    # Verify solution length matches number of variables
    if len(sol_list) != len(variables):
        raise ValueError(f"Solution length ({len(sol_list)}) does not match number of variables ({len(variables)})")
    
    # Create a solution object using docplex's built-in method
    solution = model.new_solution()
    
    # Add variable values to the solution
    for i, var in enumerate(variables):
        solution.add_var_value(var, sol_list[i])
    
    # Use docplex's built-in feasibility check
    unsatisfied_constraints = solution.find_unsatisfied_constraints(model)
    
    # Check if feasible
    is_feasible = len(unsatisfied_constraints) == 0
    
    # Get objective value for the specific solution (without solving)
    objective_value = None
    if is_feasible:
        # Evaluate the objective expression
        objective_value = _evaluate_objective_expression()
    
    return is_feasible, objective_value


def _evaluate_objective_expression() -> float:
    """
    Compute the value of the model’s objective for the candidate Solution
    that was just built in `is_feasible_with_objective()`.

    Notes
    -----
    * We recover the `SolveSolution` instance created upstream via the
      calling frame (no extra parameters needed, so we leave the rest of
      the code untouched).
    * The objective expression itself is available from the parent
      `Model` via the ``objective_expr`` property.:contentReference[oaicite:0]{index=0}
    * DOcplex’s documented ``Solution.get_value()`` method can evaluate
      *any* expression (variable, linear/quadratic term, KPI, …) in the
      context of the supplied variable assignments, so we simply feed it
      the objective expression.:contentReference[oaicite:1]{index=1}
    * No solver call is triggered – this is a pure, in-memory arithmetic
      evaluation.

    Returns
    -------
    float
        Objective value of the candidate solution.
    """
    import inspect

    # Look at the caller’s local scope to grab the Solution instance.
    caller_locals = inspect.currentframe().f_back.f_locals
    solution = caller_locals.get("solution")
    if solution is None:
        raise RuntimeError(
            "Internal error: expected a `solution` object in caller scope."
        )

    # The model linked to that solution already knows its objective expr.
    objective_expr = solution.model.objective_expr  # built-in accessor:contentReference[oaicite:2]{index=2}

    # DOcplex evaluates the expression with the provided variable values.
    total_value = solution.get_value(objective_expr)  # built-in evaluator:contentReference[oaicite:3]{index=3}
    return total_value


def is_feasible(lp_file: str, sol: Sequence[int]) -> bool:
    """
    Check feasibility of a given binary solution vector against a docplex Model.
    
    This is the original function that just returns feasibility.
    """
    feasible, _ = is_feasible_with_objective(lp_file, sol)
    return feasible