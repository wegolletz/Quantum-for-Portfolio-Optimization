from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class BondData:
    price: float  # Normalized around 1.0 (par = 1.0)
    min_trade: float  # In lot units (e.g., 1.0 = 1 lot)
    max_trade: float  # In lot units
    inventory: float  # In lot units
    increment: float  # In lot units (e.g., 0.1 = 0.1 lots)
    bucket: str  # 'GOVT' or 'CORP'
    characteristics: Dict[str, float]  # j -> beta_{c,j} normalized per lot

@dataclass
class Guardrails:
    """Value-based guardrails for each (bucket, characteristic) pair - normalized per lot"""
    lower_bounds: Dict[Tuple[str, str], float]  # (bucket, char) -> lower bound per lot
    upper_bounds: Dict[Tuple[str, str], float]  # (bucket, char) -> upper bound per lot

@dataclass
class ProblemInstance:
    bonds: List[BondData]
    targets: Dict[Tuple[str, str], float]  # (bucket, characteristic) -> target per lot
    weights: Dict[str, float]  # characteristic -> rho_j (dimensionless)
    mv_basket: float  # Basket market value in lots (e.g., 100.0 = 100 lots)
    min_rc: float  # Minimum residual cash in lot units
    max_rc: float  # Maximum residual cash in lot units
    max_bonds: int  # Maximum number of bonds (dimensionless)
    solution: List[int]  # binary solution vector
    guardrails: Guardrails  # Value-based constraints normalized per lot