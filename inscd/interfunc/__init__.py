from .cdmfkc import CDMFKC_IF
from .dp import DP_IF
from .irt import IRT_IF
from .kancd import KANCD_IF
from .kscd import KSCD_IF
from .mf import MF_IF
from .mirt import MIRT_IF
from .ncd import NCD_IF
from .rcd import RCD_IF
from .scd import SCD_IF
from .disengcd import DISENGCD_IF
__all__ = [
    "CDMFKC_IF",
    "DP_IF",
    "IRT_IF",
    "KANCD_IF",
    "KSCD_IF",
    "MF_IF",
    "MIRT_IF",
    "NCD_IF",
    "RCD_IF",
    "SCD_IF",
    "DISENGCD_IF"
]

INTERACTION_FUNCTIONS = {
    "CDMFKC_IF": {"class": CDMFKC_IF, "require_transfer": True},
    "DP_IF": {"class": DP_IF, "require_transfer": True},
    "IRT_IF": {"class": IRT_IF, "require_transfer": False},
    "KANCD_IF": {"class": KANCD_IF, "require_transfer": False},
    "KSCD_IF": {"class": KSCD_IF, "require_transfer": True},
    "MF_IF": {"class": MF_IF, "require_transfer": True},
    "MIRT_IF": {"class": MIRT_IF, "require_transfer": False},
    "NCD_IF": {"class": NCD_IF, "require_transfer": True},
    "RCD_IF": {"class": RCD_IF, "require_transfer": False},
    "SCD_IF": {"class": SCD_IF, "require_transfer": True},
    "DISENGCD_IF": {"class": DISENGCD_IF, "require_transfer": False}
}

def get_interaction_function(name: str):
    """
    Get the corresponding interaction function class based on the given name.

    Args:
        name (str): The name of the interaction function (e.g., 'MF', 'IRT', etc.)

    Returns:
        tuple: A tuple containing the corresponding interaction function class and the require_transfer flag.

    Raises:
        ValueError: If the name is invalid or not found.
    """
    try:
        interaction_func = INTERACTION_FUNCTIONS[name.upper()]
        return interaction_func["class"], interaction_func["require_transfer"]
    except KeyError:
        print(name)
        raise ValueError(f"Unsupported interaction function name: {name}. "
                         f"Supported names: {list(INTERACTION_FUNCTIONS.keys())}")