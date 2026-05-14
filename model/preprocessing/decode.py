import ast
import logging
from typing import Union

import numpy as np
import pandas as pd
from web3_input_decoder import decode_function

logger = logging.getLogger(__name__)

def decode_input(
        input_value: str,
        contract: pd.Series
) -> Union[pd.Series, None]:
    """
    Decodes the input data of an Ethereum transaction using the contract ABI.

    The function validates the transaction input, retrieves the contract ABI and
    proxy ABI, and attempts to decode the input data. If decoding with the main
    ABI fails, it tries again using the proxy ABI when available.

    Args:
        input_value (str): Hexadecimal transaction input data.
        contract (pd.Series): Contract metadata containing the ABI and optional
            proxy ABI.

    Returns:
        Union[pd.Series, float]: Decoded transaction input if successful, or
        `np.nan` if the input is invalid or decoding fails.
    """
    if not isinstance(input_value, str) or not input_value.startswith("0x") or len(input_value) < 10:
        #logger.warning("ABI: Invalid input")
        return np.nan
    abi = contract.get("abi", [])
    proxy_abi = contract.get("proxy_abi", [])
    if not isinstance(abi, list): abi = []
    if not isinstance(proxy_abi, list): proxy_abi = []
    try:
        return decode_function(abi, input_value)
    except Exception as e:
        #logger.warning(f"ABI: {e}")
        if proxy_abi:
            try:
                return decode_function(proxy_abi, input_value)
            except Exception as e:
                pass
                #logger.warning(f"PROXY ABI: {e}")
        return np.nan
        
def count_elements(value):
    """
    Counts the total number of scalar elements inside a value.

    If the input value is a nested list, the function recursively counts all
    inner elements. Non-list values are counted as one element.

    Args:
        value: Value to count. It may be a scalar value, a list or a nested list.

    Returns:
        int: Total number of scalar elements contained in the input value.
    """
    if isinstance(value, list):
        return sum(count_elements(item) for item in value)
    else:
        return 1

def normalize_abi(x):
    """
    Normalizes an ABI value into a list.

    The function returns the input directly if it is already a list. If the input
    is a non-empty string, it tries to parse it as a Python literal and returns it
    only if the parsed value is a list. Invalid, empty or unsupported values are
    converted into an empty list.

    Args:
        x: ABI value to normalize. It may be a list, a string or another object.

    Returns:
        list: Normalized ABI list, or an empty list if parsing fails.
    """
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.strip():
        try:
            parsed = ast.literal_eval(x)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []