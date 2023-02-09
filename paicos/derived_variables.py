"""
This is where we import all the various default functions for getting derived variables.
"""
from . import derived_variables_gas

user_functions = {}

default_functions = {}

# Add all gas functions to the defaults
for func_name, func in derived_variables_gas.functions.items():
    default_functions[func_name] = func
