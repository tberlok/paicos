from . import derived_variables_gas

user_functions = {}

default_functions = {}

for key in derived_variables_gas.functions.keys():
    default_functions[key] = derived_variables_gas.functions[key]
