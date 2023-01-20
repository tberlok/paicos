

def get_variable_function(variable_str, info=False):
    """
    Convenience function for getting (derived) variables.

    Returns a function.
    """

    assert type(variable_str) is str

    if not variable_str[0].isnumeric() or variable_str[1] != '_':
        msg = ('\n\nKeys are expected to consist of an integer ' +
               '(the particle type) and a blockname, separated by a ' +
               ' _. For instance 0_Density. You can get the ' +
               'available fields like so: snap.info(0)')
        raise RuntimeError(msg)

    parttype = int(variable_str[0])
    # name = variable_str[2:]

    # User functions are always preferred
    from .util import user_functions, use_only_user_functions

    if info:
        res = []
        for key in user_functions.keys():
            if key[:1] == variable_str[:1]:
                res.append(key)
        if use_only_user_functions:
            return res
    else:
        if variable_str in user_functions:
            return user_functions[variable_str]
        else:
            if use_only_user_functions:
                msg = ('The derived variable {} is not found in the user ' +
                       'defined functions and use_only_user_functions is True')
                raise RuntimeError(msg.format(variable_str))

    if parttype == 0:
        from .derived_variables_gas import get_variable_function_gas
        func_or_list = get_variable_function_gas(variable_str, info)
    else:
        if info:
            func_or_list = list({}.keys())
        else:
            msg = ('\n\nA function to calculate the variable {} is not ' +
                   'implemented!\n\nIn fact, no derived variables are ' +
                   'available for this parttype.')
            raise RuntimeError(msg.format(variable_str))

    if info:
        return list(set(func_or_list).union(set(res)))
    else:
        return func_or_list


def get_variable(snap, variable_str):
    """
    Convenience function for getting (derived) variables.

    Returns an array.
    """

    assert type(variable_str) is str

    if not variable_str[0].isnumeric() or variable_str[1] != '_':
        msg = ('\n\nKeys are expected to consist of an integer ' +
               '(the particle type) and a blockname, separated by a ' +
               ' _. For instance 0_Density. You can get the ' +
               'available fields like so: snap.info(0)')
        raise RuntimeError(msg)

    parttype = int(variable_str[0])
    name = variable_str[2:]
    if name in snap.info(parttype, False):
        snap.load_data(parttype, name)
        return snap[variable_str]
    else:
        return get_variable_function(variable_str, False)(snap)
