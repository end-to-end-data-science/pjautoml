###  From StackOverflow  ###
from functools import wraps
import inspect


###########################################################################################################################
# //////|   Decorator   |//////////////////////////////////////////////////////////////////////////////////////////////////#
###########################################################################################################################

def initializer(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        _assign_args(self, list(args), kwargs, function)
        function(self, *args, **kwargs)

    return wrapped


###########################################################################################################################
# //////|   Utils   |//////////////////////////////////////////////////////////////////////////////////////////////////////#
###########################################################################################################################

def _assign_args(instance, args, kwargs, function):
    def set_attribute(instance, parameter, default_arg):
        if not (parameter.startswith("_")):
            setattr(instance, parameter, default_arg)

    def assign_keyword_defaults(parameters, defaults):
        for parameter, default_arg in zip(reversed(parameters), reversed(defaults)):
            set_attribute(instance, parameter, default_arg)

    def assign_positional_args(parameters, args):
        for parameter, arg in zip(parameters, args.copy()):
            set_attribute(instance, parameter, arg)
            args.remove(arg)

    def assign_keyword_args(kwargs):
        for parameter, arg in kwargs.items():
            set_attribute(instance, parameter, arg)

    def assign_keyword_only_defaults(defaults):
        return assign_keyword_args(defaults)

    def assign_variable_args(parameter, args):
        set_attribute(instance, parameter, args)

    POSITIONAL_PARAMS, VARIABLE_PARAM, _, KEYWORD_DEFAULTS, _, KEYWORD_ONLY_DEFAULTS, _ = inspect.getfullargspec(function)
    POSITIONAL_PARAMS = POSITIONAL_PARAMS[1:]  # remove 'self'

    if (KEYWORD_DEFAULTS): assign_keyword_defaults(parameters=POSITIONAL_PARAMS, defaults=KEYWORD_DEFAULTS)
    if (KEYWORD_ONLY_DEFAULTS): assign_keyword_only_defaults(defaults=KEYWORD_ONLY_DEFAULTS)
    if (args): assign_positional_args(parameters=POSITIONAL_PARAMS, args=args)
    if (kwargs): assign_keyword_args(kwargs=kwargs)
    if (VARIABLE_PARAM): assign_variable_args(parameter=VARIABLE_PARAM, args=args)

###########################################################################################################################$#//////|   Tests   |//////////////////////////////////////////////////////////////////////////////////////////////////////#$###########################################################################################################################$$if __name__ == "__main__":$$#######|   Positional   |##################################################################################################$$  class T:$    @auto_assign_arguments$    def __init__(self, a, b):$      pass$$  t = T(1, 2)$  assert (t.a == 1) and (t.b == 2)$$#######|   Keyword   |#####################################################################################################$$  class T:$    @auto_assign_arguments$    def __init__(self, a="KW_DEFAULT_1", b="KW_DEFAULT_2"):$      pass$$  t = T(a="kw_arg_1", b="kw_arg_2")$  assert (t.a == "kw_arg_1") and (t.b == "kw_arg_2")$$#######|   Positional + Keyword   |########################################################################################$$  class T:$    @auto_assign_arguments$    def __init__(self, a, b, c="KW_DEFAULT_1", d="KW_DEFAULT_2"):$      pass$$  t = T(1, 2)$  assert (t.a == 1) and (t.b == 2) and (t.c == "KW_DEFAULT_1") and (t.d == "KW_DEFAULT_2")$$  t = T(1, 2, c="kw_arg_1")$  assert (t.a == 1) and (t.b == 2) and (t.c == "kw_arg_1") and (t.d == "KW_DEFAULT_2")$$  t = T(1, 2, d="kw_arg_2")$  assert (t.a == 1) and (t.b == 2) and (t.c == "KW_DEFAULT_1") and (t.d == "kw_arg_2")$$#######|   Variable Positional   |#########################################################################################$$  class T:$    @auto_assign_arguments$    def __init__(self, *a):$      pass$$  t = T(1, 2, 3)$  assert (t.a == [1, 2, 3])$$#######|   Variable Positional + Keyword   |###############################################################################$$  class T:$    @auto_assign_arguments$    def __init__(self, *a, b="KW_DEFAULT_1"):$      pass$$  t = T(1, 2, 3)$  assert (t.a == [1, 2, 3]) and (t.b == "KW_DEFAULT_1")$$  t = T(1, 2, 3, b="kw_arg_1")$  assert (t.a == [1, 2, 3]) and (t.b == "kw_arg_1")$$#######|   Variable Positional + Variable Keyword   |######################################################################$$  class T:$    @auto_assign_arguments$    def __init__(self, *a, **kwargs):$      pass$$  t = T(1, 2, 3, b="kw_arg_1", c="kw_arg_2")$  assert (t.a == [1, 2, 3]) and (t.b == "kw_arg_1") and (t.c == "kw_arg_2")$$#######|   Positional + Variable Positional + Keyword   |##################################################################$$  class T:$    @auto_assign_arguments$    def __init__(self, a, *b, c="KW_DEFAULT_1"):$      pass$$  t = T(1, 2, 3, c="kw_arg_1")$  assert (t.a == 1) and (t.b == [2, 3]) and (t.c == "kw_arg_1")$$#######|   Positional + Variable Positional + Variable Keyword   |#########################################################$$  class T:$    @auto_assign_arguments$    def __init__(self, a, *b, **kwargs):$      pass$$  t = T(1, 2, 3, c="kw_arg_1", d="kw_arg_2")$  assert (t.a == 1) and (t.b == [2, 3]) and (t.c == "kw_arg_1") and (t.d == "kw_arg_2")$$#######|   Keyword Only   |################################################################################################$$  class T:$    @auto_assign_arguments$    def __init__(self, *, a="KW_DEFAULT_1"):$      pass$$  t = T(a="kw_arg_1")$  assert (t.a == "kw_arg_1")$$#######|   Positional + Keyword Only   |###################################################################################$$  class T:$    @auto_assign_arguments$    def __init__(self, a, *, b="KW_DEFAULT_1"):$      pass$$  t = T(1)$  assert (t.a == 1) and (t.b == "KW_DEFAULT_1")$$  t = T(1, b="kw_arg_1")$  assert (t.a == 1) and (t.b == "kw_arg_1")$$#######|   Private __init__ Variables (underscored)   |####################################################################$$  class T:$    @auto_assign_arguments$    def __init__(self, a, b, _c):$      pass$$  t = T(1, 2, 3)$  assert hasattr(t, "a") and hasattr(t, "b") and not(hasattr(t, "_c"))
