import numpy as np

def check_is(name, required_type, imputed_type):
    if isinstance(imputed_type, required_type):
        return True
    else:
        # type(x).__name__
        raise  ValueError("´{0}´ is not instance of {1} instead {2} was given.".format(name, required_type, imputed_type))


def check_min_max(name, imputed_value, min_value, max_value):
    if not min_value <= imputed_value <= max_value:
        raise ValueError("´{0}´ is not between {1} and {2}.".format(name, min_value, max_value))
    return True


def check_int(name, imputed_value, min_value, max_value):
    check_is(name, int, imputed_value)
    check_min_max(name, imputed_value, min_value, max_value)
    return True


def check_float(name, imputed_value, min_value, max_value):
    check_is(name, float, imputed_value)
    check_min_max(name, imputed_value, min_value, max_value)
    return True


def  check_X_y(X, y): #TODO: this function might be replaced by sklearn's default(?)
    check_is("X", np.ndarray, X)
    check_is("y", np.ndarray, y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("´X´ and ´y´ have different sizes.")
    return True


