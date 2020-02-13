import functools
import operator


def get_n_params(model) -> int:
    return sum((functools.reduce(operator.mul, p.size()) for p in model.parameters()))

