from hyperopt import hp, fmin, tpe, Trials

from daisy import main
from daisy.utils.parser import parse_args, parse_space

if __name__ == '__main__':
    args = parse_args(tune=True)

    space = parse_space(args, tune=True)

