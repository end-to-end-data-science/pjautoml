import sys
import os
import numpy as np
import pandas as pd

# adding the project root to import 'paje' modules
PATH = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.split(PATH)[0]
sys.path.append(MODULE_PATH)

def func1(a, b):
    return a + b

def test_func1():
    assert func1(1,2) == 3
