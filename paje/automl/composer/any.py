from numpy.random import choice
from paje.automl.composer.composer import Composer

class Any(Composer):
    @staticmethod
    def sampling_function(config_spaces):
        return [choice(config_spaces).sample()]
