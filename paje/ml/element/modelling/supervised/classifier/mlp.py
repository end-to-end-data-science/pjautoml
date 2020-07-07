import math
from sklearn.neural_network import MLPClassifier

from paje.searchspace.configspace import HPTree
from paje.ml.element.modelling.supervised.classifier.classifier import Classifier
from paje.util.distributions import exponential_integers


class MLP(Classifier):
    def build_impl(self):
        # Convert '@' hyperparameters to sklearn format.
        n_hidden_layers = 0
        new_kwargs = self.config.copy()

        if '@neurons' in new_kwargs:
            neurons = new_kwargs.pop('@neurons')
            # in_out = new_kwargs.pop('@in_out')
            l = [0, 0, 0, 0]
            for k in self.config:
                if k.startswith('@hidden_layer_size'):
                    layer = int(k[-1])
                    n_hidden_layers = max(n_hidden_layers, layer)
                    l[layer] = self.config.get(k)
                    del new_kwargs[k]

            l_sum = sum(l)
            if l_sum > 0:
                # free_neurons = math.pow(free_parameters/in_out,
                #                         1/n_hidden_layers)
                l = [math.ceil(i*neurons/l_sum) for i in l]

            if n_hidden_layers == 1:
                values = (l[1],)
            elif n_hidden_layers == 2:
                values = (l[1], l[2])
            elif n_hidden_layers == 3:
                values = (l[1], l[2], l[3])
            elif n_hidden_layers == 0:
                values = None
            else:
                raise Exception('unexpected number of layers', n_hidden_layers)
            if sum(l) > 0:
                new_kwargs['hidden_layer_sizes'] = values
        self.model = MLPClassifier(**new_kwargs)


    @classmethod
    def cs_impl(self):

        # Todo: set random seed
        max_neurons = 10000

        node = {
            'alpha': ['o',
                      [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                       1000, 10000]],
            # https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html
            'max_iter': ['c', [10000]],  # We assume that non converged is bad.
            # 'Number of epochs'/'gradient steps'.
            'tol': ['o',
                    [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                     1000, 10000]],
            # Maybe useless when learning_rate is set to ‘adaptive’.
            'nesterovs_momentum': ['c', [True, False]],
            # number of inputs times outputs
            # '@in_out': ['c', [data.n_attributes * data.n_classes]],
            '@neurons': ['c', exponential_integers(max_neurons, 3)]
        }

        zero_hidden_layers = HPTree({
            'hidden_layer_sizes': ['c', [()]],
        }, children=[])

        one_hidden_layer = HPTree({
            '@hidden_layer_size1': ['r', [0, 1]],
            # @ indicates that this hyperparameter is auxiliary
            # (will be converted in constructor)
            'activation': ['c', ['identity', 'logistic', 'tanh', 'relu']],
            # Only used when there is at least one hidden layer
        }, children=[])

        two_hidden_layers = HPTree({
            '@hidden_layer_size2': ['r', [0, 1]],
            # @ indicates that this hyperparameter is auxiliary
            # (will be converted in constructor)
        }, children=[one_hidden_layer])

        three_hidden_layers = HPTree({
            '@hidden_layer_size3': ['r', [0, 1]],
            # @ indicates that this hyperparameter is auxiliary
            # (will be converted in constructor)
        }, children=[two_hidden_layers])

        layers = [zero_hidden_layers, one_hidden_layer, two_hidden_layers,
                  three_hidden_layers]

        early_stopping = HPTree({
            'early_stopping': ['c', [True]],
            # Only effective when solver=’sgd’ or ‘adam’.
            'validation_fraction': ['c', [0.01, 0.05, 0.1, 0.15,
                                          0.20, 0.25, 0.30]],
            # Only used if early_stopping is True.
        }, children=layers)

        late_stopping = HPTree({
            'early_stopping': ['c', [False]],
            # Only effective when solver=’sgd’ or ‘adam’.
        }, children=layers)

        stoppings = [early_stopping, late_stopping]

        solver_adam = HPTree({
            'solver': ['c', ['adam']],
            'beta_1': ['r', [0.0, 0.999999]],  # 'adam'
            'beta_2': ['r', [0.0, 0.999999]],  # 'adam'
            'epsilon': ['r', [0.0000000001, 1.0]],  # 'adam'
        }, children=stoppings)

        learning_rate_constant = HPTree({
            'learning_rate': ['c', ['constant']],
            # only for solver=sgd (i will believe the docs, but it seems like
            # 'learning_rate' is for 'adam' also).
            'power_t': ['r', [0.0, 2.0]],
            # only for learning_rate=constant; it is unclear if MLP benefits
            # from power_t > 1
        }, children=stoppings)

        learning_rate_invscaling = HPTree({
            'learning_rate': ['c', ['invscaling']],
            # only for solver=sgd (i will believe the docs, but it seems like
            # 'learning_rate' is for 'adam' also).
        }, children=stoppings)

        learning_rate_adaptive = HPTree({
            'learning_rate': ['c', ['adaptive']],
            # only for solver=sgd (i will believe the docs, but it seems like
            # 'learning_rate' is for 'adam' also).
        }, children=stoppings)

        solver_sgd = HPTree({
            'solver': ['c', ['sgd']],
            'momentum': ['r', [0.0, 1.0]],
            # Only used when solver=’sgd’ (i will believe the docs,
            # but it seems like 'momentum' is for 'adam' also).
        }, children=[learning_rate_constant, learning_rate_invscaling,
                     learning_rate_adaptive])

        solver_non_newton = HPTree({
            'n_iter_no_change': ['c', [10]],
            # Only effective when solver=’sgd’ or ‘adam’.
            'batch_size': ['c', ['auto']],
            #                      min([1000, floor(data.n_instances() / 2)])]],
            # useless for solver lbfgs
            # useless for solver lbfgs
            'learning_rate_init': ['r', [0.000001, 0.5]],
            # Only used when solver=’sgd’ or ‘adam’
            'shuffle': ['c', [True, False]],
            # Only used when solver=’sgd’ or ‘adam’.
        }, children=[solver_adam, solver_sgd])

        solver_lbfgs = HPTree({
            'solver': ['c', ['lbfgs']],
        }, children=layers)

        tree = HPTree(node, children=[solver_non_newton, solver_lbfgs])

        return tree

    @classmethod
    def tree_impl_back(cls, data=None):
        cls.check_data(data)

        # Todo: set random seed
        max_neurons = int(
            (data.n_instances() / (data.n_attributes() + data.n_classes()))
        )

        node = {
            'alpha': ['o',
                      [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                       1000, 10000]],
            # https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html
            'max_iter': ['c', [10000]],  # We assume that non converged is bad.
            # 'Number of epochs'/'gradient steps'.
            'tol': ['o',
                    [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                     1000, 10000]],
            # Maybe useless when learning_rate is set to ‘adaptive’.
            'nesterovs_momentum': ['c', [True, False]],
            # number of inputs times outputs
            # '@in_out': ['c', [data.n_attributes * data.n_classes]],
            '@neurons': ['c', exponential_integers(max_neurons, 3)]
        }

        zero_hidden_layers = HPTree({
            'hidden_layer_sizes': ['c', [()]],
        }, children=[])

        one_hidden_layer = HPTree({
            '@hidden_layer_size1': ['r', [0, 1]],
            # @ indicates that this hyperparameter is auxiliary
            # (will be converted in constructor)
            'activation': ['c', ['identity', 'logistic', 'tanh', 'relu']],
            # Only used when there is at least one hidden layer
        }, children=[])

        two_hidden_layers = HPTree({
            '@hidden_layer_size2': ['r', [0, 1]],
            # @ indicates that this hyperparameter is auxiliary
            # (will be converted in constructor)
        }, children=[one_hidden_layer])

        three_hidden_layers = HPTree({
            '@hidden_layer_size3': ['r', [0, 1]],
            # @ indicates that this hyperparameter is auxiliary
            # (will be converted in constructor)
        }, children=[two_hidden_layers])

        layers = [zero_hidden_layers, one_hidden_layer, two_hidden_layers,
                  three_hidden_layers]

        early_stopping = HPTree({
            'early_stopping': ['c', [True]],
            # Only effective when solver=’sgd’ or ‘adam’.
            'validation_fraction': ['c', [0.01, 0.05, 0.1, 0.15,
                                          0.20, 0.25, 0.30]],
            # Only used if early_stopping is True.
        }, children=layers)

        late_stopping = HPTree({
            'early_stopping': ['c', [False]],
            # Only effective when solver=’sgd’ or ‘adam’.
        }, children=layers)

        stoppings = [early_stopping, late_stopping]

        solver_adam = HPTree({
            'solver': ['c', ['adam']],
            'beta_1': ['r', [0.0, 0.999999]],  # 'adam'
            'beta_2': ['r', [0.0, 0.999999]],  # 'adam'
            'epsilon': ['r', [0.0000000001, 1.0]],  # 'adam'
        }, children=stoppings)

        learning_rate_constant = HPTree({
            'learning_rate': ['c', ['constant']],
            # only for solver=sgd (i will believe the docs, but it seems like
            # 'learning_rate' is for 'adam' also).
            'power_t': ['r', [0.0, 2.0]],
            # only for learning_rate=constant; it is unclear if MLP benefits
            # from power_t > 1
        }, children=stoppings)

        learning_rate_invscaling = HPTree({
            'learning_rate': ['c', ['invscaling']],
            # only for solver=sgd (i will believe the docs, but it seems like
            # 'learning_rate' is for 'adam' also).
        }, children=stoppings)

        learning_rate_adaptive = HPTree({
            'learning_rate': ['c', ['adaptive']],
            # only for solver=sgd (i will believe the docs, but it seems like
            # 'learning_rate' is for 'adam' also).
        }, children=stoppings)

        solver_sgd = HPTree({
            'solver': ['c', ['sgd']],
            'momentum': ['r', [0.0, 1.0]],
            # Only used when solver=’sgd’ (i will believe the docs,
            # but it seems like 'momentum' is for 'adam' also).
        }, children=[learning_rate_constant, learning_rate_invscaling,
                     learning_rate_adaptive])

        solver_non_newton = HPTree({
            'n_iter_no_change': ['c', [10]],
            # Only effective when solver=’sgd’ or ‘adam’.
            'batch_size': ['c', ['auto']],
            #                      min([1000, floor(data.n_instances() / 2)])]],
            # useless for solver lbfgs
            # useless for solver lbfgs
            'learning_rate_init': ['r', [0.000001, 0.5]],
            # Only used when solver=’sgd’ or ‘adam’
            'shuffle': ['c', [True, False]],
            # Only used when solver=’sgd’ or ‘adam’.
        }, children=[solver_adam, solver_sgd])

        solver_lbfgs = HPTree({
            'solver': ['c', ['lbfgs']],
        }, children=layers)

        tree = HPTree(node, children=[solver_non_newton, solver_lbfgs])

        return tree
