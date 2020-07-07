""" TODO the docstring documentation
"""
from abc import abstractmethod, ABC
from paje.base.data import Data
from paje.base.noniterable import NonIterable


class AutoML(NonIterable, ABC):
    """ TODO the docstring documentation
    """

    def __init__(self,
                 components,
                 max_iter,
                 n_jobs=1,
                 verbose=True,
                 **kwargs):
        super().__init__(**kwargs)
        """ TODO the docstring documentation
        """

        # Attributes set in the build_impl.
        # These attributes identify uniquely AutoML.
        # This structure is necessary because the AutoML is a Component and it
        # could be used into other Components, like the Pipeline one.

        self.components = components

        # Other class attributes.
        # These attributes can be set here or in the build_impl method. They
        # should not influence the AutoML final storage.
        self.n_jobs = n_jobs  # I am not sure if this variable should be here.
        self.verbose = verbose

        # Class internal attributes
        # Attributes that were not parameterizable
        self.model = None
        self.all_eval_results = None
        self.fails, self.locks, self.successes, self.total = 0, 0, 0, 0
        self.current_iteration = 0
        self.max_iter = max_iter

    def eval_pipelines_par(self, pipelines, data, eval_results):
        """ TODO the docstring documentation
        """

    def eval_pipelines_seq(self, configs, data, eval_results):
        """ TODO the docstring documentation
        """
        for config in configs:
            self.total += 1
            pipe, eval_result = self.eval(config, data)
            if self.verbose:
                print(pipe)
            if pipe.failed:
                self.fails += 1
            elif pipe.locked_by_others:
                self.locks += 1
            else:
                self.successes += 1
            eval_results.append(eval_result)

    def apply_impl(self, data):
        """ TODO the docstring documentation
        """
        self.all_eval_results = []

        for iteration in range(1, self.max_iter + 1):
            self.current_iteration = iteration
            if self.verbose:
                print("####------##-----##-----##-----##-----##-----####")
                print("Current iteration = ", self.current_iteration)
            # Evaluates current hyperparameter (space-values) combination.
            pipelines = self.next_pipelines(data)

            # this variable saves the results of the current iteration
            eval_result = []
            if self.n_jobs > 1:
                # Runs all pipelines generated in this iteration (parallelly)
                # and put the results in the eval_result variable
                self.eval_pipelines_par(pipelines, data, eval_result)
            else:
                # Runs all pipelines generated in this iteration (sequentially)
                # and put the results in the eval_result variable
                self.eval_pipelines_seq(pipelines, data, eval_result)

            # This attribute save all results
            self.all_eval_results.append(eval_result)

            self.process_step(eval_result)
            if self.verbose:
                print("Current ...............: ", self.get_current_eval())
                print("Best ..................: ", self.get_best_eval())
                print("Locks/fails/successes .: {0}/{1}/{2}".format(
                    self.locks, self.fails, self.successes))
                print("-------------------------------------------------\n")

        self.process_all_steps(self.all_eval_results)

        self.model = self.get_best_pipeline()
        if self.verbose:
            print("Best pipeline found:")
            print(self.model)

        return self.model.apply(data)

    def use_impl(self, data):
        """ TODO the docstring documentation
        """
        return self.model.use(data)

    def get_best_eval(self):
        """ TODO the docstring documentation
        """

    def get_current_eval(self):
        """ TODO the docstring documentation
        """

    def process_step(self, eval_result):
        """ TODO the docstring documentation
        """

    def process_all_steps(self, eval_results):
        """ TODO the docstring documentation
        """
        pass

    @abstractmethod
    def get_best_pipeline(self):
        """ TODO the docstring documentation
        """
        raise NotImplementedError(
            "AutoML has no get_best_pipeline() implemented!"
        )

    @abstractmethod
    def next_pipelines(self, data):
        """ TODO the docstring documentation
        """
        raise NotImplementedError(
            "AutoML has no next_pipelines() implemented!"
        )

    @classmethod
    def cs_impl(cls):
        """ TODO the docstring documentation
        """
        raise NotImplementedError(
            "AutoML has no tree() implemented!"
        )

    @abstractmethod
    def eval(self, config, data):
        pass

    def modifies(self, op):
        if op not in ['a', 'u']:
            raise Exception('Wrong op:', op)

        if self._modified[op] is None:
            if op == 'a':
                # Assumes all fields will be modified by AutoML during apply().
                self._modified[op] = Data.all_mats.keys()
            else:
                self._modified[op] = self.model.modifies(op)

        return self._modified[op]
