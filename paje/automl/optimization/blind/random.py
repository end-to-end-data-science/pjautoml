import numpy as np
from paje.automl.automl import AutoML
from paje.automl.composer.iterator import Iterator
from paje.automl.composer.seq import Seq
from paje.base.cache import Cache
from paje.ml.element.posprocessing.metric import Metric
from paje.ml.element.posprocessing.summ import Summ
from paje.ml.element.preprocessing.supervised.instance.sampler.cv import CV


class RandomAutoML(AutoML):

    def __init__(self,
                 preprocessors,
                 modelers,
                 pipe_length,
                 repetitions,
                 random_state,
                 cache_settings_for_components=None,
                 **kwargs):
        """
        AutoML
        :param preprocessors: list of modules for balancing,
            noise removal, sampling etc.
        :param modelers: list of modules for prediction
            (classification or regression etc.)
        :param repetitions: how many times can a module appear
            in a pipeline
        :param method: TODO
        :param max_iter: maximum number of pipelines to evaluate
        :param max_depth: maximum length of a pipeline
        :param static: are the pipelines generated always exactly
            as given by the ordered list preprocessors + modelers?
        :param fixed: are the pipelines generated always with
            length max(max_depth, len(preprocessors + modelers))?
        :param random_state: TODO
        :return:
        """

        AutoML.__init__(self,
                        components=preprocessors + modelers,
                        **kwargs)

        # These attributes identify uniquely AutoML.
        # This structure is necessary because the AutoML is a Component and it
        # could be used into other Components, like the Pipeline one.
        # build_impl()
        self.repetitions = repetitions
        self.pipe_length = pipe_length
        # __init__()
        self.modelers = modelers
        self.preprocessors = preprocessors

        if not isinstance(modelers, list) or \
                not isinstance(preprocessors, list):
            print(modelers)
            print(preprocessors)
            raise TypeError("The modelers/preprocessors must be list.")

        if not modelers:
            raise ValueError("The list length must be greater than one.")

        # Other class attributes.
        # These attributes can be set here or in the build_impl method. They
        # should not influence the AutoML final result.
        self.storage_settings_for_components = cache_settings_for_components

        # Class internal attributes
        # Attributes that were not parameterizable
        self.best_eval = float('-Inf')
        self.best_pipe = None
        self.curr_eval = None
        self.curr_pipe = None

        self.random_state = random_state
        np.random.seed(self.random_state)

    def next_pipelines(self, data):
        """ TODO the docstring documentation
        """
        components = self.choose_modules()
        tree = Seq.cs(config_spaces=components)

        config = tree.sample()

        config['random_state'] = self.random_state
        # self.curr_pipe = Pipeline(
        #     config, storage_settings=self.storage_settings_for_components
        # )
        self.curr_pipe = config
        return [config]

    def choose_modules(self):
        """ TODO the docstring documentation
        """
        take = np.random.randint(0, self.pipe_length)

        preprocessors = self.preprocessors * (self.repetitions + 1)
        np.random.shuffle(preprocessors)
        return preprocessors[:take] + [np.random.choice(self.modelers)]

    def process_step(self, eval_result):
        """ TODO the docstring documentation
        """
        self.curr_eval = eval_result[0][1] or 0
        if self.curr_eval is not None \
                and self.curr_eval > self.best_eval:
            self.best_eval = self.curr_eval
            self.best_pipe = self.curr_pipe

    def get_best_pipeline(self):
        """ TODO the docstring documentation
        """
        return Seq(self.best_pipe)

    def get_current_eval(self):
        """ TODO the docstring documentation
        """
        return self.curr_eval

    def get_best_eval(self):
        """ TODO the docstring documentation
        """
        return self.best_eval

    def eval(self, pip_config, data):
        fast = False  # TODO: True -> composer line 35, KeyError: 'random_state'
        if fast:
            internal = Seq.cfg(
                configs=[
                    pip_config,
                    Metric.cfg(function='accuracy')  # from Y to r
                ],
                random_state=self.random_state
            )
        else:
            internal = Seq.cfg(
                configs=[
                    Cache.cfg(
                        configs=[pip_config],
                        **self.storage_settings_for_components
                    ),
                    Metric.cfg(function='accuracy')  # from Y to r
                ],
                random_state=self.random_state
            )

        iterat = Seq.cfg(
            configs=[
                Iterator.cfg(
                    iterable=CV.cfg(split='cv', steps=10, fields=['X', 'Y']),
                    configs=[internal],
                    field='r'
                ),
                Summ.cfg(field='s', function='mean')
            ]
        )

        if fast:
            dic = self.storage_settings_for_components
            dic['configs'] = [iterat]
            pip = Cache(config=dic)
        else:
            pip = Seq(
                config={
                    'configs': [iterat],
                    'random_state': self.random_state
                }
            )

        print('aaaaaaaaaaaaaaaaaaaaaaaaa')
        datapp = pip.apply(data)
        if datapp is None:
            return pip, (None, None)

        print('uuuuuuuuuuuuuuuuuuuuuuuuu')
        datause = pip.use(data)
        if datause is None:
            return pip, (None, None)

        return pip, (datapp.s, datause.s)
