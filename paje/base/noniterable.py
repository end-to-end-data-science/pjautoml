from abc import abstractmethod
import numpy
from paje.base.component import Component
from paje.base.exceptions import handle_exception, UseWithoutApply
from paje.util.time import time_limit


class NonIterable(Component):
    _ps = '''ps.: All Data transformation must be done via method updated() with 
        explicit keyworded args (e.g. X=X, y=...)!
        This is needed because modifies() will inspect the code and look for 
        the fields that can be modified by the component.'''

    @abstractmethod
    def apply_impl(self, data):
        f"""

        {self._ps} 
        """

    @abstractmethod
    def use_impl(self, data):
        f"""

        {self._ps}
        """

    def apply(self, data=None):
        """Todo the doc string
        """
        self.op = 'a'
        if data is None:
            print(f"Applying {self.name} on None returns None.")
            return None  # If the Pipeline is done, that's ok.

        self._train_data_uuid__mutable = data.uuid()

        self.handle_warnings()
        if self.name != 'CV':
            print('Applying ' + self.name + '...')
        start = self.clock()
        self.failure = None
        try:
            if self.max_time is None:
                output_data = self.apply_impl(data)
            else:
                with time_limit(self.max_time):
                    output_data = self.apply_impl(data)
        except Exception as e:
            print(e)
            self.failed = True
            self.failure = str(e)
            self.locked_by_others = False
            handle_exception(self, e)
            output_data = None
        self.time_spent = self.clock() - start
        # self.msg('Component ' + self.name + ' applied.')
        self.dishandle_warnings()

        return output_data

    def use(self, data=None):
        """Todo the doc string
        """
        self.op = 'u'

        if data is None:
            print(f"Using {self.name} on None returns None.")
            return None

        self.check_if_applied()

        self.handle_warnings()
        if self.name != 'CV':
            print('Using ', self.name, '...')

        # TODO: put time limit and/or exception handling like in apply()?
        start = self.clock()
        output_data = self.use_impl(data)  # TODO:handl excps like in apply?
        self.time_spent = self.clock() - start

        # self.msg('Component ' + self.name + 'used.')
        self.dishandle_warnings()

        return output_data

    @staticmethod
    def handle_warnings():
        # Mahalanobis in KNN needs to supress warnings due to NaN in linear
        # algebra calculations. MLP is also verbose due to nonconvergence
        # issues among other problems.
        numpy.warnings.filterwarnings('ignore')

    @staticmethod
    def dishandle_warnings():
        numpy.warnings.filterwarnings('always')

    def check_if_applied(self):
        if self._train_data_uuid__mutable is None:
            str = f'{self.name} should be applied after a build!'
            raise UseWithoutApply(str)

    def sid(self):
        """
        Short uuID
        First 5 chars of uuid for printing purposes.
        :return:
        """
        return self.uuid[:10]
