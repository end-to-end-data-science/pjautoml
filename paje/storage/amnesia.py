import _pickle as pickle
from pathlib import Path

from paje.storage.persistence import Persistence
from paje.util.encoders import pack_comp, unpack_comp


class Amnesia(Persistence):
    def __init__(self, nested_storage=None, sync=False):
        super().__init__(nested_storage=nested_storage, sync=sync)

    def store_data_impl(self, data, file=None):
        return None

    def lock_impl(self, component, input_data):
        pass

    def get_result_impl(self, component, input_data):
        return None, False, False

    def store_result_impl(self, component, input_data, output_data):
        pass

    def get_data_by_name_impl(self, name, fields=None, history=None):
        raise NotImplementedError('Storage in dump mode (pickledfile) cannot'
                                  ' recover data by name')

    def get_data_by_uuid_impl(self, datauuid):
        raise NotImplementedError('Storage in dump mode (pickledfile) cannot'
                                  ' recover (training?) data by uuid')

    def get_finished_names_by_mark_impl(self, mark):
        raise NotImplementedError('Storage in dump mode (pickledfile) cannot'
                                  ' recover data by mark')
