from abc import ABC, abstractmethod

# Disabling profiling when not needed.

try:
    import builtins

    profile = builtins.__dict__['profile']
except KeyError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


class Persistence(ABC):
    """
    The children classes are expected to provide storage in e.g.:
     SQLite, remote/local MongoDB, MySQL server or even CSV files.
    """

    def __init__(self, nested_storage=None, sync=False):
        """
        This class stores and recovers results from some place.
        :param nested_storage:
            NOTE: Apparently, this nested implementation is useless,
            since one can emulate it nesting two Storages manually.

            ORIGINAL TEXT:
            usually the nested storage is local and the
            other is remote. So, all operations occur locally,
            but failed local look ups are tried again remotely.
            If the look up succeeds, then it inserts a replication locally.

            Inserts are replicated in both storages.

            More than one nesting level can exist, good luck doing that.

            ps.: only useful for basic, fully defined, operations
            (not get_names_by_mark, e.g.)

        :param sync: EXPERIMENTAL!
        whether to replicate cached local results (found in nested_storage)
            in remote storage. Usually this is not needed.
            However, if one wants to perform a first fast run in a local
            storage (e.g., non-nested SQLite) and be able to sync to a remote
            server later (e.g., MySQL), they could enable sync in the second
            run using the local storage nested into the remote with sync=True.
            ps. Currently, it does not check if the results are already
            remotely stored. It assumes that if the user is asking to sync,
            then it means that the remote part is not synced. If this is not
            the case for some results, useless traffic will be generated,
            but the respective operations will be ignored at the database level.

        """
        self.nested_storage = nested_storage
        self.sync = sync
        self.name = ' < < < < ' + self.__class__.__name__ + ' > > > > '

    @profile
    def _nested_first(self, f, *args):
        """
        Take a member function name and try first on nested_storage, then on
        self.
        :param f:
        :param kwargs:
        :return:
        """
        if self.nested_storage is not None:
            result = getattr(self.nested_storage, f)(*args)
            if result is not None:
                return result
        return getattr(self, f + '_impl')(*args)

    @profile
    def get_result(self, component, op, train_data_uuid, data):
        component.op = op
        component._train_data_uuid__mutable = train_data_uuid

        # try locally
        if self.nested_storage is not None:
            local_result, started, ended = \
                self.nested_storage.get_result(component, data)
            if started:
                if self.sync:
                    print('Replicating remotely...', self.name)
                    self.lock_impl(component, data)
                    if ended:
                        self.store_result_impl(
                            component, data, local_result
                        )
                return local_result, started, ended

        # try remotely
        if self.nested_storage is not None:
            print('Trying remotely...', self.name)
        remote_result, started, ended = \
            self.get_result_impl(component, data)

        # replicate locally if required
        if started and self.nested_storage is not None:
            print('Replicating locally...', self.nested_storage.name)
            self.nested_storage.lock(component, data)
            if ended:
                self.nested_storage.store_result(
                    component, data, remote_result
                )

        return remote_result, started

    @profile
    def get_data_by_uuid(self, data_uuid):
        # try locally
        if self.nested_storage is not None:
            local_result = self.nested_storage.get_data_by_uuid(data_uuid)
            if local_result is not None:
                if self.sync:
                    print('Replicating remotely...', self.name)
                    self.store_data_impl(local_result)
                return local_result

        # try remotely
        if self.nested_storage is not None:
            print('Trying remotely...', self.name)
        remote_result = self.get_data_by_uuid_impl(data_uuid)
        if remote_result is None:
            return None

        # replicate locally if required
        if self.nested_storage is not None:
            print('Replicating locally...', self.nested_storage.name)
            self.nested_storage.store_data(remote_result)

        return remote_result

    @profile
    def get_data_by_name(self, name, fields=None):
        # try locally
        if self.nested_storage is not None:
            local_result = self.nested_storage.get_data_by_name(name, fields)
            if local_result is not None:
                if self.sync:
                    print('Replicating remotely...', self.name)
                    self.store_data_impl(local_result)
                return local_result

        # try remotely
        if self.nested_storage is not None:
            print('Trying remotely...', self.name)
        remote_result = self.get_data_by_name_impl(name, fields)
        if remote_result is None:
            return None

        # replicate locally if required
        if self.nested_storage is not None:
            print('Replicating locally...', self.nested_storage.name)
            self.nested_storage.store_data(remote_result)

        return remote_result

    @profile
    def get_finished_names_by_mark(self, mark):
        # TODO: somehow replicate locally the remote results?
        #  or expect the user to run the needed scripts before the current one?
        return self._nested_first('get_finished_names_by_mark', mark)

    @profile
    def lock(self, component, input_data):
        """
        Lock the given task to avoid wasting concurrent processing.
        ps. When nesting storages, lock first the remote, since it assumes
        exclusive access to local storage.
        :param component:
        :param input_data:
        :return:
        """
        self.lock_impl(component, input_data)
        if not component.locked_by_others and self.nested_storage is not None:
            print('Locking locally...', self.nested_storage.name)
            self.nested_storage.lock(component, input_data)

    @profile
    def store_data(self, data):
        self.store_data_impl(data)
        if self.nested_storage is not None:
            self.nested_storage.store_data(data)

    @profile
    def store_result(self, component, input_data, output_data):
        self.store_result_impl(component, input_data, output_data)
        if self.nested_storage is not None:
            self.nested_storage.store_result(
                component, input_data, output_data
            )

    @profile
    def syncronize_copying_from_nested(self):
        """
        Replicate all nested content into the nesting storage.
        Needed only when one wants to ensure the nesting storage is equal or a
        superset of the nested storage.
        Although sharing only of results could be done by sync=True,
        it may be useful to distribute complete data stored locally by a
        previous non-nesting (and probably offline) run.
        :return:
        """
        raise NotImplementedError('this method will upload from local to '
                                  'remote storage')

    @profile
    def syncronize_copying_to_nested(self):
        """
        Replicate all nesting content into the nested storage.
        Needed only when one wants to ensure the nested store is equal or a
        superset of the nesting storage.
        Regarding only results, this is already done by an ordinary nesting.
        However, complete data from the nesting storage can be fetched by this
        method.
        Both situations are useful when one wants to be able to continue
        running locally, but offline (i.e. with a non nesting storage).
        It takes advantage from results previously stored remotely by any
        other node (or even itself).
        :return:
        """
        raise NotImplementedError('this method will download from remote '
                                  'storage to the local one')

    @abstractmethod
    def get_result_impl(self, component, data):
        pass

    @abstractmethod
    def get_data_by_uuid_impl(self, data_uuid):
        pass

    @abstractmethod
    def get_data_by_name_impl(self, name, fields=None):
        pass

    @abstractmethod
    def get_finished_names_by_mark_impl(self, mark):
        pass

    @abstractmethod
    def lock_impl(self, component, test):
        pass

    @abstractmethod
    def store_data_impl(self, data):
        pass

    @abstractmethod
    def store_result_impl(self, component, test, testout):
        pass

    # @abstractmethod
    # def get_component_by_uuid(self, component_uuid):
    #     pass

    # @abstractmethod
    # def get_component(self, component, train_data, input_data):
    #     pass
