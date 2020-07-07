import arff
import numpy as np
import pandas as pd
import sklearn.datasets as ds

from paje.base.chain import Chain
from paje.ml.element.preprocessing.supervised.instance.sampler.cv \
    import CV
from paje.util.encoders import pack_data, uuid, json_unpack, zlibext_pack


class Data:
    """ Data
    """
    _vectors = {i: i.upper() for i in ['y', 'z', 'v', 'w']}
    _scalars = {'r': 'R', 's': 'S', 't': 'T'}
    from_alias = _vectors.copy()
    from_alias.update(_scalars)
    from_alias.update({
        'X': 'X',
        'U': 'U',
        'l': 'l',
        'm': 'm',
        'C': 'C',
        'P': 'P',
        'Q': 'Q'
    })
    to_alias = {v: k for k, v in from_alias.items()}
    from_alias.update({
        'Y': 'Y',
        'Z': 'Z',
        'V': 'V',
        'W': 'W',
        'R': 'R',
        'S': 'S',
        'T': 'T'
    })
    all_mats = to_alias.keys()

    def __init__(self, name,
                 X, Y=None, Z=None, P=None,
                 U=None, V=None, W=None, Q=None,
                 R=None, S=None,
                 l=None, m=None,
                 T=None,
                 C=None,
                 columns=None, history=None):
        """
        TODO: update.

            Immutable lazy data for all machine learning scenarios
             we could imagine.
             Matrices Y, Z, V and W can be accessed as vectors y, z, v and w
             if there is only one output.
             Otherwise, they are multitarget matrices.

             Vector e (or f) summarizes e.g. some operation over y,z (or v,w)
             k is a vector indicating the time step of each row.
             Vector k has the time steps of each instance, t has a single time
             step for the entire chunk (all rows in Data). t == k[0]

             For instance, when the vectors e,f,k have only one value,
             which is the most common scenario, they can be accessed as values
             r,s,t. r == e[0] and s == f[0]

             Vector of metafeatures 'm' can have any length, which is
             dependent on the MF() component used and explicited in history.

        :param name:

        :param X: attribute values
        :param Y: labels
        :param Z: Predictions
        :param P: Predicted probabilities
        :param e: Results, summarized in some way

        :param U: X of unlabeled set
        :param V: Y of unlabeled set
        :param W: Predictions for unlabeled set
        :param Q: Predicted probabilities for unlabeled set
        :param f: Results for unlabeled, summarized in some way

        :param k: Time steps for the current chunk (this Data), one per row

        :param l: supervised metafeatures characterizing this Data's X and y
        :param m: unsupervised metafeatures characterizing this Data's X+U

        :param columns: attribute names
        :param history: History of transformations suffered by the data

        #:param modified: list of fields modified in the last transformation

        :param task: intended use for this data: classification, clustering,...
        :param n_classes: this could be calculated,
            but it is too time consuming to do it every transformation.
        """
        # ALERT: all single-letter args will be considered matrices/vectors!
        self.fields = {k: v for k, v in locals().items() if len(k) == 1}
        self.fields['C'] = Chain() if C is None else C
        self.__dict__.update(self.fields)

        self.name = f'unnamed[{self.uuid()}]' if name is None else name
        self.history = history

        if Y is not None:
            self.n_classes = np.unique(Y).shape[0]
        if X is not None:
            self.n_instances = X.shape[0]
            self.n_attributes = X.shape[1]

        # Add vectorized shortcuts for matrices.
        for k, v in self._vectors.items():
            self.__dict__[k] = self._matrix_to_vector(self.__dict__[v])

        # Add scalar shortcuts for matrices.
        for k, v in self._scalars.items():
            self.__dict__[k] = self._matrix_to_scalar(self.__dict__[v])

        # Add lazy cache for dump and uuid
        for var in self.fields:
            self.__dict__['_dump' + var] = None
            self.__dict__['uuid' + var] = None

        self._uuid = None
        self._history_uuid = None
        self._history_dump = None
        self._name_uuid = None
        self.Xy = X, self.__dict__['y']
        self.columns = columns

    def field_dump(self, field_name):
        """
        Dump of the matrix/vector associated to the given field.
        :param field_name:
        :return: binary compressed dump
        """
        if field_name not in self.fields:
            raise Exception(f'Field {field_name} not available in this Data!')

        key = '_dump' + field_name
        if self.__dict__[key] is None:
            self.__dict__[key] = pack_data(self.fields[field_name])
        return self.__dict__[key]

    def field_uuid(self, field_name):
        """
        UUID of the matrix/vector associated to the given field.
        :param field_name: Case sensitive.
        :return: UUID
        """
        key = 'uuid' + field_name
        if self.__dict__[key] is None:
            uuid_ = uuid(self.field_dump(field_name))
            self.__dict__[key] = uuid_
        return self.__dict__[key]

    # remover ?
    def uuids_dumps(self):
        """
        :return: pair uuid-dump of each matrix/vector.
        """
        return {self.field_uuid(k): self.field_dump(k) for k in self.fields}

    # remover ?
    def uuids_fields(self):
        """
        :return: pair uuid-field of each matrix/vector.
        """
        return {self.field_uuid(k): k for k in self.fields}

    @staticmethod
    def read_arff(file, target=None):
        """
        Create Data from ARFF file.
        See read_data_frame().
        :param file:
        :param target:
        :return:
        """
        data = arff.load(open(file, 'r'), encode_nominal=True)
        df = pd.DataFrame(data['data'],
                          columns=[attr[0] for attr in data['attributes']])
        return Data.read_data_frame(df, file, target)

    @staticmethod
    def read_csv(file, target=None):
        """
        Create Data from CSV file.
        See read_data_frame().
        :param file:
        :param target:
        :return:
        """
        df = pd.read_csv(file)  # 1169_airlines explodes here with RAM < 6GiB
        return Data.read_data_frame(df, file, target)

    @staticmethod
    def read_data_frame(df, file, target=None):
        """
        ps. Assume X,y classification task.
        Andd that there was no transformations (history) on this Data.
        :param df:
        :param file: name of the dataset (if a path, name will be extracted)
        :param target:
        :return:
        """
        name = file.split('/')[-1]
        Y = target and Data._as_column_vector(
            df.pop(target).values.astype('float'))
        X = df.values.astype('float')  # Do not call this before setting Y!
        return Data(name=name, X=X, Y=Y, columns=list(df.columns))

    @staticmethod
    def random_classification(n_attributes, n_classes, n_instances):
        """
        ps. Assume X,y classification task.
        :param n_attributes:
        :param n_classes:
        :param n_instances:
        :return:
        """
        X, y = ds.make_classification(n_samples=n_instances,
                                      n_features=n_attributes,
                                      n_classes=n_classes,
                                      n_informative=int(
                                          np.sqrt(2 * n_classes)) + 1)
        return Data(X=X, Y=Data._as_column_vector(y))

    @staticmethod
    def read_from_storage(name, storage, fields=None):
        """
        To just recover an original dataset you can pass fields='X,Y'
        (case sensitive)
        or just None to recover as many fields as stored at the moment.
        :param name:
        :param storage:
        :param fields: if None, get complete Data, including predictions if any
        :return:
        """
        return storage.get_data_by_name(name, fields)

    def store(self, storage):
        storage.store_data(self)

    def updated(self, component, **kwargs):
        """ Return a new Data updated by given values.
        :param component: for history purposes
        :param kwargs:
        :return:
        """
        new_args = self.fields.copy()
        for field, value in kwargs.items():
            if field in self._vectors:
                new_args[self._vectors[field]] = self._as_column_vector(value)
            elif field in self._scalars:
                new_args[self._scalars[field]] = np.array(value, ndmin=2)
            else:
                new_args[field] = value

        new_args['name'] = kwargs['name'] if 'name' in kwargs else self.name

        new_args['columns'] = kwargs['columns'] \
            if 'columns' in kwargs else self.columns

        if 'history' in kwargs:
            new_args['history'] = kwargs['history']
        else:
            new_args['history'] = (component.config, component.op, self.history)

        if 'C' in kwargs:
            new_args['C'] = kwargs['C']
        return Data(**new_args)

    def get_matrix(self, name):
        return object.__getattribute__(self, self.from_alias[name])

    def get(self, name):
        """
        User-friendly access to matrices, vectors and scalars by name.
        Parameters
        ----------
        name

        Returns
        -------

        """
        mat = object.__getattribute__(self, self.from_alias[name])
        if name in self._vectors:
            return self._as_vector(mat)
        elif name in self._scalars:
            return mat[0][0]
        else:
            return mat

    def sid(self):
        """
        Short uuID
        First 10 chars of uuid for printing purposes.
        Max of 1 collision each 1048576 combinations.
        :return:
        """
        return self.uuid()[:10]

    def uuid(self):
        if self._uuid is None:
            # The scenario when a dataset with the same name and fields
            # has more than a row in the storage is when different
            # models provide different dataset predictions/transformations.
            # This is solved by adding the history_uuid of transformations
            # into the data.UUID.
            self._uuid = uuid((self.name_uuid() + self.history_uuid()).encode())

        return self._uuid

    def name_uuid(self):
        if self._name_uuid is None:
            self._name_uuid = uuid(self.name.encode())
        return self._name_uuid

    def history_uuid(self):
        if self._history_uuid is None:
            self._history_uuid = uuid(self.history_dump())
        return self._history_uuid

    def history_dump(self):
        if self._history_dump is None:
            self._history_dump = zlibext_pack(self.history)
        return self._history_dump

    def __str__(self):
        txt = []
        [txt.append(f'{k}: {str(v.shape)}')
         for k, v in self.fields.items() if v is not None]
        child = self.history
        h = 'History\n'
        htab = ''
        while child:
            h += htab + str(child[0], child[1]) + '\n'
            child = child[2]
            htab += '    '
        return '\n'.join(txt) + "\nname: " + self.name + "\n" + h

    def split(self, test_size=0.25, fields=None, random_state=0):
        fields = ['X', 'Y'] if fields is None else fields

        cv_exp = CV(
            config={'random_state': random_state, 'split': 'holdout',
                    'test_size': test_size, 'steps': 1,
                    'fields': fields}
        ).iterations(self)[0]

        return cv_exp.apply(self), cv_exp.use(self)

    @staticmethod
    def _as_vector(mat):
        s = mat.shape[0]
        return mat.reshape(s)

    @staticmethod
    def _as_column_vector(vec):
        return vec.reshape(len(vec), 1)

    @staticmethod
    def _matrix_to_vector(m, default=None):
        return default if m is None else Data._as_vector(m)

    @staticmethod
    def _matrix_to_scalar(m, default=None):
        return default if m is None else m[0][0]

    # # Todo: Jogar fora?
    # def field_names(self) -> str:
    #     if self.fields is None:
    #         sortd = list(self.fields.keys())
    #         sortd.sort()
    #         self._set('_fields', ','.join(sortd))
    #     return self.fields

    def merged(self, new_data):
        """
        Get more matrices/vectors (or new values) from another Data.
        The longest history will be kept.
        They should have the same name and 'new_data' history should contain
        the history of self.
        Otherwise, an exception will be raised.
        new_data has precedence over self.
        :param new_data:
        :return:
        """
        # checking...
        if self.name != new_data.name:
            raise Exception(f'Merging {self.name} with {new_data.name}')

        # Check if new history includes the old one.
        newnode = new_data.history
        newrev = []
        while newnode is not None:
            newrev.append(newnode)
            newnode = newnode[2]

        oldnode = self.history
        oldrev = []
        while oldnode is not None:
            oldrev.append(oldnode)
            oldnode = oldnode[2]

        while oldrev:
            newnode = newrev.pop()
            oldnode = oldrev.pop()
            if oldnode[0] != newnode[0]:
                print('>>', new_data.history, '<<\n>>', self.history, '<<')
                print('>>', newnode[0], '<<\n>>', oldnode[0], '<<')
                print('>>', newnode[1], '<<\n>>', oldnode[1], '<<')
                raise Exception('Incompatible transformations, self.history '
                                'should be the start of new_data.history')

        newfields = {k: v for k, v in new_data.fields.items() if v is not None}
        hist = new_data.history
        if newnode is None:
            hist = ('MergedWith:' + ''.join(newfields.keys()), 'm', hist)

        dic = self.fields.copy()
        dic.update(newfields)

        return Data(name=self.name, history=hist, columns=self.columns, **dic)

    # def select(self, fields):
    #     """
    #     Conveniently converts field names to return the dict of matrices/vectors
    #     ps.: Automatically convert vectorized/single-valued shortcuts to
    #     matrices/vectors.
    #     ps 2: ignore inexistent fields
    #     ps 3: raise exception in none fields
    #     :param fields: 'all' means 'don't touch anything'
    #     'except:z,w' means keep all except z,w
    #     :return: a subset of the dictionary of kwargs.
    #     """
    #     fields.replace(':', ',')
    #     if fields == 'all':
    #         fields_lst = self.fields()
    #     else:
    #         fields_lst = fields.split(',')
    #         if fields_lst and fields_lst[0] == 'except':
    #             fields_lst = sub(self.fields(), fields_lst[1:])
    #
    #     namesmats = [(x.upper() if x in self.vectors() else x)
    #                  for x in fields_lst]
    #     namesvecs = [vec for val, vec in self._val2vec.items()
    #                  if val in fields_lst]
    #     names = namesmats + namesvecs
    #
    #     # Raise exception if any requested matrix is None.
    #     if any([mv not in self.matvecs() for mv in names]):
    #         raise Exception('Requested None or inexistent matrix/vector/value',
    #                         fields, self.matvecs().keys)
    #
    #     return {k: v for k, v in self.matvecs().items() if k in names}
