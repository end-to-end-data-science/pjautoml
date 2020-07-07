import socket
import sqlite3

from paje.storage.sql import SQL


class SQLite(SQL):
    def __init__(self, db='/tmp/paje', debug=False, read_only=False,
                 nested_storage=None, sync=False):
        super().__init__(nested_storage=nested_storage, sync=sync)
        self.info = db
        self.read_only = read_only
        self.hostname = socket.gethostname()
        self.database = db + '.db'
        self.debug = debug
        self._open()

    def _open(self):
        # isolation_level=None -> SQLite autocommiting
        # isolation_level='DEFERRED' -> SQLite transactioning
        self.connection = sqlite3.connect(self.database,
                                          isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()

        # Create tables if they don't exist yet.
        try:
            self.query(f"select 1 from res")
        except:
            if self.debug:
                print('creating database', self.database, '...')
            self._setup()

    def _now_function(self):
        return 'datetime()'

    def _auto_incr(self):
        return 'AUTOINCREMENT'

    def _keylimit(self):
        return ''

    def _on_conflict(self, fields=''):
        return f'ON CONFLICT{fields} DO UPDATE SET'
