import socket

import pymysql
import pymysql.cursors

from paje.storage.sql import SQL


class MySQL(SQL):
    def __init__(self, server='user:pass@ip', db='curumim',
                 debug=False, read_only=False, nested=None, sync=False):
        super().__init__(nested_storage=nested, sync=sync)
        self.info = server + ', ' + db
        self.read_only = read_only
        self.database = server
        credentials, self.host = server.split('@')
        self.user, self.password = credentials.split(':')
        self.db = db
        self.debug = debug
        if '-' in db:
            raise Exception("'-' not allowed in db name!")
        self.hostname = socket.gethostname()
        self._open()

    def _open(self):
        """
        Each reconnection has a cost of approximately 150ms in ADSL (ping=30ms).
        :return:
        """
        if self.debug:
            print('getting connection...')
        self.connection = pymysql.connect(host=self.host,
                                          user=self.user,
                                          password=self.password,
                                          charset='utf8mb4',
                                          cursorclass=pymysql.cursors.DictCursor)
        # self.connection.client_flag &= pymysql.constants.CLIENT.MULTI_STATEMENTS
        self.connection.autocommit(True)

        if self.debug:
            print('getting cursor...')
        self.cursor = self.connection.cursor()

        # Create db if it doesn't exist yet.
        self.query(f"SHOW DATABASES LIKE '{self.db}'")
        setup = self.get_one() is None
        if setup:
            if self.debug:
                print('creating database', self.db, 'on', self.database, '...')
            self.cursor.execute("create database if not exists " + self.db)

        if self.debug:
            print('using database', self.db, 'on', self.database, '...')
        self.cursor.execute("use " + self.db)

        if setup:
            self._setup()
        return self

    def _now_function(self):
        return 'now()'

    def _auto_incr(self):
        return 'AUTO_INCREMENT'

    def _keylimit(self):
        return '(190)'

    def _on_conflict(self, fields=None):
        return 'ON DUPLICATE KEY UPDATE'
