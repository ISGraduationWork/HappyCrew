import sqlite3
from sqlite3 import Error

class SQL:
    def __init__(self, db_file):
        """
        コンストラクタ。データベースファイルのパスを受け取り、初期化する。
        Args:
            db_file (str): データベースファイルのパス。デフォルトは './Graduation_work.db'。
        """
        self.db_file = db_file
        self.connection = None

    def __call__(self, query, params=()):
        """
        SQLクエリを実行するためのメソッド。クエリとパラメータを受け取る。
        Args:
            query (str): 実行するSQLクエリ。
            params (tuple): クエリに使用するパラメータ。デフォルトは空のタプル。

        Returns:
            list: クエリの結果を返す。エラーが発生した場合はNoneを返す。
        """
        try:
            # データベースへの接続を確立
            self.connection = sqlite3.connect(self.db_file)
            cursor = self.connection.cursor()
            # クエリの実行
            cursor.execute(query, params)
            # 結果の取得
            result = cursor.fetchall()
            # 変更をコミット
            self.connection.commit()
            return result
        except Error as e:
            print(f"Error executing query: {e}")
            return None
        finally:
            # 接続を閉じる
            if self.connection:
                self.connection.close()

    def __del__(self):
        """
        デストラクタ。オブジェクトが破棄されるときに接続を閉じる。
        """
        if self.connection:
            self.connection.close()

    def result(self, result):
        """
        クエリの結果をフォーマットして出力するメソッド。
        Args:
            result (list): クエリの結果のリスト。
        """
        if result:
            for row in result:
                return ", ".join([f"{value}" for value in row])
        else:
            print("No results or error occurred.")
            return "None"

    def transaction(self, queries):
        """
        トランザクションを使って複数のクエリを実行するメソッド。
        Args:
            queries (list of tuples): (クエリ, パラメータ) のリスト。

        Returns:
            bool: トランザクションが成功したかどうか。
        """
        try:
            self.connection = sqlite3.connect(self.db_file)
            cursor = self.connection.cursor()
            for query, params in queries:
                cursor.execute(query, params)
            self.connection.commit()
            return True
        except Error as e:
            print(f"Transaction error: {e}")
            self.connection.rollback()
            return False
        finally:
            if self.connection:
                self.connection.close()