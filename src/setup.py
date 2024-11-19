import sqlite3
from sqlite3 import Error
from datetime import datetime

def create_connection(db_file):
    """ SQLiteデータベースに接続する """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"SQLite Database connection successful. SQLite version: {sqlite3.version}")
    except Error as e:
        print(f"Error: '{e}'")
    return conn

def execute_query(connection, query, params=None):
    """ SQLクエリを実行する """
    try:
        cursor = connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
        return cursor.lastrowid
    except Error as e:
        print(f"Error: '{e}'")

# データベースファイルのパス
db_file = "Graduation_work.db"

# データベースに接続
connection = create_connection(db_file)

if connection is not None:
    # テーブルを削除してから再作成する関数
    def recreate_table(connection, table_name, create_table_query):
        drop_table_query = f"DROP TABLE IF EXISTS {table_name};"
        execute_query(connection, drop_table_query)
        execute_query(connection, create_table_query)

    # usersテーブルを再作成
    create_table_users = """
    CREATE TABLE IF NOT EXISTS users (
	user_id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_name TEXT,
	password TEXT
    );
    """
    recreate_table(connection, "users", create_table_users)

    # userテーブルに仮のデータを登録(無し)
    # users_insert = """
    # INSERT INTO users (name, password)
    # VALUES (?, ?);
    # """
    # user1_id = execute_query(connection, users_insert, ('佐藤秀', 'syu'))
    # user2_id = execute_query(connection, users_insert, ('長谷川日向', 'hinata'))


    # class_infoテーブルを作成
    create_table_class_info = """
    CREATE TABLE IF NOT EXISTS class_info (
	class_id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL,
	class_name TEXT NOT NULL,
	teacher TEXT NOT NULL,
    position INTEGER NOT NULL,
	FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """
    recreate_table(connection, "class_info", create_table_class_info)


    # class_detailテーブルを作成
    create_table_class_detail = """
    CREATE TABLE IF NOT EXISTS class_detail (
	class_detail_id INTEGER PRIMARY KEY AUTOINCREMENT,
	class_detail_name TEXT,
	class_id INTEGER,
	class_time TEXT DEFAULT "エラー",
	neutral INTEGER DEFAULT 0,
	happy INTEGER DEFAULT 0,
	sad INTEGER DEFAULT 0,
	surprised INTEGER DEFAULT 0,
	anger INTEGER DEFAULT 0,
	count INTEGER DEFAULT 0,
	class_score INTEGER DEFAULT 0,
	FOREIGN KEY (class_id) REFERENCES class_info(class_id)
    );
    """
    recreate_table(connection, "class_detail", create_table_class_detail)


    # login_historyテーブルを再作成
    create_table_login_history = """
    CREATE TABLE IF NOT EXISTS login_history (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER,
	login_time DATETIME,
	FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """
    recreate_table(connection, "login_history", create_table_login_history)

    # 接続を閉じる
    connection.close()
    print("SQLite connection is closed")
else:
    print("Error! Cannot create the database connection.")