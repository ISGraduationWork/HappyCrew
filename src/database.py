import sqlite3

# データベースファイル名を指定
dbname = "Graduation_work.db"

# SQLite データベースに接続する
conn = sqlite3.connect(dbname)

cur = conn.cursor()

# テーブルの作成

# ユーザーテーブル
# クラステーブル
# sql =   """
#         CREATE TABLE IF NOT EXISTS classes (
#         class_id INTEGER PRIMARY KEY AUTOINCREMENT,
#         user_id INTEGER NOT NULL,
#         class_name TEXT NOT NULL,
#         teacher TEXT NOT NULL,
#         position INTEGER NOT NULL,
#         FOREIGN KEY (user_id) REFERENCES users(id)
#         );
#         """

# クラステーブル
# sql =   """
#         CREATE TABLE IF NOT EXISTS classes (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         name TEXT NOT NULL,
#         teacher TEXT NOT NULL,
#         user_id INTEGER NOT NULL,
#         FOREIGN KEY (user_id) REFERENCES users(id)
#         );
#         """

# 授業
# class_id = "1" #授業idが1の授業
# sql =   f"""
#         CREATE TABLE IF NOT EXISTS class_{class_id} (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         class_minutes INT,
#         neu INTEGER,
#         hap INTEGER,
#         sad INTEGER,
#         sur INTEGER,
#         ang INTEGER
#         );
#         """

# sql2 = f"""
#         INSERT INTO class_{class_id} (class_minutes, neu, hap, sad, sur, ang)
#         VALUES (5, 24000, 12000, 6000, 3000, 1500),
#         (5, 27000, 15000, 3000, 1500, 3000);
#         """

sql = '''
        ALTER TABLE class_info
        ADD COLUMN position INTEGER NOT NULL DEFAULT 0
'''

cur.execute(sql)
# cur.execute(sql2)

# ユーザーの追加
# cur.execute("INSERT INTO users (name, password, class) VALUES (?, ?, ?)",
#             ('noraneko', 'noraneko2319', 0))

# 変更をコミットする（保存する）
conn.commit()

# 接続を閉じる
conn.close()

print("OK")
