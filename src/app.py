from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import os
from sql_cmd import SQL
import time
import sys
from face_emotion import emotions
from class_score import class_score

# Flaskアプリケーションの設定
app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = secrets.token_hex(16)

# データベースファイルのパス
db_name = r"./Graduation_work.db"
sql = SQL(db_file=db_name)

# アップロードフォルダの設定
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'video')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'mp4'}

print(f"Upload folder: {UPLOAD_FOLDER}")
print(f"Current working directory: {os.getcwd()}")
print(f"__file__: {__file__}")
print(f"Absolute path of __file__: {os.path.abspath(__file__)}")

# アップロードフォルダが存在しない場合は作成
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload folder: {UPLOAD_FOLDER}")

# ファイルの拡張子を確認する関数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Userクラスの定義
class User(UserMixin):
    def __init__(self, id, name, password):
        self.id = id
        self.name = name
        self.password = password

# ログインマネージャーの設定
login_manager = LoginManager()
login_manager.init_app(app)

# ユーザーをロードする関数
@login_manager.user_loader
def load_user(user_id):
    cmd = f"SELECT user_id, user_name, password FROM users WHERE user_id = {user_id}"
    result = sql(cmd)
    if result:
        user_data = result[0]
        user = User(user_data[0], user_data[1], user_data[2])
        return user
    else:
        return None

# ログイン画面のルート
@app.route('/')
def index():
    return render_template('login.html')

# ログイン処理のルート
@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cmd = f"SELECT user_id, user_name, password FROM users WHERE user_name = '{username}'"
        result = sql(cmd)
        if result:
            user = result[0]
            if check_password_hash(user[2], password):
                user_obj = User(user[0], user[1], user[2])
                login_user(user_obj)
                flash('ログインしました', 'success')
                return redirect(url_for('home'))
            else:
                flash('ユーザー名またはパスワードが正しくありません', 'error')
                return redirect(url_for('index'))
        else:
            flash('ユーザー名またはパスワードが正しくありません', 'error')
            return redirect(url_for('index'))

# ユーザー登録のルート
@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        cmd = "SELECT user_id FROM users WHERE user_name = ?"
        result = sql(cmd, (username,))
        if result:
            flash('このユーザー名は既に使用されています', 'error')
            return redirect(url_for('register'))

        cmd = "INSERT INTO users (user_name, password) VALUES (?, ?)"
        sql(cmd, (username, hashed_password))
        flash('ユーザー登録が完了しました', 'success')
        return redirect(url_for('index'))

# ログアウト処理のルート
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('ログアウトしました', 'success')
    return redirect(url_for('index'))

# ホーム画面のルート
@app.route('/home')
@login_required
def home():
    user_id = current_user.id
    cmd = "SELECT * FROM class_info WHERE user_id = ? ORDER BY position"
    class_info = sql(cmd, (user_id,))
    print(class_info)
    return render_template('home.html', class_info=class_info)

# クラス作成のルート
@app.route('/create_class', methods=['POST'])
@login_required
def create_class():
    if request.method == 'POST':
        data = request.json
        class_name = data.get('class_name')
        teacher = data.get('teacher')

        print(f"Received data: class_name={class_name}, teacher={teacher}")

        if not class_name or not teacher:
            return jsonify({'error': 'クラス名と担当教師を入力してください', 'received_data': data}), 400

        user_id = current_user.id   # 現在ログインしているユーザーのIDを取得

        # 現在の最大positionを取得
        cmd = "SELECT MAX(position) FROM class_info WHERE user_id = ?"
        result = sql(cmd, (user_id,))
        max_position = result[0][0] if result[0][0] is not None else 0
        print(max_position)


        try:
            cmd = "INSERT INTO class_info (class_name, teacher, user_id, position) VALUES (?, ?, ?,?)"
            sql(cmd, (class_name, teacher, user_id, max_position + 1))
            # userテーブルのクラスを+1する処理を追加
            cmd = f"""UPDATE users SET class = class + 1 WHERE class_id = {user_id}"""
            sql(cmd)

            return jsonify({'message': 'クラスが作成されました'}), 200
        except Exception as e:
            print(f"Error creating class: {str(e)}")
            return jsonify({'error': 'クラスの作成中にエラーが発生しました', 'details': str(e)}), 500

    return jsonify({'error': 'リクエストが不正です'}), 400

# クラス順序更新のルート
@app.route('/update_class_order', methods=['POST'])
@login_required
def update_class_order():
    try:
        data = request.json
        order = data.get('order', [])

        # トランザクション処理用のクエリリスト
        queries = []
        for index, class_id in enumerate(order):
            queries.append(("UPDATE class_info SET position = ? WHERE class_id = ?", (index, class_id)))


        # トランザクション処理
        if sql.transaction(queries):
            return jsonify({'success': True})
        else:
            return jsonify({'success': False})

    except Exception as e:
        print(f"Error updating class order: {e}")
        return jsonify({'success': False})

# データベースのクラスの順序を更新する処理
def update_class_order_in_db(class_id, position):
    cmd = "UPDATE class_info SET position = ? WHERE class_id = ?"
    sql(cmd, (position, class_id))

# クラスの詳細ページ
@app.route('/class/<int:class_id>')
@login_required
def class_detail(class_id):
    user_id = current_user.id
    cmd = "SELECT * FROM class_info WHERE class_id = ? AND user_id = ?"
    class_info = sql(cmd, (class_id, user_id))
    cmd = "SELECT * FROM class_info WHERE user_id = ? ORDER BY position"
    class_all_info = sql(cmd, (user_id,))

    if not class_info:
        flash('クラスが見つかりません', 'error')
        return redirect(url_for('home'))

    print(f"class_id: {class_id}")
    cmd = f"""
        SELECT class_detail_id, class_time, neutral, happy, sad, surprised, anger, count, class_score
        FROM class_detail
        WHERE class_id = {class_id}
    """
    class_result_data = sql(cmd)

    print(f"class_info: {class_info}")
    print(f"class_result_data: {class_result_data}")

    if class_result_data is None:
        class_result = []
    else:
        class_result = [list(data) for data in class_result_data]
        for i in range(len(class_result)):
            for j in range(2, 7):
                count = class_result[i][7]  # countのインデックスを7に修正
                if count > 0:
                    score = class_result[i][j]
                    score = (int((score / count * 10) // 1)) / 10
                    class_result[i][j] = score
                else:
                    class_result[i][j] = 0
    print(class_info[0])
    return render_template('class_detail.html', class_info=class_info, class_result=class_result, class_all_info=class_all_info)

# ファイルアップロードのルート
@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Current working directory: {os.getcwd()}")

    if 'file' not in request.files:
        return jsonify({'error': 'ファイルが選択されていません'}), 400
    file = request.files['file']
    #授業IDを取得
    class_id = request.form.get('class_id')
    if file.filename == '':
        return jsonify({'error': 'ファイルが選択されていません'}), 400
    if file and allowed_file(file.filename):
        cmd = "SELECT COUNT(*) FROM class_detail WHERE class_id = ?"
        result = sql(cmd, (class_id,))
        class_count = int(sql.result(result))  # 文字列を整数に変換
        filename = f"class_{class_id}_{class_count + 1}.mp4"
        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(f"File saved at: {file_path}")
            print(f"File exists: {os.path.exists(file_path)}")

            start = time.time()  # 現在時刻（処理開始前）を取得


            # 実行したい処理を記述
            neu, hap, sad, sur, ang, count, score = emotions(file_path, class_id)

            end = time.time()  # 現在時刻（処理完了後）を取得
            time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
            print(f"処理にかかった時間:{time_diff}")  # 処理にかかった時間データを使用


            return jsonify({'message': f'動画が正常に送信されました\n neu : {neu}, hap : {hap}, sad : {sad}, sur : {sur}, ang : {ang}, count : {count}\n 授業スコア: {score}'}), 200
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return jsonify({'error': 'ファイルの保存中にエラーが発生しました'}), 500
    else:
        return jsonify({'error': '許可されていないファイル形式です'}), 400

# モジュールのエクスポート
if __name__ == '__main__':
    app.run(debug=True)