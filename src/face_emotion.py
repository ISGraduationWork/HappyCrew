import cv2
import numpy as np
import openvino.runtime as ov
import os
import time
import sys
from sql_cmd import SQL
from class_score import class_score
from concurrent.futures import ThreadPoolExecutor

def process_emotion(face_image, model_emotion, ie, xmin, ymin, xmax, ymax):
    # 感情分析のためのInferRequestを個別に作成
    infer_request_emotion = model_emotion.create_infer_request()

    # 顔画像を感情モデルに適合するようにリサイズし、前処理を行う
    img = cv2.resize(face_image, (64, 64))  # サイズ変更
    img = img.transpose((2, 0, 1))  # HWC > CHW
    img = np.expand_dims(img, axis=0)  # 次元合せ

    # 推論実行（感情認識）
    out = infer_request_emotion.infer(inputs={model_emotion.input(0): img})
    out = next(iter(out.values()))  # 出力テンソルを取得
    out = out.squeeze()  # 不要な次元を削除

    # 感情分析のスコアを計算
    emotion_scores = [int(out[i] * 100) for i in range(len(out))]
    index_max = np.argmax(out) # 最大スコアのインデックスを取得

    # 各感情の文字列をリスト化
    list_emotion = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    emotion_text = list_emotion[index_max]  # 最大スコアに対応する感情を取得
    emotion_colors = {
        'neutral': (255, 255, 255),  # 白
        'happy': (0, 165, 255),  # オレンジ
        'sad': (255, 0, 0),  # 青
        'surprise': (0, 255, 255),  # 黄色
        'anger': (0, 0, 255)  # 赤
    }

    # 文字列描画
    cv2.putText(face_image, emotion_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_colors[emotion_text], 1)

    # バウンディングボックス表示
    cv2.rectangle(face_image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)  # 緑

    # 結果を返す（フレームと感情スコア）
    return face_image, emotion_scores, emotion_text, index_max

def emotions(file_path, class_id):
    # SQLクラスのインスタンスを作成
    db_name = r"./Graduation_work.db"
    sql = SQL(db_file=db_name)

    # IEコアの初期化
    ie = ov.Core()

    # モデルの準備（顔検出）
    file_path_face = r'.\\face-detection-retail-0004'
    model_face = file_path_face + '.xml'

    # モデルの準備（感情分類）
    file_path_emotion = r'.\\emotions-recognition-retail-0003'
    model_emotion = ie.compile_model(file_path_emotion + '.xml', device_name='CPU')

    # モデルの読み込み（顔検出）
    model_face = ie.compile_model(model_face, device_name='CPU')
    infer_request_face = model_face.create_infer_request()

    print(f"入力情報 : file_path: {file_path}, class_id: {class_id}")
    print(f"入力ファイルパス: {file_path}")
    print(f"ファイルが存在するか: {os.path.exists(file_path)}")

    # 動画の読み込み
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("エラー: 動画ファイルを開けませんでした。")
        return None

    # 動画の情報を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 動画の幅
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 動画の高さ
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # フレームレート
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 総フレーム数
    analysis_time_sec = int((total_frames / fps)) # 分析時間

    TIME =  int((total_frames / fps) // 1) #小数点以下は切り捨て
    if TIME >= 60:
        TIME_m = TIME // 60
        TIME_s = TIME % 60
        TIME_str = str(f"{TIME_m}分{TIME_s}秒")
    else:
        TIME_str = str(f"{TIME}秒")

    print(f"動画情報: 幅={width}, 高さ={height}, FPS={fps}, 総フレーム数={total_frames}")



    # 出力動画ファイルのパス
    output_filename = os.path.splitext(os.path.basename(file_path))[0] + '_emotions.avi'
    output_path = os.path.abspath(os.path.join(os.path.dirname(file_path), output_filename))

    print(f"OpenCV version: {cv2.__version__}")
    print(f"出力ファイルパス: {output_path}")

    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Initial VideoWriter object type: {type(video_writer)}")
    print(f"Initial VideoWriter is opened: {video_writer.isOpened()}")

    if not isinstance(video_writer, cv2.VideoWriter):
        print(f"エラー: VideoWriterの初期化に失敗しました。型: {type(video_writer)}")
        return

    if not video_writer.isOpened():
        print(f"エラー: 出力動画ファイルを開けませんでした。パス: {output_path}")
        print(f"VideoWriter パラメータ: fourcc={fourcc}, fps={fps}, size=({width}, {height})")
        return

    # メインループの直前で再度確認
    print(f"Before main loop - VideoWriter object type: {type(video_writer)}")
    print(f"Before main loop - VideoWriter is opened: {video_writer.isOpened()}")

    # 顔追跡用の変数
    face_id_counter = 1
    faces_dict = {}

    # 感情の累積値を追跡するための変数
    total_emotions = {'neu': 0, 'hap': 0, 'sad': 0, 'sur': 0, 'ang': 0}
    emotion_count = 0   # 感情分析の実行回数

    # メインループ
    frame_count = 0  # 処理したフレーム数
    start_time = time.time() # 処理開始時刻を記録

    while cap.isOpened():
        ret, frame = cap.read() # フレームの読み込み
        if not ret:
            break

        frame_count += 1

        if frame_count % 10 == 0:  # 10フレームごとに状態を出力
            print(f"処理中のフレーム: {frame_count}")

        # 入力データフォーマットへ変換
        img = cv2.resize(frame, (300, 300))  # 顔検出モデルの入力サイズに合わせる
        img = img.transpose((2, 0, 1))  # HWC > CHW
        img = np.expand_dims(img, axis=0)  # 次元合せ

        # 推論実行（顔検出）
        out_face = infer_request_face.infer(inputs={model_face.input(0): img})
        out_face = next(iter(out_face.values()))  # 出力テンソルを取得
        out_face = out_face.squeeze()  # 不要な次元を削除

        # 検出された人数をカウント
        face_count = 0
        current_faces = []

        # 並列での感情分析処理
        with ThreadPoolExecutor() as executor:
            future_emotions = []

            # 検出されたすべての顔領域に対して１つずつ処理
            for detection in out_face:
                # conf値の取得
                confidence = float(detection[2])

                # バウンディングボックス座標を入力画像のスケールに変換
                xmin = int(detection[3] * frame.shape[1])
                ymin = int(detection[4] * frame.shape[0])
                xmax = int(detection[5] * frame.shape[1])
                ymax = int(detection[6] * frame.shape[0])

                # conf値が0.5より大きい場合のみ感情推論とバウンディングボックス表示
                if confidence > 0.5:
                    face_count += 1  # 人数をカウント

                    # 顔領域のみ切り出し
                    frame_face = frame[ymin:ymax, xmin:xmax]

                    # 感情分析を並列実行
                    future_emotions.append(
                        executor.submit(process_emotion, frame_face, model_emotion, ie, xmin, ymin, xmax, ymax)
                    )

            # 並列実行の結果をフレームに反映
            for future in future_emotions:
                if future.result():
                    frame, emotion_score, emotion_text, emotion_index = future.result()
                    for i, emotion in enumerate(['neu', 'hap', 'sad', 'sur', 'ang']):
                        total_emotions[emotion] += emotion_score[i]
                    emotion_count += 1
        # 現在までの処理時間と残り時間を計算
        elapsed_time = time.time() - start_time  # 開始からの経過時間（秒）
        processed_ratio = frame_count / cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 処理した割合
        estimated_total_time = elapsed_time / processed_ratio  # 総時間の予測
        remaining_time_sec = estimated_total_time - elapsed_time  # 残り時間
        remaining_time_min = remaining_time_sec // 60
        remaining_time_sec = remaining_time_sec % 60
        sys.stdout.write("\r残り分析時間: {}分 {}秒".format(int(remaining_time_min), int(remaining_time_sec)))
        sys.stdout.flush()

        # 検出されなくなった顔をリストから削除
        for face_id in list(faces_dict.keys()):
            if face_id not in current_faces:
                del faces_dict[face_id]

        # 人数を左上に出力
        cv2.putText(frame, f'People: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # フレームを出力動画に書き込む
        try:
            if isinstance(frame, np.ndarray):
                if isinstance(video_writer, cv2.VideoWriter) and video_writer.isOpened():
                    video_writer.write(frame)
                else:
                    print(f"エラー: VideoWriterが無効です。型: {type(video_writer)}, isOpened: {video_writer.isOpened() if isinstance(video_writer, cv2.VideoWriter) else 'N/A'}")
                    break
            else:
                print(f"エラー: frameがnumpy.ndarrayではありません。型: {type(frame)}")
        except Exception as e:
            print(f"フレーム書き込みエラー: {e}")
            print(f"Current VideoWriter object type: {type(video_writer)}")
            break

    # 1行追加
    cmd = f"""
    INSERT INTO class_detail (class_id)
    VALUES ({class_id});
    """
    sql(cmd)

    #class_detail_idを取得
    cmd = f"""
    SELECT MAX(class_detail_id)
    FROM class_detail
    WHERE class_id = {class_id}
    """
    result = sql(cmd)
    print(result)
    class_detail_id = sql.result(result)
    print(f"class_detail_id: {class_detail_id}")

    # 授業スコアを算出
    neu_avg = total_emotions['neu'] / emotion_count
    hap_avg = total_emotions['hap'] / emotion_count
    sad_avg = total_emotions['sad'] / emotion_count
    sur_avg = total_emotions['sur'] / emotion_count
    ang_avg = total_emotions['ang'] / emotion_count
    score = int(class_score(neu_avg, hap_avg, sad_avg, sur_avg, ang_avg) // 1)

    # データベースを更新
    print("データベース更新処理")
    cmd = f"""
    UPDATE class_detail
    SET class_time = "{TIME_str}",
        neutral = neutral + {total_emotions['neu']},
        happy = happy + {total_emotions['hap']},
        sad = sad + {total_emotions['sad']},
        surprised = surprised + {total_emotions['sur']},
        anger = anger + {total_emotions['ang']},
        count = count + {emotion_count},
        class_score = {score}
    WHERE class_detail_id = {class_detail_id};
    """
    sql(cmd)
    print("データベース更新完了")

    # 更新後のデータを確認
    result = sql(f"SELECT * FROM class_detail WHERE class_id = {class_id}")
    if result:
        print(sql.result(result))
    else:
        print("No results or error occurred.")

    # 終了処理
    cap.release()
    if isinstance(video_writer, cv2.VideoWriter):
        video_writer.release()
    else:
        print(f"エラー: video_writerがVideoWriterオブジェクトではありません。型: {type(video_writer)}")

    cv2.destroyAllWindows()


    print(f"処理が完了しました。")
    print(f"入力ファイル: {file_path}")
    print(f"出力ファイル: {output_path}")
    print(f"処理されたフレーム数: {frame_count}")

    # 感情の値を返す
    cmd = f"""
    SELECT neutral, happy, sad, surprised, anger, count, class_score
    FROM class_detail
    WHERE class_detail_id = {class_detail_id}
    """
    result = sql(cmd)
    if result:
        result_str = sql.result(result)
        if result_str and result_str != 'None':
            try:
                neu, hap, sad, sur, ang, count, score = map(int, result_str.split(', '))
                return neu, hap, sad, sur, ang, count, score
            except ValueError as e:
                print(f"Error converting database result to integers: {e}")
        else:
            print("Database returned None or empty result")
    else:
        print("No results or error occurred in database query")

    # エラーの場合はデフォルト値を返す
    print("エラー")
    return 0, 0, 0, 0, 0, 0