import cv2
import numpy as np
import openvino.runtime as ov
import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor

def process_emotion(face_image, model_emotion, ie, xmin, ymin, xmax, ymax):
    # 感情分析のためのInferRequestを作成
    infer_request_emotion = model_emotion.create_infer_request()

    # 顔画像を感情モデルに適合するようにリサイズし、前処理を行う
    img = cv2.resize(face_image, (64, 64))
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # 推論実行（感情認識）
    out = infer_request_emotion.infer(inputs={model_emotion.input(0): img})
    out = next(iter(out.values())).squeeze()

    # 感情スコアを計算
    emotion_scores = [int(out[i] * 100) for i in range(len(out))]
    index_max = np.argmax(out)

    # 感情のテキストとバウンディングボックスを描画
    list_emotion = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    emotion_text = list_emotion[index_max]
    emotion_colors = {
        'neutral': (255, 255, 255),
        'happy': (0, 165, 255),
        'sad': (255, 0, 0),
        'surprise': (0, 255, 255),
        'anger': (0, 0, 255)
    }

    cv2.putText(face_image, emotion_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_colors[emotion_text], 1)
    cv2.rectangle(face_image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)

    return face_image, emotion_scores, emotion_text, index_max

def predict_analysis_time(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("エラー: 動画ファイルを開けませんでした。")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    analysis_time_sec = int((total_frames / fps))

    cap.release()
    return analysis_time_sec

def emotions(file_path):
    ie = ov.Core()

    # モデルの準備
    file_path_face = r'.\\face-detection-retail-0004.xml'
    file_path_emotion = r'.\\emotions-recognition-retail-0003.xml'
    model_face = ie.compile_model(file_path_face, device_name='CPU')
    model_emotion = ie.compile_model(file_path_emotion, device_name='CPU')
    infer_request_face = model_face.create_infer_request()

    # 分析時間を予測
    analysis_time_sec = predict_analysis_time(file_path)
    if analysis_time_sec is None:
        return

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("エラー: 動画ファイルを開けませんでした。")
        return

    # 動画情報の取得
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_filename = os.path.splitext(os.path.basename(file_path))[0] + '_emotions.avi'
    output_path = os.path.abspath(os.path.join(os.path.dirname(file_path), output_filename))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 分析を進める
    frame_count = 0
    start_time = time.time()  # 処理開始時刻を記録

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 顔検出と感情分析
        img = cv2.resize(frame, (300, 300)).transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        out_face = infer_request_face.infer(inputs={model_face.input(0): img})
        out_face = next(iter(out_face.values())).squeeze()

        with ThreadPoolExecutor() as executor:
            future_emotions = []
            for detection in out_face:
                confidence = float(detection[2])
                if confidence > 0.5:
                    xmin = int(detection[3] * frame.shape[1])
                    ymin = int(detection[4] * frame.shape[0])
                    xmax = int(detection[5] * frame.shape[1])
                    ymax = int(detection[6] * frame.shape[0])
                    frame_face = frame[ymin:ymax, xmin:xmax]
                    future_emotions.append(executor.submit(process_emotion, frame_face, model_emotion, ie, xmin, ymin, xmax, ymax))

            for future in future_emotions:
                if future.result():
                    frame, emotion_score, emotion_text, emotion_index = future.result()

        # 現在までの処理時間と残り時間を計算
        elapsed_time = time.time() - start_time  # 開始からの経過時間（秒）
        processed_ratio = frame_count / cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 処理した割合
        estimated_total_time = elapsed_time / processed_ratio  # 総時間の予測
        remaining_time_sec = estimated_total_time - elapsed_time  # 残り時間
        remaining_time_min = remaining_time_sec // 60
        remaining_time_sec = remaining_time_sec % 60

        # 残り時間を表示（上書き表示）
        sys.stdout.write("\r残り分析時間: {}分 {}秒".format(int(remaining_time_min), int(remaining_time_sec)))
        sys.stdout.flush()

        # 動画に結果を保存
        video_writer.write(frame)

    cap.release()
    video_writer.release()


# 実行例
emotions("C:\\Users\\sy200\\noraneko_workspace\\Class Video\\Class video.mp4")
