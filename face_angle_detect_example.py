from FaceAngleDetector import FaceAngleDetector

import cv2
import numpy as np 
import psutil

# FaceAngleDetector 클래스 인스턴스화
detector = FaceAngleDetector()

cap = cv2.VideoCapture(0)  # 웹캠 인덱스

while cap.isOpened():
    ret, img = cap.read()
    if ret:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        
        # 얼굴 각도 및 방향 계산
        pitch_pred, yaw_pred, roll_pred, nose_x, nose_y = detector.get_face_angles(img)
        
        if pitch_pred is not None:
            
            # 얼굴 방향 텍스트 생성
            text = detector.get_face_direction(pitch_pred, yaw_pred)
            # 라디안을 각도로 변환
            pitch_pred_deg = pitch_pred * 180 / np.pi
            yaw_pred_deg = yaw_pred * 180 / np.pi
            roll_pred_deg = roll_pred * 180 / np.pi
            
            # cpu 및 ram 사용량 계산
            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent
            # cpu 사용량 평균 계산
            average_cpu_usage = detector.get_average_cpu_usage(cpu_usage)

            # 얼굴 방향 축 그리기
            img = detector.draw_axes(img, pitch_pred, yaw_pred, roll_pred, int(nose_x * img.shape[1]), int(nose_y * img.shape[0]))

            img = detector.render_face_pose_stats(img, text, pitch_pred_deg, yaw_pred_deg, roll_pred_deg, average_cpu_usage, ram_usage)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('img', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()


