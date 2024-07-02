import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import psutil
from collections import deque

class FaceAngleDetector:
    def __init__(self, model_path='./best_model.pkl', max_cpu_history=10):
        # Face Mesh와 모델 로드, CPU 사용량 기록을 위한 초기 설정
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.cpu_usage_history = deque(maxlen=max_cpu_history)
        self.model = pickle.load(open(model_path, 'rb'))
        self.cols = []
        for pos in ['nose_', 'forehead_', 'left_eye_', 'mouth_left_', 'chin_', 'right_eye_', 'mouth_right_']:
            for dim in ('x', 'y'):
                self.cols.append(pos + dim)

    def extract_features(self, img):
        """
        이미지에서 얼굴 특징점을 추출하는 함수
        """
        NOSE = 1
        FOREHEAD = 10
        LEFT_EYE = 33
        MOUTH_LEFT = 61
        CHIN = 199
        RIGHT_EYE = 263
        MOUTH_RIGHT = 291

        result = self.face_mesh.process(img)
        face_features = []
        
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE, RIGHT_EYE]:
                        face_features.append(lm.x)
                        face_features.append(lm.y)

        return face_features

    def normalize(self, poses_df):
        """
        얼굴 특징점을 정규화하는 함수
        - 코를 기준으로 중심 이동
        - 눈과 입 사이의 거리를 이용해 스케일 조정
        """
        normalized_df = poses_df.copy()
        
        for dim in ['x', 'y']:
            # 코를 기준으로 중심 이동
            for feature in ['forehead_' + dim, 'nose_' + dim, 'mouth_left_' + dim, 'mouth_right_' + dim, 'left_eye_' + dim, 'chin_' + dim, 'right_eye_' + dim]:
                normalized_df[feature] = poses_df[feature] - poses_df['nose_' + dim]
            
            # 스케일 조정
            diff = normalized_df['mouth_right_' + dim] - normalized_df['left_eye_' + dim]
            for feature in ['forehead_' + dim, 'nose_' + dim, 'mouth_left_' + dim, 'mouth_right_' + dim, 'left_eye_' + dim, 'chin_' + dim, 'right_eye_' + dim]:
                normalized_df[feature] = normalized_df[feature] / diff
        
        return normalized_df

    def get_face_angles(self, img):
        """
        얼굴의 롤, 피치, 요 각도를 반환하는 함수(라디안 단위입니다.)
        """
        face_features = self.extract_features(img)
        if face_features:
            face_features_df = pd.DataFrame([face_features], columns=self.cols)
            face_features_normalized = self.normalize(face_features_df)
            pitch_pred, yaw_pred, roll_pred = self.model.predict(face_features_normalized).ravel()
            return pitch_pred, yaw_pred, roll_pred, face_features_df['nose_x'].values[0], face_features_df['nose_y'].values[0]
        return None, None, None, None, None

    def get_face_direction(self, pitch_pred, yaw_pred, pitch_threshold=0.3, yaw_threshold=0.3):
        """
        현재 얼굴의 방향을 반환하는 함수
        - pitch_threshold: 위아래 방향의 임계값
        - yaw_threshold: 좌우 방향의 임계값
        - pitch_threshold를 올릴수록 위아래 방향의 감도가 낮아집니다.
        - yaw_threshold를 올릴수록 좌우 방향의 감도가 낮아집니다.
        """
        if pitch_pred > pitch_threshold:
            if yaw_pred > yaw_threshold:
                return 'Top Left'
            elif yaw_pred < -yaw_threshold:
                return 'Top Right'
            return 'Top'
        elif pitch_pred < -pitch_threshold:
            if yaw_pred > yaw_threshold:
                return 'Bottom Left'
            elif yaw_pred < -yaw_threshold:
                return 'Bottom Right'
            return 'Bottom'
        elif yaw_pred > yaw_threshold:
            return 'Left'
        elif yaw_pred < -yaw_threshold:
            return 'Right'
        return 'Forward'

    def draw_axes(self, img, pitch, yaw, roll, tx, ty, size=50):
        """
        얼굴 방향을 시각화하기 위해 축을 그리는 함수
        """
        yaw = -yaw
        rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
        axes_points = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=np.float64)
        axes_points = rotation_matrix @ axes_points
        axes_points = (axes_points[:2, :] * size).astype(int)
        axes_points[0, :] = axes_points[0, :] + tx
        axes_points[1, :] = axes_points[1, :] + ty
        
        new_img = img.copy()
        cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)    
        cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)    
        cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
        return new_img

    def get_average_cpu_usage(self, new_usage, max_len=10):
        """
        평균 CPU 사용량을 계산하는 함수
        """
        self.cpu_usage_history.append(new_usage)
        if len(self.cpu_usage_history) > max_len:
            self.cpu_usage_history.popleft()
        return sum(self.cpu_usage_history) / len(self.cpu_usage_history)

    def render_face_pose_stats(self, img, text='', pitch_pred_deg=None, yaw_pred_deg=None, roll_pred_deg=None, average_cpu_usage=None, ram_usage=None):
        """
        얼굴 자세와 시스템 정보를 화면에 표시하는 함수
        """
        stats = [
            (text, (25, 75)),
            (f'Pitch: {pitch_pred_deg:.2f}' if pitch_pred_deg is not None else '', (25, 125)),
            (f'Yaw: {yaw_pred_deg:.2f}' if yaw_pred_deg is not None else '', (25, 175)),
            (f'Roll: {roll_pred_deg:.2f}' if roll_pred_deg is not None else '', (25, 225)),
            (f"CPU: {average_cpu_usage:.1f}%" if average_cpu_usage is not None else '', (25, 275)),
            (f"RAM: {ram_usage}%" if ram_usage is not None else '', (25, 325))
        ]

        for stat, pos in stats:
            if stat:
                cv2.putText(img, stat, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return img
