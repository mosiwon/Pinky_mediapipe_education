import mediapipe as mp
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # for mac, WSL2

def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
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
    cv2.putText(new_img, 'x', tuple(axes_points[:, 0].ravel()), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)
    cv2.putText(new_img, 'y', tuple(axes_points[:, 1].ravel()), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    cv2.putText(new_img, 'z', tuple(axes_points[:, 2].ravel()), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)

    return new_img

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print("비디오에서 프레임을 읽을 수 없습니다.")
        break

    # 얼굴 랜드마크 모델 초기화
    face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawing_specs = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
    results = face_mesh.process(img)
    
    img_h, img_w, img_c = img.shape

    # 아무 얼굴도 감지되지 않은 경우
    if results.multi_face_landmarks == None:
        continue
    else:
        # 얼굴 랜드마크를 이미지 위에 그리기
        for face_landmarks in results.multi_face_landmarks:        
            mp.solutions.drawing_utils.draw_landmarks(image=img, landmark_list=face_landmarks, landmark_drawing_spec=drawing_specs)
            for idx, lm in enumerate(results.multi_face_landmarks[0].landmark):
                if idx == 1:
                    nose_x = lm.x * img_w
                    nose_y = lm.y * img_h

    # 코 위치 좌표를 왼쪽 위에 Text로 표시
    text = "Nose: ({}, {})".format(int(nose_x), int(nose_y))
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # 이미지를 보여주기
    cv2.imshow('Face Mesh', img)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
