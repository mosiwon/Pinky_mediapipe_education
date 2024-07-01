import cv2
import mediapipe as mp

# Mediapipe Pose 모듈 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 웹캠에서 비디오 캡처
## 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            break

        # BGR 이미지를 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 성능을 향상시키기 위해 이미지를 쓰기 불가능으로 설정
        image.flags.writeable = False
        results = pose.process(image)

        # 이미지를 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 포즈 랜드마크를 이미지 위에 그리기
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # 결과 이미지를 화면에 표시
        cv2.imshow('MediaPipe Pose', image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

