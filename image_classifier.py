import cv2
import glob
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

## 이미지 표시 함수
def show(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

## 모델 옵션 설정
base_options = python.BaseOptions(model_asset_path='classifier.tflite')
options = vision.ImageClassifierOptions(base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)

## 이미지 파일 불러오기
images_path = glob.glob('image*.jpg')
for image_path in images_path:
    ## Mediapipe 이미지 객체 생성
    image1 = mp.Image.create_from_file(image_path)

    ## Mediapipe 이미지 객체를 numpy 배열로 변환 (OpenCV에서 사용하기 위해)
    image_data = image1.numpy_view()
    image = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

    ## 이미지 분류
    classification_result = classifier.classify(image1)

    ## 분류 결과를 이미지에 추가
    for idx, category in enumerate(classification_result.classifications[0].categories):
        text = f"{category.category_name}: {category.score:.2f}"
        position = (10, 30 + idx * 30)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    ## 결과 이미지 표시
    show(image)
