from ultralytics import YOLO
import os
from tumor_segmentation import YOLOTester

def Prediction(prediction_area, model_path, test_path):
    yolo_tester = YOLOTester(model=model_path)
    for i in range(30):
        print('-', end='')
    print('\n')
    print(f"{prediction_area} TUMOR PREDICTION")
    print('\n')
    for i in range(30):
        print('-', end='')

    for img_name in os.listdir(test_path):
        img_path = os.path.join(test_path, img_name)
        results = yolo_tester.make_prediction(img_path)
        print(results)  # or process the results as needed


# brain tumor prediction
brain_tumor_yolo_model_path = '/home/bilal/Desktop/tumor_detection_with_medical_images/runs/segment/yolov8m-seg/weights/best.pt'
brain_tumor_test_path = "/home/bilal/Desktop/tumor_detection_with_medical_images/datasets/brain_dataset_for_YOLO/test/images/"

Prediction(prediction_area="BRAIN", model_path=brain_tumor_yolo_model_path, test_path=brain_tumor_test_path)

# breast tumor prediction
breast_tumor_yolo_model_path = '/home/bilal/Desktop/tumor_detection_with_medical_images/runs/segment/yolov8l-seg/weights/best.pt'
breast_tumor_test_path = "/home/bilal/Desktop/tumor_detection_with_medical_images/datasets/breast_dataset_for_YOLO/test/images/"

Prediction(prediction_area="BREAST", model_path=brain_tumor_yolo_model_path, test_path=brain_tumor_test_path)


