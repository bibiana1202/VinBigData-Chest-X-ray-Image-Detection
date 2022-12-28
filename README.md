


# VinBigData-Chest-X-ray-Image-Detection

- VinBigData Chest X-ray Image Detection : <a href="https://www.notion.so/wew1202/VinBigData-Chest-X-ray-Detection-5c03f0811f5a47adb314f918795a2056">![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)

- Kaggle - VinBigData chest X-ray abnormalities detection contest : <a href="https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/overview">![kaggle](https://img.shields.io/badge/-kaggle-blue)  
    
---
## ☺️ Introduction
* ### Why Chest X-ray?

  - Chest X-ray는 기본중에 기본인 검사.
  - 생명과 직결되는 부위이기 때문에 정확한 진단이 필요하다.
  - 다른 부위의 X-ray와는 다르게 놓칠 가능성이 있는 부위라 많고 많은 X-ray data중 chest X-ray를 선택하였다.
* ### Why Chest X-ray needs AI?

  - 예를 들어 종양 판독의 경우 종양이 크다면 누구나 판독이 가능하다. 하지만 만약 결절의 size가 작다면 놓칠 수 있는 가능성이 있다. 이렇게 병변을 놓치지 않기위해 AI가 의사의 도움이 되어 진단에 도움이 될 수 있다.
또한 일반 X-ray 전문의사(방사선과)가 아니더라도 판독에 도움을 받을 수 있기때문에 AI를 이용한다면 좀 더 정확하게 판독이 가능할 것이다.

* ### purpose
  - 폐와 관련된 14가지의 질병을 detecting하여 data augmentation에 따른 여러 model의 performance 비교
---
## 🫁 Materials & Methods
* ### Materials
  
  - Vietnam hospitals dataset (the Hospital 108 and the Hanoi Medical University Hospital)
  - train images: 15,000 (normal: 10,606, patient: 4,394)
  - test images: 3,000
  - bbox info: image_id, class_id, x_min, y_min, x_max, y_max
  - image resize: 512 x 512 , 1024 x 1024
  - 병명 사전조사 : <a href="https://www.notion.so/wew1202/8204385788fd45c1adeb7c0c7dc5e4db">![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)
  
* ### Methods
  - Tools: OpenCV, PyTorch, numpy, pandas, sklearn, seaborn, matplotlib
  - Augmentations: Rotation(random), Flip(horizontal), Zoomin(10%), Cutmix, CLAHE, Equlization
  - Models: Faster RCNN, YOLOv5, RetinaNet, Yolof, Yolox, CenterNet
  - Workflow : 
  
  
  
---
## ☺️ Results
* ### EDA
![https://user-images.githubusercontent.com/61971952/193209874-ebc78a59-5b58-4816-8412-c841a3b6099f.png](https://user-images.githubusercontent.com/61971952/193209874-ebc78a59-5b58-4816-8412-c841a3b6099f.png)
    
    - 정상인의 데이터를 삭제하고 적은 양의 환자 데이터만 남음
    - 라벨 간의 극단적인 양 차이 -> 데이터 불균형
    - 단일 이미지안에 다중 라벨
    
* ### Augmentation에 따른 dataset 종류

    - category A: no augmentation (15000장)
    - category B: rotation, flip, zoomin (15000장 + 6250장)
    - categroy C: rotation, flip, zoomin(10%), CLAHE, equalization (15000장 + 6250장)


* ### Model에 따른 성능 비교(kaggle score)

M/D	| 512 A | 512 B | 512 C | 1024A | 1024B | 1024C |
--------------|-------|-------|-------|-------|-------|-------|
Faster R-CNN | 0.129 | 0.135 | --- | 0.131 | 0.137 | 0.123 |
Yolov5 | 0.179 | 0.114 | 0.119 | 0.165 | 0.123 | 0.134 |
RetinaNet | 0.041 | --- | --- | 0.135 | 0.126 | 0.142 |
Yolof | 0.039 | --- | --- | 0.142 | 0.135 | 0.126 |
Yolox | 0.145 | 0.108 | --- | 0.141 | 0.127 | 0.179 |
CenterNet | 0.038 | --- | --- | 0.069 | --- | 0.083 |

* ### 앙상블 성능 비교(kaggle score)
  - 2-pred score
    
n | WBF | Faster R-CNN | Yolov5 | RetinaNet | Yolof | Yolox | SCORE |
--------------|-------|-------|-------|-------|-------|-------|-------|
#1 | Faster,Retina | 1024A(0.131/0.110) | - | 1024A(0.135/0.114) | - | - | 0.179/0.151 |
#2 | Faster,yolov5 | 1024A(0.131/0.110) | 512A KFLOD(0.179/0.149) | - | - | - | 0.210/0.196 |
#3 | Retina,yolov5 | - | 512A KFLOD(0.179/0.149) | 1024A(0.135/0.114) | - | - | 0.212/0.204 |
#4 | Faster,yolov5,Retina | 1024A(0.131/0.110) | 512A KFLOD (0.179/0.149) | 1024A (0.135/0.114) | - | - | 0.216/0.209 |
#5 | yolov5,yolovf | - | 512A KFLOD (0.179/0.149) | - | 1024A (0.142/0.106) | - | 0.218/0.205 |
#6 | Faster,yolov5,Retina,yolovf| 1024A (0.131/0.110) | 512A KFLOD (0.179/0.149) | 1024A (0.135/0.114) | 1024A  (0.142/0.106) | - | 0.229/0.206 |
#7 | Faster,Retina,yolof,yolox| 1024A (0.131/0.110) | - | 1024A (0.135/0.114),1024C(0.142/0.121) | 1024A (0.142/0.106),1024B(0.135/0.122) | 1024C(0.179/0.156) | 0.215/0.165 |
#8 | Faster,Retina,yolof,yolox| 1024A (0.131/0.110) | - | 1024A (0.135/0.114),1024C(0.142/0.121) | 1024A (0.142/0.106),1024B (0.135/0.122) | 1024A(0.141/0.118),1024C(0.179/0.156) | 0.222/0.170 |    

    
* ### 예측 이미지

---
## ☺️ Discussion
 - Medical trend인 object detection 을 공부하기 위하여 선택한 프로젝트    

* ### Model selection
    - 1 stage model
        - YOLOv5, YoloX, YoloX: 1 stage에서 유명하고 속도가 빠르기때문에 사용함.
        - EfficientDet (one-stage detector paradigm 기반으로 구성됨): 사람들이 주로 사용하는 YOLO v5보다 average precision이 좋기때문에 선택.
    - 2 stage model
        -  Faster R-CNN: 이전 수업에서 사용했던 model이 1 stage라서 2 stage 공부 겸 여전히 현역으로 쓰이고 있는 기초적인 모델이라서 선택하였음.

* ### Augumentation
    ![image](https://user-images.githubusercontent.com/105691874/207202832-2d228bd9-3971-4314-95af-e90ca88904e2.png)
    - 학습하는 data 양이 15,000장인줄 알았지만 data를 분석한 결과 정상인을 제외한 환자의 data는 4,394장이었다.
    - 질병을 학습해야하는 model이기때문에 환자의 data만 갖고 학습을 시켜야하는데 data의 양이 너무 적어 양을 늘리기 위하여 여러 augmentation을 적용해보았다.
    - Augmentation에 따른 성능 평가를 비교해 보기위해 augmentation을 안한 A그룹과 기본적인 augmentation을 한 B그룹, 마지막으로 기본적인 augmentation외 여러 다양한 기법까지 적용한 C그룹으로 나누었다.
그 결과 3개의 model 모두 augmentation을 하면 할수록 성능이 향상됨을 확인하였다.
    - Data내에서 모든 label이 비슷한 양으로 존재하지않고 특정 label 위주로 존재하고있다. 즉, data imbalance가 심한상황.
    - Data imbalance 문제를 해결하기 위해 적은 양을 갖는 label(0,3,9,10,11,13)엔 여러가지 augmentation으로 up sampling하는 작업을 하였다.
    - 하지만 기본 augmentation외에 이미지 크기가 1024 인 경우가 성능이 대체로 더 잘 나왔다.
    - imbalance와 data의 양을 동시에 해결한다면 이보다 훨씬 더 좋은 성능을 낼 수 있지 않을까 싶다.

---
## Ref
- https://www.kaggle.com/code/dschettler8845/visual-in-depth-eda-vinbigdata-competition-data
- https://www.kaggle.com/code/yerramvarun/pytorch-fasterrcnn-with-group-kfold-14-class
- https://www.kaggle.com/code/pestipeti/vinbigdata-fasterrcnn-pytorch-inference/notebook
- https://www.kaggle.com/code/pestipeti/vinbigdata-fasterrcnn-pytorch-train/notebook

