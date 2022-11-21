# VinBigData-Chest-X-ray-Image-Detection
  - VinBigData Chest X-ray Image Detection : <a href="https://www.notion.so/wew1202/VinBigData-Chest-X-ray-Detection-5c03f0811f5a47adb314f918795a2056">![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)
---

* Kaggle - VinBigData chest X-ray abnormalities detection contest : <a href="https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/overview">![kaggle](https://img.shields.io/badge/-kaggle-blue)  
  
  
---
## 🫁Introduction
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
## 🫁Materials & Methods
* ### Materials
  - Vietnam hospitals dataset (the Hospital 108 and the Hanoi Medical University Hospital)
  - train images: 15,000 (normal: 10,606, patient: 4,394)
  - test images: 3,000
  - bbox info: image_id, class_id, x_min, y_min, x_max, y_max
  - image resize: 512 x 512 , 1024 x 1024
  - 병명 사전조사 : <a href="https://www.notion.so/wew1202/8204385788fd45c1adeb7c0c7dc5e4db">![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)
  
* ### Methods
  - Tools: OpenCV, PyTorch, numpy, pandas, sklearn, seaborn, matplotlib
  - Augmentations: Rotation(90º), Flip(horizontal), Zoomin(10%), Cutmix, CLAHE, Equlization, Mosaic
  - Models: 
  - Workflow
  
  
  
---
## 🫁Results
EDA

정상인의 데이터를 삭제하고 적은 양의 환자 데이터만 남음
라벨 간의 극단적인 양 차이 -> 데이터 불균형
단일 이미지안에 다중 라벨

Augmentation에 따른 dataset 종류

category A: no augmentation (4,394장)
category B: rotation, flip, zoomin (17,576장)
categroy C: rotation, flip, zoomin, cutmix, CLAHE, equalization, mosaic (30,758장)
category D: 데이터 불균형 해소를 위해 가장 적은 양의 라벨을 갖는 사진만 augmentation을 적용하고 나머지 라벨은 down sampling (5,999장)


Category A	Category B	Category C	Category D
원본	ROTATION	ROTATION	ROTATION
 	FLIP	FLIP	FLIP
 	ZOOM IN	ZOOM IN	ZOOM IN
 		CUTMIX	CUTMIX
 		CLAHE	CLAHE
 		EQUALIZATION	EQUALIZATION
 		MOSAIC	MOSAIC
Model에 따른 성능 비교(kaggle score)


Model	Category A	Category B	Category C	Category D
EfficientDet	0.038	0.046	0.052	--
Faster R-CNN	0.012	0.098	0.013	--
YOLOX	0.021	0.068	0.147	0.070
* 예측 이미지

---
## 🫁Discussion
Bio의 trend인 object detection 을 공부하기 위하여 선택한 프로젝트

학습 data안에서 train & valid로 나누지 않고 Group K-Fold를 사용했던 이유?

적은 data set에 대하여 정확도를 향상시킬 수 있다.
∵ training/validation/test 세개의 집단으로 분류하는 것보다 training & test로만 분류시 학습할 data가 더 많게되어 underfitting등 성능이 미달되는 model로 학습되지 않도록 함.
또한 1개의 이미지에 다중 label이므로 예측의 정확도를 확실히 평가하기위해 train set & valid에 포함된 image가 겹치지 않도록 하기위하여 k-fold중에서도 group k-fold를 사용하였다.
하지만 학습 시간이 꽤 오래걸려 시간상 k = 1 로 세팅해 놓았음.
(즉, kfold를 하지 않고 train:valid = 8:2로 데이터셋을 별도로 나눠 학습한 것과 같음.)
10epoch으로 학습시 k=1로만 본것과 k=5로 하여 성능을 비교한 결과 public score가 0.014에서 0.025로 향상됨을 확인할 수 있었다.
그러므로 데이터를 augmenation한 B와 C도 제대로 k = 5로 세팅해서 학습했다면 더 좋은 성능을 보였을 듯 하다.

k	EPOCH	Score	EPOCH	Score
1	10e	0.014	20e	0.016
5	10e	0.025	20e	0.024
Zoom in augmentation시 10%로 한 이유?

10%보다 더 zoom in을 했을경우 이미지의 가장자리에 위치하던 병변들이 잘리는 경우들이 있어서 이를 막기위해 10%정도만 zoom in을 하였다.
normalization 후 다시 size(512x512) 재정의시 정수화함에따라 같은 값을 갖게되는 경우가 있었다.

병변이 너무도 작아 bbox의 y_max와 y_min이 별 차이가 나지 않았기 때문.
이런 데이터로인해 학습시 오류가 발생하여 해당 데이터(10개 미만)는 삭제하기로 함.
Model selection

1 stage model
YOLOX: 1 stage에서 유명하고 속도가 빠르기때문에 사용함.
EfficientDet (one-stage detector paradigm 기반으로 구성됨): 사람들이 주로 사용하는 YOLO v5보다 average precision이 좋기때문에 선택.
2 stage model
Faster R-CNN: 이전 수업에서 사용했던 model이 1 stage라서 2 stage 공부 겸 여전히 현역으로 쓰이고 있는 기초적인 모델이라서 선택하였음.
학습하는 data 양이 15,000장인줄 알았지만 data를 분석한 결과 정상인을 제외한 환자의 data는 4,394장이었다.

질병을 학습해야하는 model이기때문에 환자의 data만 갖고 학습을 시켜야하는데 data의 양이 너무 적어 양을 늘리기 위하여 여러 augmentation을 적용해보았다.
Augmentation에 따른 성능 평가를 비교해 보기위해 augmentation을 안한 A그룹과 기본적인 augmentation을 한 B그룹, 마지막으로 기본적인 augmentation외 여러 다양한 기법까지 적용한 C그룹으로 나누었다.
그 결과 3개의 model 모두 augmentation을 하면 할수록 성능이 향상됨을 확인하였다.
Data내에서 모든 label이 비슷한 양으로 존재하지않고 특정 label위주로 존재하고있다. 즉, data imbalance가 심한상황.

Data imbalance 문제를 해결하기 위해 너무 많은 양을 갖고있는 특정 label(0,3,11,13)은 down sampling하고 적은 양을 갖는 label(1,12)엔 여러가지 augmentation으로 up sampling하는 작업을 하였다.
1,12에는 Rotation, Flip, Zoomin, Cutmix, CLAHE, Equalization 을 적용하고
0,3,11,13은 약 3,000개로 down sampling하여 label간의 극단적인 차이가 어느정도 해소된 D그룹을 만들었다.
가장 성능이 좋았고 학습속도가 빠른 YOLOX로 D그룹을 학습한 결과 A그룹에 비해 성능이 향상됨 을 확인할 수 있었고 양이 적은데도 불구하고 기본 3가지 augmentation을 한 B그룹보다 성능이 좋았다.
하지만 기본 augmentation외에 추가적인 augmentation을 했던 C그룹보다는 성능이 덜 나왔다.
(이는 data의 양이 6배나 차이가 나기때문에 나온 결과)
C그룹에서 훨씬 성능이 좋았던것을 통해 data imbalance를 해결한것보다는 data의 양이 충분히 있는것이 성능향상에 더 많은 효과가 있음을 유추할 수 있었고
imbalance와 data의 양을 동시에 해결한다면 이보다 훨씬 더 좋은 성능을 낼 수 있지 않을까 싶다.

---
## Ref
https://www.kaggle.com/code/dschettler8845/visual-in-depth-eda-vinbigdata-competition-data
https://www.kaggle.com/code/yerramvarun/pytorch-fasterrcnn-with-group-kfold-14-class
https://www.kaggle.com/code/pestipeti/vinbigdata-fasterrcnn-pytorch-inference/notebook
https://www.kaggle.com/code/pestipeti/vinbigdata-fasterrcnn-pytorch-train/notebook

