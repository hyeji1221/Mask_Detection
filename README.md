# Mask_Detection

### 설명

코로나19로 인한 마스크 착용으로 webcam을 통해 마스크 착용 유무를 알려줍니다.

### 팀원

|                            팀원                             |               역할               |
| :---------------------------------------------------------: | :------------------------------: |
|      [hyeji1221(임혜지)](https://github.com/hyeji1221)      |    프로젝트 관리(PM), OpenCV     |
| [An-Byeong-Seon(안병선)](https://github.com/An-Byeong-Seon) | 모델 훈련에 필요한 데이터 전처리 |
|       [BaeEunGi(배은기)](https://github.com/BaeEunGi)       |        모델 개발 및 훈련         |
|        [ulimsss(서유림)](https://github.com/ulimsss)        |                                  |
|    [ParkSuBin01(박수빈)](https://github.com/ParkSuBin01)    |                                  |

### 사용 기술

- Python
- Tensorflow
- Keras
- OpenCV

### 실행화면

<img src = "https://user-images.githubusercontent.com/59350891/108947848-a03d0600-76a4-11eb-9e27-eb9c39bfbce7.png" width = 35%>
<img src = "https://user-images.githubusercontent.com/59350891/108947858-a3d08d00-76a4-11eb-9a8c-71ae6644605d.png" width = 35%>

------

#### Model

2021.02.03 전처리 코드 틀 잡기 - 안병선     
2021.02.10 데이터셋 npy 형태로 변환 - 임혜지     
2021.02.10 전처리 코드 완성 - 안병선     
2021.02.13 모델 설계 및 훈련 ver -1.0 #배은기     
2021.02.19 데이터 넘파이 배열화 #배은기     
2021.02.19 모델 설계 및 훈련 ver -2.0 #배은기     
2021.02.21 image processing 수정 후 모델 재훈련 ver 3.0 - 임혜지

#### OpenCV

2021.01.31 얼굴 인식 후 직사각형 그리기, 정확도 출력 - 임혜지    
2021.01.31 webcam 좌우 반전 - 임혜지      
2021.02.17 이미지 전처리, 예측값 출력 - 임혜지
