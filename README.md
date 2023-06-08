<p align='center'>
  <img src='https://github.com/prasunroy/openpose-pytorch/raw/master/assets/image_1.jpg' />
</p>


# OpenPose PyTorch
## 목차

* [Topic Introduction](#topic-introduction)
* [Requirements](#requirements)
* [Installation](#installation)
* [Analysis](#analysis)
* [Visualization](#visualization)
* [Results/Results Analysis](#results-results-analysis)
* [Team Introduction](#team-introduction)

---


## Topic Introduction 
### OpenPose PyTorch
![스크린샷 2023-06-08 105524](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/e8a7a62c-5eab-44b6-9263-c63f49ad5422)

**<OpenPose api wrapper in PyTorch. Whole-body (Head, body, left-top, right-top) 2D Pose Estimation>**

OpenPose는 이미지, 동영상에서 몸체, 머리, 왼쪽 상반신, 오른쪽 상반신으로 나누어 각각 6, 4, 4, 4개의 키포인트를 감지하여 연결해 다중으로 사람의 움직임을 감지한다.

### OpenPose System

![스크린샷 2023-06-08 110107](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/2896f944-e56a-46c5-bdec-d93fec9cfe5d)

input으로 image가 들어오면 VGG-19 network와 같은 convolution network를 통해 feature map(=F)를 얻는다. F를 두 branch의 input으로 사용하는데, 각 branch는 part confidence map, part affinity fields 를 예측한다. 각 branch에서 얻은 결과와 stage1의 input으로 사용한 feature map을 concat해서 다음 stage의 input으로 사용한다.

-> confidence map 추정시, 이전 stage 의 affinity field 정보를 활용해 현재 stage 의 confidence map 을 추정한다. affinity field 추정시에도 confidence map 정보를 활용한다.

-> 이러한 과정으로 얻은 confidence map과 affinity field 를 통해 key point들을 얻고, key points끼리의 connection을 완성한다.

#### Confidence Map

![스크린샷 2023-06-08 110353](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/2d5e2267-52bc-41f1-8a97-a805dd9d23b9)
<br/><br/><br/>
![스크린샷 2023-06-08 110647](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/9a7a9b58-8f28-4361-b364-d995ea754cdf)

confidence map이란, 이미지를 보고 각 사람의 몸에서 어디에 관절이 있을지를 찾아내는 역할을 하는 것이다.  이 때, 위의 사진에서 branch1(confidence map)을 거쳐 나오면 사람의 관절이 있을것이라 예측되는 곳이 heatmap으로 표현되어 나오게 된다. 위 사진은 왼쪽에서 오른쪽으로 갈 수록 stage가 높아지는데 오른쪽 손목의 관절 위치를 예측한 결과물이다. 그 결과물은 위의 사진처럼 heatmap으로 표현되는데 오른쪽 사진으로 갈 수록 오른손목이라고 예측되는 곳의 heatmap의 선명도가 높아지는 것을 확인 할 수 있다. 이처럼 stage가 진행됨에 따라 confidence map과 part affinity fields를 정확하게 예측한다.

#### Affinity Fields

![스크린샷 2023-06-08 110716](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/4ceb28df-f414-4f11-9872-e1ef4c1cd481)
affinity fields는 2차원 벡터공간이다. 이 벡터공간은 (위치, 방향)을 원소로 하고 있어 2차원 벡터공간이 되는데, 각 관절과 관절 사이에 사람 몸 위에서 방향관계를 표시함으로써 현재 관절 다음에 어느 방향에 있는 관절이 해당 사람 몸에 있는 다음 관절이 맞을 지를 예측하는데 도움을 준다.



---

## Requirements
* Python 3.9.12

requirements.txt
```
opencv-python==4.7.0.72
openpose-pytorch==0.1.0
torch==1.12.1
torchvision==0.13.1
Pillow==9.4.0
imageio==2.30.0
numpy
scipy
tqdm
datetime
time
```



## Installation
#### 사전 준비
Torch 를 깔아준다.
```
pip3 install torch torchvision torchaudio
```
 ```
 pip install -r requirements.txt
 ```


##### Option 1: pip을 통해 설치
```
pip install git+https://github.com/prasunroy/openpose-pytorch.git
```
##### Option 2: source를 통해 설치
```
git clone https://github.com/prasunroy/openpose-pytorch.git
cd openpose-pytorch
python setup.py install
```


## 실행 examples
프로젝트 root에서 아래 command를 차례대로 실한다.
```
cd examples
python image.py --image_filename example.jpg --save_option Y //사진 example 생성
python video.py --video_filename example.mp4 --save_option Y //비디오 example 생성
```
OR
```
cd examples
python3 image.py --image_filename example.jpg --save_option Y //사진 example 생성
python3 video.py --video_filename example.mp4 --save_option Y //비디오 example 생성
```
python이나 python3 둘다 해도 실행가능
## Results
```
python image.py --image_filename example.jpg --save_option Y
```
위 코드 실행시<br>

![image](https://github.com/Team-Lucky2022/openpose_pytorch/assets/74056843/9b2fe93d-de7c-44c1-bdaf-2bab69665a1d)

```
python video.py --video_filename example.mp4 --save_option Y
``` 
위 코드 실행시<br>
![image](https://github.com/Team-Lucky2022/openpose_pytorch/assets/74056843/698e3c58-b113-4760-9842-a3dd04850c63)

위와 같은 결과가 나오면 examples 실행 성공

---

## Analysis(openpose 코드 분석)
1. openpose 파일: openposw 프로젝트의 주 실행 파일, 사용자 인터페이스와 인자 처리 담당, 사용자가 입력한 설정에 따라 실행 
1) body/estimator.py
신체 키포인트를 추정하는 핵심 기능.
**_pad_image**: 입력 이미지 패딩, 비율 유지, 스트라이드, 패드 밸류 고려
**_get_keypoint**: 후보 키포인트, 서브셋 구성으로 최종 키포인트 좌표 생성

**class BodyPoseEstimator(object)**:
OpenPose 딥 러닝 아키텍처를 사용하여 이미지에서 인체 자세를 추정,
모델 초기화, 이미지 전처리, 딥러닝 아키텍처에 이미지 전달하는 클래스, 모델 출력은 키포인트 위치와 히트맵, PAFs로 구성
평균화된 히트맵에서 각 키포인트의 피크를 추출, 임계값을 적용한 후 최종 키포인트를 구성 -> PAF를 사용하여 감지된 키포인트 간의 연결 형성 -> 감지된 키포인트 연결을 서브셋 배열로 구성하여 입력 이미지에서 발견된 가능한 인간 자세 나타냄                                                                        
**__init__**: 토치 모듈 상속, 사전 학습된 모듈을 사용할 수 있도록 함
**estimate**: 이미지 전처리, 스케일로 크기를 조정하고 패딩을 적용
**_pad_image**: 이미지를 전처리하고 패딩을 추가하는 데 사용, 결과 이미지는 모델을 통과할 때 잡음이 줄어든 것으로 인식
**body_part_heatmaps, pafs = self(*preprocessed_images)**: 전처리된 이미지가 모델에 전달된 후 중요한 결과인 신체 부위의 히트맵과 PAF(경로 밀도 필드) 반환
**body_part_candidates_list,body_parts_subset_list=self._get_keypoints(heatmaps=body_part_heatmaps, pafs=pafs)**: 신체부위 히트맵과 PAF를 사용하여 실제 신체부위   키포인트 연결을 산출
**return body_parts_subset_list, body_part_candidates_list**: 클래스의 출력 값, 좌표로 구성된 키포인트 배열, 이 배열을 이용하여 감지된 인간 자세를 시각화하거나 분석가능
 
2) body/model.py
pose estimation(관절 각 추정)을 위한 모델을 정의.
바른 구조(coco, mpi, body_25)에 따라 키 포린트의 개수와 연결 관계 정의
모델에 대한 정보 저장, 텐서를 생성하여 body/estimator.py에 전달
3) utils.py
프로젝트 내에서 공통으로 사용되는 배열조작, FPS 계산 등의 기능을 제공
draw_keypoints, draw_body_connections: 이미지에 키 포인트와 관절 연결을 그림
draw_keypoints: 원본 이미지를 복사하여 새로운 이미지(overlay)생성
for kp inkeypoints: for x,y,v in kp: x,y는 좌표이며 v는 표시 여부, 키포인트를 반복하면서 표시 여부가 True인 곳에 cv2의 circle을 이용하여 원을 그린 후 addWeighred를 사용하여 overlay와 원본이미지를 결합하여 return
draw_body_connections: 원본 이미지 복사하여 새로운 이미지(overlay)생성.
b_conn: 몸체, h_conn: 머리, l_conn: 왼쪽 상반신, r_conn: 오른쪽 상반신 으로 나누어 연결선 간에 어떤 키포인트 연결해야 하는지 정의
_draw_connection으로 각 정의된 부분 연결
addWeighted로 원본 이미지와 결합 후 반환
draw_face_connections, draw_hand_connections: 아직 구현 안됐으므로 에러 발생
_draw_connection 함수 (helper 함수): x1, y1, v1 = point1 / x2, y2 v2 = point2: 시작점과 끝점의 x, y 좌표 및 표시 여부를 각각 저장
if v1 and v2: 시작점과 끝점의 표시 여부가 모두 True일 때만 선을 그림.
cv2의 line을 이용하여 이미지에 시작점에서 끝점까지 선을 그림
그려진 이미지 return 

2. 실행파일 (demo.py 분석)
1) 필요 라이브러리, 모듈 import:  cv2로 이미지 처리, estimator.py의 BodyPoseEstimator 클래스를 불러와 신체 키 포인트 추정에 사용, utils.py의 draw_keypoints/draw_body_connections를 불러와 이미지 위에 키포인트와 관절 연결을 그림
2) BodyPoseEstimator 객체 생성: 키포인트 추정에 필요한 BodyPoseEstimator 객체를 생성. 이때 사전 훈련된 가중치를 사용하도록 설정(pretrained=True)
3) 이미지 불러오기: cv2의 imread를 사용하여 입력 이미지를 불러와 image_src에 저장
4) 키포인트 추정: BodyPoseEstimator 객체를 사용하여 이미지에서 키포인트를 찾고, 결과를 keypoints에 저장
5) 관절 연결 그리기: draw_body_connections 함수 사용, 파라미터로 두께 4, 투명도 0.7로 주어 이미지 위에 관절 연결을 그림.
6) 키포인트 그리기: draw_keypoints 함수 사용, 파라미터로 반지름 5, 투명도 0.8로 주어 키포인트를 이미지 위에 그림.
7) 결과 이미지 표시와 저장: cv2의 imshow, waitkey를 통해 생성된 이미지를 화면에 보여줌.


---
## Visualization
코드 추가 부분(visualizations)

1. 터미널에서 파일명 보이기
수정 부분: requirement.txt 추가, examples에 input, output 파일 추가, 터미널창 –image_filename추가
코드 설명: 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_filename', required=True, help='File name of video')
    parser.add_argument('--save_option', required=True, help='Whether to save output as JPG(Answer in Y/N)')

    args = parser.parse_args()

    human_pose(args.image_filename, args.save_option)
    
import argparse를 통해 argparse를 불러와 parser의 add_addargument를 사용하여 –image_filename, –save_option 설정을 추가한다. 

2. output 사진 폴더에 저장
수정 부분: requirement.txt 추가, examples에 input, output 파일 추가, 터미널창 –save_option 추가
코드 설명:
if(save_option == 'Y'):
        cv2.imwrite("./output/" + image_filename.split('.')[0]+"_output.png",image_dst)

 원래의 repository 코드는 단순히 시각화 된 이미지를 보여주기만 하고 저장하지 않는다. 따라서 시각화 된 이미지가 필요에 따라 저장되도록 하기 위해서 save_option을 추가하고, 해당 옵션이 설정되어 있으면 cv2의 imwrite를 이용하여 output파일 경로에 해당 결과물을 저장하도록 변경하였다.


---
## Results/Results Analysis

### Results

### image.py
사진 속 인원 수에 따른 프로그램 런타임의 차이를 분석해보기 위해 다음과 같이 시행했다.

* input되는 사진들의 크기가 모두 달라,사진의  크기가 런타임에 주는 영향을 줄이기 위해 사진의 크기가 일정 크기를 넘으면 사진을 축소하는 코드를 추가해 실행
* 각 사진 별로 10회씩 코드를 실행
* 각 회차 별로 런타임을 기록
* 인원 수에 따른 평균 런타임을 구함
* 사진 속 사람의 수를  x축, 평균 런타임을 y축으로 하는 점 그래프를 그리고, 보간법을 통해 점들의 경향을 관찰.(여기서는 4차 다항식 보간법 사용)
*  결과분석

#### <실행결과_image>
![스크린샷 2023-06-07 183230](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/cc5573b1-43ee-4358-a904-4735b7bea3fc)
![스크린샷 2023-06-07 183307](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/e74092f4-e41d-4989-8271-f767a2beb25d)
![스크린샷 2023-06-07 183507](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/4244410d-3f82-442c-b4a9-92c522c6a644)
![스크린샷 2023-06-07 183531](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/8f05f77f-899c-4b12-af48-b9488abd1b03)
![people_about_37_output](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/4cc56c11-a8b9-4c6e-b9cb-5441026ded7b)
(37명)

#### <실행결과>
![스크린샷 2023-06-07 171511](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/7b7cc23e-7144-4eb9-a25e-c3a1bcac2b85)
![스크린샷 2023-06-07 164451](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/8d6c47c6-7b59-4056-9449-3dd2979a8ee9)

(노란색: 사진 속 인물들의 수,  선홍색: 시행횟수,  주황색: 평균 프로그램 런타임)

#### <실행결과 분석>
이 그래프의 x축은 사진 속 인물들의 수, y축은 x값에 따른 프로그램의 런타임을 나타낸다. 파란색 그래프는 점들의 값을 4차 다항식으로 보간한 그래프이다.

그래프를 보면, 사진 속 인원 수가 증가함에 따라 대체적으로 프로그램 런타임이 길어지는 것을 알 수 있다. 특히 1명에서부터 14명까지 인원수가 증가할수록 런타임도 급격하게 증가한다. 그러나 중간에 경향을 따르지 않는 점들도 존재하는데, 이는 사진 속 인물들의 형태가 정확하지 않거나 인원이 많아질수록 다른 인물들과 겹쳐져 있어, 프로그램이 인물들을 잘 인지하지 못해 발생하는 문제로 보인다.

 이를 통해 사진 속 인물들의 수가 증가할수록 프로그램의 런타임이 대체적으로 증가한다는 것을 알 수 있다. 
 
 ### video.py
 동영상 속의 인물들의 수에 따른 프로그램 런타임의 차이를 알아보기 위해 다음과 같이 시행했다.
* 모든 동영상을 4초로 준비
* 동영상은 사진보다 용량이 크기 때문에 실행횟수를 3번으로 함
* 각 실행 회차 별 프로그램 런타임 기록
* 동영상 속 인물들의 수에 따른 평균 프로그램 런타임 기록
* x축을 인물들의 수, y축을 평균 프로그램 런타임으로 하는 점 그래프를 그리고, 이를 보간하여 점들의 추이를 분석
* 결과분석

#### <실행결과>
![스크린샷 2023-06-07 205302](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/cf29f449-d544-453b-b702-5b13b998b912)

(노란색: 동영상 속 인물들의 수,  초록록색: 시행횟수,  주황색: 평균 프로그램 런타임)

![스크린샷 2023-06-07 210040](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/f4a6a1a8-deff-4bd4-abd8-f566265d2c55)
![스크린샷 2023-06-07 212033](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/710aaafd-5bbd-4612-b908-ad68c9b64177)

#### <실행결과 분석> 
 첫번째 그래프의 x축은 동영상 속 인물들의 수, y축은 x값에 따른 프로그램의 런타임을 나타낸다. 파란색 그래프는 점들의 값을 스플라인 보간법으로  보간한 그래프이다.
이 그래프를 보면 동영상 속 인물들의 수의 변화와 관계없이 프로그램 런타임이 측정되었다. 이는 동영상의 시간을 통일 할 때 0.n초 차이로 인해 동영상들의 프레임 수와 용량의 크기 차이가 생겨 그런 것으로 보인다.  동영상 속 인물들의 수는 프로그램 런타임의 크기에 큰 영향을 미치지는 않는 것으로 보인다.

두번째 그래프의 x축은 동영상의 용량(MB), y축은 x값에 따른 프로그램의 런타임을 나타낸다.
보라색 그래프는 점들의 값을 2차 다항식으로 보간한 그래프이다.

이 그래프를 보면 동영상의 용량이 증가하면 어느정도는 프로그램의 런타임이 증가하는 것처럼 보인다. 그러나 뚜렷한 경향이 보이지 않고 용량이 증가해도 프로그램의 런타임이 증가하지 않는 점들도 많기에 용량 자체로는 프로그램의 런타임의 크기에 큰 영향을 미치지 않는 것으로 보인다.

두 그래프를 통해 유의미한 결과를 얻지 못했는데 이는 동영상의 용량을 통일하지 못했기 때문으로 추정한다. 그러나 동영상의 인물들의 수보다는, 용량의 크기가 프로그램의 런타임의 크기에 더 영향을 미치는 것을 알 수 있다.



---
## References
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* [PyTorch OpenPose](https://github.com/Hzzone/pytorch-openpose)
* https://github.com/prasunroy/openpose-pytorch

## License
* [OpenPose License](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)
* [OpenPose PyTorch License](https://github.com/prasunroy/openpose-pytorch/blob/master/LICENSE)

---

## Team Introduction
* 길윤빈 202211259 : 팀장(코드 작성 및 실행 데이터 수집/ 정리) 
* 박세준 202011295 : 팀원(전체 피드백 및 발표 자료 작성/ 발표)
* 장지은 202211363 : 팀원(코드 분석 및 보고서 작성)
* 정소현 202211368 : 팀원(결과 분석 및 보고서 작성)


<br />
<br />
