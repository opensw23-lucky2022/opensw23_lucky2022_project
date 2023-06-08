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
Torch 를 깔아줍니다
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
프로젝트 root에서 아래 command를 차례대로 실행합니다.
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
python이나 python3 둘다 해도 실행가능합니다
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

위와 같은 결과가 나오면 examples 실행 성공입니다.

---

## Analysis Visualization

## References
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* [PyTorch OpenPose](https://github.com/Hzzone/pytorch-openpose)
* https://github.com/prasunroy/openpose-pytorch

## License
* [OpenPose License](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)
* [OpenPose PyTorch License](https://github.com/prasunroy/openpose-pytorch/blob/master/LICENSE)

---

## Team Introduction
* 길윤빈 202211259 : 팀장 
* 박세준 202011295 : 팀원 
* 장지은 202211363 : 팀원
* 정소현 202211368 : 팀원


<br />
<br />
