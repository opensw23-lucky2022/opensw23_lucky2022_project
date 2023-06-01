<p align='center'>
  <img src='https://github.com/prasunroy/openpose-pytorch/raw/master/assets/image_1.jpg' />
</p>

# OpenPose PyTorch
**OpenPose api wrapper in PyTorch.**
Whole-body (Body, Foot, Face, and Hands) 2D Pose Estimation

OpenPose는 단일 이미지에서 인체, 손, 얼굴 및 발 키포인트(총 135개 키포인트)를 공동으로 감지하는 다중 사람 시스템을 나타냅니다.

<br/><br/><br/>

## 목차
* [Requirements](#requirements)
* [Installation](#installation)
* [Results](#results)
* [Analysis/Visualization](#analysis-visualization)
* [Team Introduction](#team-introduction)

---

## Requirements
* Python 3.9.12

module version
* torch 1.12.1+cpu
* cv2 4.7.0
* 


## Installation
#### 사전 준비
Torch 를 깔아줍니다
```
pip3 install torch torchvision torchaudio
```


#### Option 1: pip을 통해 설치
```
pip install git+https://github.com/prasunroy/openpose-pytorch.git
```
#### Option 2: source를 통해 설치
```
git clone https://github.com/prasunroy/openpose-pytorch.git
cd openpose-pytorch
python setup.py install
```

## 실행 examples
프로젝트 root에서 아래 command를 차례대로 실행합니다.
```
cd examples
python image_demo.py //사진 example 생성
python video_demo.py //비디오 example 생성
```

## Results
```
python image_demo.py
```
위 코드 실행시<br>

![image](https://github.com/Team-Lucky2022/openpose_pytorch/assets/74056843/9b2fe93d-de7c-44c1-bdaf-2bab69665a1d)

```
python video_demo.py
``` 
위 코드 실행시<br>
![image](https://github.com/Team-Lucky2022/openpose_pytorch/assets/74056843/698e3c58-b113-4760-9842-a3dd04850c63)

위와 같은 결과가 나오면 examples 실행 성공입니다.

---

## Analysis Visualization
* 

## References
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* [PyTorch OpenPose](https://github.com/Hzzone/pytorch-openpose)
* https://github.com/Team-Lucky2022/openpose_pytorch

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

