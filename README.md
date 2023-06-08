<p align='center'>
  <img src='https://github.com/prasunroy/openpose-pytorch/raw/master/assets/image_1.jpg' />
</p>


# OpenPose PyTorch
## 목차

* [Topic Introduction](#topic-introduction)
* [Requirements](#requirements)
* [Installation](#installation)
* [Analysis](#analysis)
* [Results/Results Analysis](#results-results-analysis)
* [Visualization](#visualization)
* [Team Introduction](#team-introduction)

---


## Topic Introduction 
### OpenPose PyTorch
![스크린샷 2023-06-08 105524](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/e8a7a62c-5eab-44b6-9263-c63f49ad5422)

**<OpenPose api wrapper in PyTorch. Whole-body (Head, body, left-top, right-top) 2D Pose Estimation>**

OpenPose는 이미지, 동영상에서 몸체, 머리, 왼쪽 상반신, 오른쪽 상반신으로 나누어 각각 6, 4, 4, 4개의 키포인트를 감지하여 연결해 다중으로 사람의 움직임을 감지한다.

### OpenPose System

![스크린샷 2023-06-08 230303](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/50d4e04b-a79c-4fc1-ba84-4523a802f9a9)
![스크린샷 2023-06-08 230316](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/a3ad00cd-7097-4a00-8491-67c849e649e8)
![스크린샷 2023-06-08 230329](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/9e8b5b4d-cb78-4c27-8474-45e96aaceccc)
![스크린샷 2023-06-08 230344](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/b9435261-ff49-4fe9-b585-f6dcc73193d7)


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


## Analysis(openpose 코드 분석)

주요 파일들

  |
  
  |-- main 실행 파일
  
  |     |-- 사용자 인터페이스 관리
  
  |     |-- 인자 처리 및 실행 설정
  
  |
  
  | -- body/estimator.py
  
  |     |-- 신체 키포트 추정 기능
  
  |
  
  |-- body/model.py
  
  |     |-- 관절 각 추정 모델 정의
  
  |
  
  |-- utils.py
 
  |     |-- 공통 기능 제공 
  
  |     | |-- 배열 조작
  
  |     | |-- FPS 조정    
        
                
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
![스크린샷 2023-06-08 224909](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/f98a86fc-7e45-454f-9995-cb180aef862f)
![스크린샷 2023-06-08 224916](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/c17c671e-a143-43eb-89a4-1dab9657639c)
![스크린샷 2023-06-08 224926](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/6283891b-600f-42a9-8528-f63c63e909bb)
![스크린샷 2023-06-08 225601](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/28a27f19-9e62-4deb-8cfa-6b760fcff176)

(31명)

#### <실행결과>
![스크린샷 2023-06-08 222443](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/2f4af0a1-f8f2-40f3-8ede-fae10c48d6a5)

(노란색: 사진 속 인물들의 수,  선홍색: 시행횟수,  주황색: 평균 프로그램 런타임)

![스크린샷 2023-06-08 223532](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/458e6383-2f7f-40ef-92e1-6ba9cb29e91c)


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
![스크린샷 2023-06-08 224606](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/69010b9e-caba-4838-8e15-3cb24c49afe3)

(노란색: 동영상 속 인물들의 수,  초록록색: 시행횟수,  주황색: 평균 프로그램 런타임)

![스크린샷 2023-06-08 224251](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/481f160a-7eac-48ba-a08a-d0404f0d4016)


![스크린샷 2023-06-07 212033](https://github.com/opensw23-lucky2022/opensw23_lucky2022_project/assets/127183125/710aaafd-5bbd-4612-b908-ad68c9b64177)

#### <실행결과 분석> 
 첫번째 그래프의 x축은 동영상 속 인물들의 수, y축은 x값에 따른 프로그램의 런타임을 나타낸다. 파란색 그래프는 점들의 값을 스플라인 보간법으로  보간한 그래프이다.
이 그래프를 보면 동영상 속 인물들의 수의 변화와 관계없이 프로그램 런타임이 측정되었다. 이는 동영상의 시간을 통일 할 때 0.n초 차이로 인해 동영상들의 프레임 수와 용량의 크기 차이가 생겨 그런 것으로 보인다.  동영상 속 인물들의 수는 프로그램 런타임의 크기에 큰 영향을 미치지는 않는 것으로 보인다.

두번째 그래프의 x축은 동영상의 용량(MB), y축은 x값에 따른 프로그램의 런타임을 나타낸다.
보라색 그래프는 점들의 값을 2차 다항식으로 보간한 그래프이다.

이 그래프를 보면 동영상의 용량이 증가하면 어느정도는 프로그램의 런타임이 증가하는 것처럼 보인다. 그러나 뚜렷한 경향이 보이지 않고 용량이 증가해도 프로그램의 런타임이 증가하지 않는 점들도 많기에 용량 자체로는 프로그램의 런타임의 크기에 큰 영향을 미치지 않는 것으로 보인다.

두 그래프를 통해 유의미한 결과를 얻지 못했는데 이는 동영상의 용량을 통일하지 못했기 때문으로 추정한다. 그러나 동영상의 인물들의 수보다는, 용량의 크기가 프로그램의 런타임의 크기에 더 영향을 미치는 것을 알 수 있다.


---
## Visualization (코드 추가 부분)
### (1) 터미널에서 파일명 보이기
* 수정 부분: requirement.txt 추가, examples에 input, output 파일 추가, 터미널창 –image_filename추가. 
* 코드 설명: import argparse를 통해 argparse를 불러와 parser의 add_addargument를 사용하여 –image_filename, –save_option 설정을 추가한다. 

```
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_filename', required=True, help='File name of video')
    parser.add_argument('--save_option', required=True, help='Whether to save output as JPG(Answer in Y/N)')

    args = parser.parse_args()

    human_pose(args.image_filename, args.save_option)
```
### (2)output 사진 폴더에 저장
* 수정 부분: requirement.txt 추가, examples에 input, output 파일 추가, 터미널창 –save_option 추가
* 코드 설명:  원래의 repository 코드는 단순히 시각화 된 이미지를 보여주기만 하고 저장하지 않는다. 따라서 시각화 된 이미지가 필요에 따라 저장되도록 하기 위해서 save_option을 추가하고, 해당 옵션이 설정되어 있으면 cv2의 imwrite를 이용하여 output파일 경로에 해당 결과물을 저장하도록 변경하였다.

```
if(save_option == 'Y'):
        cv2.imwrite("./output/" + image_filename.split('.')[0]+"_output.png",image_dst)
```
### (3) 이미지 파일 크기 조정
* 코드 설명: 원래의 repository 코드는 입력받은 이미지 크기를 조정하지 않고 사용하기때문에 이미지가 크기가 너무 큰 경우 output결과 팝업창에서 사진이 잘려보이는 경우가 발생하였다. 이를 방지하기 위해 cv2의 resize를 이용하여 사진크기에 제한을 두도록 변경하였다.

```
MAX_WIDTH = 1000
MAX_HEIGHT = 1000
image_src = cv2.imread(image_dir)
height, width, _ = image_src.shape

    if width > MAX_WIDTH or height > MAX_HEIGHT:
        scale = min(MAX_WIDTH/width, MAX_HEIGHT/height)
        resized_width = int(width*scale)
        resized_height = int(height*scale)
        resized_image = cv2.resize(image_src, (resized_width, resized_height))
    else:
        resized_image = image_src
```

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
* 장지은 202211363 : 팀원(결과 분석 및 보고서 작성)
* 정소현 202211368 : 팀원( 분석 및 보고서 작성)


<br />
<br />
