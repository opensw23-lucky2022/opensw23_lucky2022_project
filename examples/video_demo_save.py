import sys
sys.path.append('../')

import cv2
import imageio #gif
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


estimator = BodyPoseEstimator(pretrained=True)
videoclip = cv2.VideoCapture('input/example.mp4')

# 비디오 속성 얻기
width  = int(videoclip.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoclip.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    =(videoclip.get(cv2.CAP_PROP_FPS))

frames = [] #gif

# 결과를 저장할 비디오 작성기 설정 mp4
#fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # MP4 코덱(*'XVID는 AVI를 사용할 때)
#output_video = cv2.VideoWriter('output/example_output.mp4', fourcc, fps, (width, height)) 


while videoclip.isOpened():
    flag, frame = videoclip.read()
    if not flag:
        break
    keypoints = estimator(frame)
    frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
    frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)

    # 결과 프레임을 output_video에 추가 mp4
    #output_video.write(frame)

    # 프레임을 BGR에서 RGB로 변환하여 리스트에 추가 gif
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    frames.append(frame_rgb)
    
    cv2.imshow('Video Demo', frame)
    if cv2.waitKey(20) & 0xff == 27: # exit if pressed `ESC`
        break

# 추가된 프레임들로부터 GIF 생성
duration = 1000/fps
imageio.mimsave('output/animated.gif', frames, duration=duration)

videoclip.release()
#output_video.release() #mp4
cv2.destroyAllWindows()


