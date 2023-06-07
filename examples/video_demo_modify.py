import sys
sys.path.append('../')

import cv2
import imageio
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints

import argparse
import time
import datetime

def human_pose(video_filename, save_option):

    estimator = BodyPoseEstimator(pretrained=True)
    video_dir = "./input/" + video_filename    
    videoclip = cv2.VideoCapture(video_dir)

    width  = int(videoclip.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoclip.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    =(videoclip.get(cv2.CAP_PROP_FPS))

    frames = []
    start = time.time()
    while videoclip.isOpened():
        flag, frame = videoclip.read()
        if not flag:
            break
        keypoints = estimator(frame)
        frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
        frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)

        
        if(save_option == 'Y'):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            frames.append(frame_rgb)

        cv2.imshow('Video Demo', frame)
        if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
            break
    end = time.time()
    sec = end - start
    result = datetime.timedelta(seconds=sec)
    print(result, end='')
    print(f",{sec:f} sec")
    
    if(save_option == 'Y'):
        duration = 1000/fps
        imageio.mimsave("./output/"+video_filename.split('.')[0]+"_output.gif", frames, duration=duration)

    videoclip.release()
    cv2.destroyAllWindows()
    print("-> All the processes are done.")
    if(save_option == 'Y'):
        print("The output GIF was saved in here: " + "./output/" + video_filename.split('.')[0]+"_output.gif")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_filename', required=True, help='File name of video')
    parser.add_argument('--save_option', required=True, help='Whether to save output as GIF(Answer in Y/N)')

    args = parser.parse_args()

    human_pose(args.video_filename, args.save_option)
