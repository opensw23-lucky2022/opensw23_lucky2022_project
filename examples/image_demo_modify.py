import sys
sys.path.append('../')

import cv2
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints

import argparse

def human_pose(image_filename, save_option):
    
    estimator = BodyPoseEstimator(pretrained=True)
    image_dir = "./input/" + image_filename
    
    image_src = cv2.imread(image_dir)
    keypoints = estimator(image_src)
    image_dst = draw_body_connections(image_src, keypoints, thickness=4, alpha=0.7)
    image_dst = draw_keypoints(image_dst, keypoints, radius=5, alpha=0.8)

    if(save_option == 'Y'):
        cv2.imwrite("./output/" + image_filename.split('.')[0]+"_output.png",image_dst)
    
    while True:
        cv2.imshow('Image Demo', image_dst)
        if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
            break
    cv2.destroyAllWindows()
    print("-> All the processes are done.")
    if(save_option == 'Y'):
        print("The output image was saved in here: " + "./output/" + image_filename.split('.')[0]+"_output.png")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_filename', required=True, help='File name of video')
    parser.add_argument('--save_option', required=True, help='Whether to save output as JPG(Answer in Y/N)')

    args = parser.parse_args()

    human_pose(args.image_filename, args.save_option)
