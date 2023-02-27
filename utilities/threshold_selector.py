import numpy as np
import matplotlib.pyplot as plt
import argparse

import cv2
import tqdm
from PIL import Image

#videoPath = '/home/jack/Videos/2023-02-09_Test_Cropped_Power1.mp4'
thresholdValues = [10, 30, 60, 100]
greenChannel = 1

regionOfInterest = [[340, 800], # y
                    [700, 1200]] # x

def getVideoFrames(videoPath):
    cam = cv2.VideoCapture(videoPath)

    frames = []

    while(True):
        ret, frame = cam.read()

        if ret:
            frames.append(frame.astype(np.uint8))

        else:
            break

    return frames

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='videoPath')

    args = parser.parse_args()
    
    videoPath = args.videoPath
    videoName = videoPath.split('/')[-1][:-4]
    videoFrames = getVideoFrames(videoPath)

    #regionOfInterest = [[None,None], [None,None]]


    videoFrames = [v[regionOfInterest[0][0]:regionOfInterest[0][1],regionOfInterest[1][0]:regionOfInterest[1][1]] for v in videoFrames]

    thresholdFrames = np.zeros((len(thresholdValues), len(videoFrames), *videoFrames[0].shape[:2]))

    for j in range(len(thresholdValues)):

        for i in range(len(thresholdFrames[0])):
            thresholdFrames[j,i] = videoFrames[i][:,:,greenChannel]
            thresholdFrames[j,i][np.where(thresholdFrames[j,i] < thresholdValues[j])] = 0
            thresholdFrames[j,i][np.where(thresholdFrames[j,i] > 0)] = 255 # Only for displaying, shouldn't be done in the reconstruction process

    images = []

    fps = 20
    loop = 0
    dsFactor = 4

    for i in tqdm.tqdm(range(len(videoFrames)//dsFactor)):

        fig, ax = plt.subplots(1, len(thresholdFrames), figsize=(len(thresholdFrames)*5, 5))

        for j in range(len(thresholdFrames)):
            ax[j].imshow(thresholdFrames[j][i*dsFactor])
            ax[j].set_xticks([])
            ax[j].set_yticks([])
            ax[j].set_title(f'Threshold: {thresholdValues[j]}')

        fig.suptitle(f'Frame {i*dsFactor}')
        fig.tight_layout()
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()

        images.append(Image.frombytes('RGB', canvas.get_width_height(),
                     canvas.tostring_rgb()))

        plt.close()

    images[0].save(f'images/{videoName}_threshold_testing.gif', save_all=True, append_images=images[1:], duration=fps, loop=loop)
