import numpy as np
import matplotlib.pyplot as plt

import argparse
import cv2
import tqdm
import os
from PIL import Image

VIDEO_EXTENSIONS = ['mov', 'avi', 'mp4']
IMAGE_EXTENSIONS = ['jpg', 'png', 'tif']

def loadImages(path, n=None):
    if path[-3:].lower() in VIDEO_EXTENSIONS:
        return getVideoFrames(path)
    
    elif os.path.isdir(path):
        images = []
        #images = [cv2.imread(path + '/' + f) for f in os.listdir(path)[:None] if f[-3:] in IMAGE_EXTENSIONS]
        for f in tqdm.tqdm(np.sort(os.listdir(path)[:n])):
            if f[-3:].lower() in IMAGE_EXTENSIONS:
                images.append(cv2.imread(path + '/' + f))
            
        return images


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


def scatterThreads(inputPath, outputPath, regionOfInterest, threshold, dsFactor, fps, dtheta):


    videoFrames = loadImages(inputPath)


    videoFrames = [v[regionOfInterest[0][0]:regionOfInterest[0][1],regionOfInterest[1][0]:regionOfInterest[1][1]] for v in videoFrames]
    
    print(f'Loaded {len(videoFrames)} images.')

    thresholdFrames = np.zeros((len(videoFrames), *videoFrames[0].shape[:2]))

    for i in range(len(thresholdFrames)):
        thresholdFrames[i] = videoFrames[i][:,:,1]
        thresholdFrames[i][np.where(thresholdFrames[i] < threshold)] = 0

    frameSpacing = 1

    scatterPoints = []

    for i in range(len(thresholdFrames)):
        planarPoints = np.array(np.where(thresholdFrames[i] > 0)).T

        scatterPoints += [(*p, frameSpacing*i) for p in planarPoints]

    scatterPoints = np.array(scatterPoints)
    
    images = []

    loop = 0

    for i in tqdm.tqdm(range(int(360 / dtheta)), desc='Generating movie'):
        fig = plt.figure(figsize=(7,7))

        ax = fig.add_subplot(projection='3d')

        ax.scatter(scatterPoints[::dsFactor,2], scatterPoints[::dsFactor,1], scatterPoints[::dsFactor,0], s=.2)
        #ax.set_zlim([0, regionOfInterest[0][1] - regionOfInterest[0][0]])
        #ax.set_ylim([0, regionOfInterest[1][1] - regionOfInterest[1][0]])
        ax.view_init(-160, i*dtheta)
        
        fig.tight_layout()
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()

        images.append(Image.frombytes('RGB', canvas.get_width_height(),
                     canvas.tostring_rgb()))
        
        #plt.show()
        plt.close()

    images[0].save(outputPath, save_all=True, append_images=images[1:], duration=fps, loop=loop)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(dest='inputPath', type=str)
    parser.add_argument('-o', dest='outputPath', help='Output file path for the gif', default='output.gif')
    parser.add_argument('-t', dest='threshold', type=float, help='Threshold pixel value to include in scatter', default=30)
    parser.add_argument('--ds', dest='downsample', type=int, help='Factor to downsample scatter points by', default=10)
    parser.add_argument('--fps', dest='fps', type=int, help='FPS of output gif', default=20)
    parser.add_argument('--dt', dest='dtheta', type=float, help='Difference in angle between each subsequent view', default=1)

    args = parser.parse_args()

    regionOfInterest = [[1400, 3750], # y
                        [2300, 5400]] # x

    scatterThreads(args.inputPath, args.outputPath, regionOfInterest, args.threshold, args.downsample, args.fps, args.dtheta)
