import numpy as np
import matplotlib.pyplot as plt

import cv2
import tqdm
from PIL import Image

VIDEO_EXTENSIONS = ['mov', 'avi', 'mp4']
IMAGE_EXTENSIONS = ['jpg', 'png', 'tif']

def loadImages(path, n=None):
    if path[-3:].lower() in VIDEO_EXTENSIONS:
        return getVideoFrames(path)
    
    elif os.path.isdir(path):
        images = []
        #images = [cv2.imread(path + '/' + f) for f in os.listdir(path)[:None] if f[-3:] in IMAGE_EXTENSIONS]
        for f in np.sort(os.listdir(path)[:n]):
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


def scatterThreads(videoPath):


    videoFrames = getVideoFrames(videoPath)

    #regionOfInterest = [[60, 340], # y
    #                    [350, 620]] # x

    #videoFrames = [v[regionOfInterest[0][0]:regionOfInterest[0][1],regionOfInterest[1][0]:regionOfInterest[1][1]] for v in videoFrames]

    threshold = 30

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
    
    dsFactor = 10

    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')

    ax.scatter(scatterPoints[::dsFactor,2], scatterPoints[::dsFactor,0], scatterPoints[::dsFactor,1], s=.2)

    ax.view_init(-160, 60)
    plt.show()

if __name__ == '__main__':

    videoPath = '/home/jack/Videos/WebScanTest_2_2023-01-26_Cropped.mp4'

    scatterThreads(videoPath)
