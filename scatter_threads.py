import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

def generateAdjMat(scatterPoints, neighborThreshold):
    # Construct the adjacency matrix
    distanceMat = np.zeros((scatterPoints.shape[0], scatterPoints.shape[0]))

    # Calculate node-node distances
    for i in range(scatterPoints.shape[-1]):
        distanceMat += np.subtract.outer(scatterPoints[:,i], scatterPoints[:,i])**2

    adjMat = np.where(distanceMat < neighborThreshold**2, 1, 0)

    return adjMat


def scatterThreads(inputPath, outputPath, regionOfInterest, threshold, frameSpacing, dsFactor, fps, dtheta, interactive, showNeighbors):

    frames = loadImages(inputPath)

    # Crop to ROI
    frames = [v[regionOfInterest[0][0]:regionOfInterest[0][1],regionOfInterest[1][0]:regionOfInterest[1][1]] for v in frames] 
    print(f'Loaded {len(frames)} images.')

    thresholdFrames = np.zeros((len(frames), *frames[0].shape[:2]))

    for i in range(len(thresholdFrames)):
        thresholdFrames[i] = frames[i][:,:,1]
        thresholdFrames[i][np.where(thresholdFrames[i] < threshold)] = 0

    scatterPoints = []

    for i in range(len(thresholdFrames)):
        planarPoints = np.array(np.where(thresholdFrames[i] > 0)).T

        scatterPoints += [(*p, frameSpacing*i) for p in planarPoints]

    scatterPoints = np.array(scatterPoints)[::dsFactor]

    if showNeighbors:
        numNeighbors = np.sum(generateAdjMat(scatterPoints, 25), axis=1) - 1
    else:
        numNeighbors = np.zeros(scatterPoints.shape[0])

    images = []
    loop = 0

    for i in tqdm.tqdm(range(int(360 / dtheta)), desc='Generating movie') if not interactive else range(1):
        fig = plt.figure(figsize=(9,9))

        ax = fig.add_subplot(projection='3d')

        scatterPlot = ax.scatter(scatterPoints[:,2], scatterPoints[:,1], scatterPoints[:,0], s=.2, c=numNeighbors, norm=LogNorm())
        #ax.set_zlim([0, regionOfInterest[0][1] - regionOfInterest[0][0]])
        #ax.set_ylim([0, regionOfInterest[1][1] - regionOfInterest[1][0]])
        ax.view_init(-160, i*dtheta)
        
        if showNeighbors:
            colorbar = fig.colorbar(scatterPlot, orientation='horizontal')
            colorbar.set_label('Neighbors')

        fig.tight_layout()

        if interactive:
            plt.show()
            return

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()

        images.append(Image.frombytes('RGB', canvas.get_width_height(),
                     canvas.tostring_rgb()))
        
        plt.close()

    images[0].save(outputPath, save_all=True, append_images=images[1:], duration=fps, loop=loop)
    return



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(dest='inputPath', type=str)
    parser.add_argument('-o', dest='outputPath', help='Output file path for the gif', default='output.gif')
    parser.add_argument('-t', dest='threshold', type=float, help='Threshold pixel value to include in scatter', default=30)
    parser.add_argument('--ds', dest='downsample', type=int, help='Factor to downsample scatter points by', default=10)
    parser.add_argument('--fps', dest='fps', type=int, help='FPS of output gif', default=20)
    parser.add_argument('--spacing', dest='spacing', type=float, help='Step spacing along scan direction', default=1)
    parser.add_argument('--dt', dest='dtheta', type=float, help='Difference in angle between each subsequent view', default=1)
    parser.add_argument('--interactive', dest='interactive', help='Whether to show an interactive plot instead of saving a movie', action='store_const', const=True, default=False)
    parser.add_argument('--neighbors', dest='neighbors', help='Whether to color scatter points based on the number of neighbors', action='store_const', const=True, default=False)

    args = parser.parse_args()

    regionOfInterest = [[1400, 3750], # y
                        [2300, 5400]] # x

    #regionOfInterest = [[None, None], [None, None]]

    scatterThreads(args.inputPath, args.outputPath, regionOfInterest,
                   args.threshold, args.spacing, args.downsample, args.fps,
                   args.dtheta, args.interactive, args.neighbors)
