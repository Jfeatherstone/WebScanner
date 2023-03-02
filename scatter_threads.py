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
        cam = cv2.VideoCapture(path)
        
        i = 0
        while(True):
            if n and i > n:
                break

            ret, frame = cam.read()

            if ret:
                yield frame.astype(np.uint8)
                i += 1
            else:
                break
    
    elif os.path.isdir(path):

        for f in tqdm.tqdm(np.sort(os.listdir(path)[:n])):

            if f[-3:].lower() in IMAGE_EXTENSIONS:
                yield np.array(cv2.imread(path + '/' + f)).astype(np.uint8)


def getNumFrames(path, n=None):
 
    if path[-3:].lower() in VIDEO_EXTENSIONS:
        cam = cv2.VideoCapture(path)
        length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    elif os.path.isdir(path):
        length = len([p for p in os.listdir(path) if p[-3:].lower() in IMAGE_EXTENSIONS])

    return length if not n else np.min(length, n)


def generateAdjMat(scatterPoints, neighborThreshold):
    # Construct the adjacency matrix
    distanceMat = np.zeros((scatterPoints.shape[0], scatterPoints.shape[0]))

    # Calculate node-node distances
    for i in range(scatterPoints.shape[-1]):
        distanceMat += np.subtract.outer(scatterPoints[:,i], scatterPoints[:,i])**2

    adjMat = np.where(distanceMat < neighborThreshold**2, 1, 0)

    return adjMat

def calculateNumNeighbors(scatterPoints, neighborThreshold):

    numNeighbors = np.zeros(scatterPoints.shape[0])

    for i in range(scatterPoints.shape[0]):
        distances = np.sum((scatterPoints[i] - scatterPoints[:])**2, axis=-1)
        numNeighbors[i] = len(np.where(distances < neighborThreshold**2)[0])

    return numNeighbors
        

def scatterThreads(inputPath, outputPath, regionOfInterest, threshold, frameSpacing, dsFactor, fps, dtheta, interactive, showNeighbors):

    scatterPoints = []
    
    i = 0
    # It's best to load the images through a generator, since we then don't
    # need to have every image active at the same time
    for image in tqdm.tqdm(loadImages(inputPath), desc='Processing images', total=getNumFrames(inputPath)):

        # Crop to ROI
        croppedImage = image[regionOfInterest[0][0]:regionOfInterest[0][1],regionOfInterest[1][0]:regionOfInterest[1][1]]
         
        # Take threshold
        thresholdFrame = croppedImage
        thresholdFrame[np.where(thresholdFrame < threshold)] = 0

        # Find non-zero pixel values and save their coordinates
        planarPoints = np.array(np.where(thresholdFrame > 0)).T
        scatterPoints += [(*p, frameSpacing*i) for p in planarPoints]
        i += 1

    # Downsample
    scatterPoints = np.array(scatterPoints)[::dsFactor]
    print(scatterPoints.shape)
    

    if showNeighbors:
        numNeighbors = calculateNumNeighbors(scatterPoints, 25) - 1#np.sum(generateAdjMat(scatterPoints, 25), axis=1) - 1
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
            colorbar.set_label('Neighbors', fontsize=18)

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

    #regionOfInterest = [[1400, 3750], # y
    #                    [2300, 5400]] # x

    regionOfInterest = [[None, None], [None, None]]

    scatterThreads(args.inputPath, args.outputPath, regionOfInterest,
                   args.threshold, args.spacing, args.downsample, args.fps,
                   args.dtheta, args.interactive, args.neighbors)
