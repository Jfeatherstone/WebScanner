import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import argparse
import cv2
import tqdm
import os
from PIL import Image

# When providing an input path, we can either specify a video
# file with one of the former extensions, or a directory containing
# images with any of the latter extensions
VIDEO_EXTENSIONS = ['mov', 'avi', 'mp4']
IMAGE_EXTENSIONS = ['jpg', 'png', 'tif']

def loadImages(path, n=None):
    """
    Load in images from either a video file, or from a directory
    of images.

    This yields each item as a generator to reduce the amount of
    memory required, which is especially helpful for high resolution
    images.

    Parameters
    ----------
    path : str
        Path to a video file or directory of images.

    n : int or None
        The maximum number of images to load, if less than the
        available number is desired.
    """
    
    if path[-3:].lower() in VIDEO_EXTENSIONS:
        # If we have a video file, we read each frame using opencv
        cam = cv2.VideoCapture(path)
       
        # To keep track of how many images we have, in case some n
        # is provided
        i = 0
        while(True):
            if n and i > n:
                break

            # Ret will be false if we've come to the end of the video
            ret, frame = cam.read()

            if ret:
                yield frame.astype(np.uint8)
                i += 1
            else:
                break
    
    elif os.path.isdir(path):
        # If we have a directory, we take every file that is an image
        for f in tqdm.tqdm(np.sort(os.listdir(path)[:n])):

            if f[-3:].lower() in IMAGE_EXTENSIONS:
                yield np.array(cv2.imread(path + '/' + f)).astype(np.uint8)


def getNumFrames(path, n=None):
    """
    Returns the number of images to be loaded.

    Since the images are loaded via a generator, we cannot
    efficiently read how many total images there are directory
    from the generator object.

    Parameters
    ----------
    path : str
        Path to a video file or directory of images.

    n : int or None
        The maximum number of images to load. If provided,
        this function will return min(available_images, n).
    """

    if path[-3:].lower() in VIDEO_EXTENSIONS:
        cam = cv2.VideoCapture(path)
        length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    elif os.path.isdir(path):
        length = len([p for p in os.listdir(path) if p[-3:].lower() in IMAGE_EXTENSIONS])

    return length if not n else np.min(length, n)


def generateAdjMat(scatterPoints, neighborThreshold):
    """
    Construct the unweighted adjacency matrix of a collection of
    points.

    NOTE: Computes the adjacency matrix using outer subtraction,
    and therefore all at the same time! This means that the function
    will require an unholy amount of memory for large (>100000) 
    numbers of points.

    In this case, it is recommded to first filter out uneccessary points,
    or to use the batch method optimized for large numbers of points,
    batchGenerateAdjMat().

    Parameters
    ----------
    scatterPoints : numpy.ndarray[N, d]
        Positions of N points in d-dimensional space.

    neighborThreshold : float
        The maximum distance between two points for which they will
        not be considered to be neighbors.
    """
    # Construct the adjacency matrix
    distanceMat = np.zeros((scatterPoints.shape[0], scatterPoints.shape[0]))

    # Calculate node-node distances
    for i in range(scatterPoints.shape[-1]):
        distanceMat += np.subtract.outer(scatterPoints[:,i], scatterPoints[:,i])**2

    # Put 1s where the distance is less than the threshold, and 0s elsewhere
    adjMat = np.where(distanceMat < neighborThreshold**2, 1, 0)

    return adjMat


# If you get errors about not being able to allocate enough space
# for arrays, or generally are just utilizing too much memory,
# this value should be reduced.
DEFAULT_MAX_POINTS_PER_GROUP = 40000

def batchCalculateNumNeighbors(scatterPoints, neighborThreshold, maxNeighbors=300, maxPointsPerGroup=DEFAULT_MAX_POINTS_PER_GROUP, finalRemove=True):
    """
    Calculate the number of neighbors (points within a certain distance) each
    point has, optimized for large numbers of points.

    1. Points are split among M groups, such that no groups has more than
        maxPointsPerGroup points.
    2. The number of neighbors for points within each group is calculated.
    3. Points with too many neighbors (higher than maxNeighbors) are thrown
        out.
    4. One-by-one, groups are merged together, and steps 2-3 are repeated for
        the merged group.
    """
    # First, we partition our points into groups
    randomPoints = np.copy(scatterPoints)

    # Randomly order the points
    order = np.arange(randomPoints.shape[0])
    np.random.shuffle(order)
    randomPoints = randomPoints[order]

    # Calculate how many groups we need
    numGroups = int(np.ceil(randomPoints.shape[0] / maxPointsPerGroup))

    pointGroups = []
    for i in range(numGroups):
        pointGroups.append(randomPoints[i*maxPointsPerGroup:(i+1)*maxPointsPerGroup,:])

    while numGroups > 1:
        print(f'Groups: {numGroups}')

        # Now we calculate the number of neighbors within each group
        # TODO: distribute across processors
        for i in range(numGroups):
            # Calculate number of neighbors
            numNeighbors = calculateNumNeighbors(pointGroups[i], neighborThreshold, int(maxNeighbors/numGroups))
            # Determine which ones will be removed
            includedPoints = np.array(numNeighbors < int(maxNeighbors/numGroups), dtype=bool)
            # Remove them
            pointGroups[i] = pointGroups[i][includedPoints,:]

        # Pairwise merge groups
        mergedGroups = []
        for i in range(int(np.ceil(numGroups / 2))):
            mergedGroups.append(np.concatenate(pointGroups[2*i:2*(i+1)]))

        pointGroups = mergedGroups
        numGroups = len(mergedGroups)

    # Finally, calculate the true number of neighbors for the full group
    numNeighbors = calculateNumNeighbors(pointGroups[0], neighborThreshold, maxNeighbors)

    # Return the points as well, since they have changed order
    if finalRemove:
        return pointGroups[0][numNeighbors < maxNeighbors], numNeighbors[numNeighbors < maxNeighbors]
    else:
        return pointGroups[0], numNeighbors


def calculateNumNeighbors(scatterPoints, neighborThreshold, maxNeighbors=None):
    """
    Calculate the number of neighbors (points within a certain distance) each
    point has.

    This can also be achieved by summing the rows of the adjacency matrix:

    ```
    adjMat = generateAdjMat(points, distance)
    numNeighbors = np.sum(adjMat, axis=0)
    ```

    but this requires few enough points that a full adjacency matrix can
    feasibly be computed.

    Parameters
    ----------
    scatterPoints : numpy.ndarray[N, d]
        Positions of N points in d-dimensional space.

    neighborThreshold : float
        The maximum distance between two points for which they will
        not be considered to be neighbors.

    maxNeighbors : int or None
        The upper limit of neighbors for points that we care about. If
        provided, points with more neighbors than this value will be
        ignored as the calculation is performed, providing a speed-up.
    """

    numNeighbors = np.zeros(scatterPoints.shape[0])
    # If provided with a maximum number of neighbors (above which we
    # don't care about that point) we can reduce the number of calculations
    # as we go, which will speed up the computation.
    includedPoints = np.ones(scatterPoints.shape[0], dtype=bool)

    # If not provided, set the maximum number of neighbors to be 1 greater
    # than the number of points (so all points will always be included).
    if not maxNeighbors:
        maxNeighbors = scatterPoints.shape[0] + 1

    for i in tqdm.tqdm(range(scatterPoints.shape[0]), desc='Computing neighbors'):
        # Euclidian distance
        distances = np.sum((scatterPoints[i] - scatterPoints[includedPoints])**2, axis=-1)
        # -1 to account for the point itself being counted as a neighbor
        numNeighbors[i] = len(np.where(distances < neighborThreshold**2)[0]) - 1
        includedPoints[i] = numNeighbors[i] <= maxNeighbors

    return numNeighbors
        

def scatterThreads(inputPath, outputPath, regionOfInterest, greenChannel, threshold, frameSpacing, dsFactor, fps, dtheta, interactive, showNeighbors):

    scatterPoints = []
    
    i = 0
    # It's best to load the images through a generator, since we then don't
    # need to have every image active at the same time
    for image in tqdm.tqdm(loadImages(inputPath), desc='Processing images', total=getNumFrames(inputPath)):

        # Crop to ROI
        croppedImage = image[regionOfInterest[0][0]:regionOfInterest[0][1],regionOfInterest[1][0]:regionOfInterest[1][1],greenChannel]

        # Find non-zero pixel values and save their coordinates
        planarPoints = np.array(np.where(croppedImage  > threshold)).T
        scatterPoints += [(*p, frameSpacing*i) for p in planarPoints]
        i += 1

    # Downsample
    scatterPoints = np.array(scatterPoints, dtype=np.float32)[::dsFactor]
    print(scatterPoints.shape)
    

    if showNeighbors:
        scatterPoints, numNeighbors = batchCalculateNumNeighbors(scatterPoints, 15, 400)
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
    parser.add_argument('--channel', dest='channel', type=int, help='Color channel of image to analyze (0, 1, or 2)', default=1)
    parser.add_argument('--fps', dest='fps', type=int, help='FPS of output gif', default=20)
    parser.add_argument('--spacing', dest='spacing', type=float, help='Step spacing along scan direction', default=1)
    parser.add_argument('--dt', dest='dtheta', type=float, help='Difference in angle between each subsequent view', default=1)
    parser.add_argument('--interactive', dest='interactive', help='Whether to show an interactive plot instead of saving a movie', action='store_const', const=True, default=False)
    parser.add_argument('--neighbors', dest='neighbors', help='Whether to color scatter points based on the number of neighbors', action='store_const', const=True, default=False)

    args = parser.parse_args()

    #regionOfInterest = [[1400, 3750], # y
    #                    [2300, 5400]] # x

    regionOfInterest = [[None, None], [None, None]]

    scatterThreads(args.inputPath, args.outputPath, regionOfInterest, args.channel,
                   args.threshold, args.spacing, args.downsample, args.fps,
                   args.dtheta, args.interactive, args.neighbors)
