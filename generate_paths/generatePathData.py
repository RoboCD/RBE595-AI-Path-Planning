import os
import glob
import numpy as np
# import cv2 as cv
from skimage import io, transform
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import pandas as pd
from warnings import warn
import heapq

# A* algorithm
class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __repr__(self):
      return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
      return self.f < other.f
    
    # defining greater than for purposes of heap queue
    def __gt__(self, other):
      return self.f > other.f

def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path


def astar(maze, start, end, allow_diagonal_movement = False):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    :param maze:
    :param start:
    :param end:
    :return:
    """

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Heapify the open_list and Add the start node
    heapq.heapify(open_list) 
    heapq.heappush(open_list, start_node)

    # Adding a stop condition
    outer_iterations = 0
    max_iterations = (len(maze[0]) * len(maze) // 2)

    # what squares do we search
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)
    if allow_diagonal_movement:
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1),)

    # Loop until you find the end
    while len(open_list) > 0:
        outer_iterations += 1

        if outer_iterations > max_iterations:
          # if we hit this point return the path such as it is
          # it will not contain the destination
          warn("giving up on pathfinding too many iterations")
          return return_path(current_node)       
        
        # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            return return_path(current_node)

        # Generate children
        children = []
        
        for new_position in adjacent_squares: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 1:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            if len([open_node for open_node in open_list if child.position == open_node.position and child.g > open_node.g]) > 0:
                continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    warn("Couldn't get a path to destination")
    return None

def main():

    # maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    projectDir = ".."
    dataPath = os.path.join(projectDir, "maps")
    mazesDir = os.path.join(dataPath, "mazes" )
    mazesTrainDir = os.path.join(mazesDir, "train/*.png")
    mazesValidateDir = os.path.join(mazesDir, "validation/*.png")
    mazesTestDir = os.path.join(mazesDir, "test/*.png")
    mazesDirs = [mazesTrainDir,mazesValidateDir, mazesTestDir]

    saveDataDir = os.path.join(projectDir, "data")
    # Loop through each of the mazes directories (train, validate, test)
    for mazeD in mazesDirs: 
        dirName = mazeD.split("/")[-2]
        print (f"Directory Name: {dirName}")
        # Loop through all images in directory
        mazesImgs = sorted(glob.glob(mazeD))
        print(len(mazesImgs))
        df = pd.DataFrame(columns = ['Image', 'Start', 'End', 'Path'])

        for imgFile in mazesImgs:

            # testImg = mazesTrainImgs[0]
            # imgFile = "../maps/mazes/train/35.png"
            print(f"image file path {imgFile}")
            # testImgPath = os.path.join(dataPath, "162.png")
            img = io.imread(imgFile, as_gray=True)
            # print(img.shape)
            # print(img)
            imgName = imgFile.split("/")[-1]

            # plt.imshow(img,"gray")
            # plt.show()
            # df = pd.DataFrame(img)
            # df.to_csv('imgTest.csv')

            downImg = transform.resize(img, (32, 32), anti_aliasing=False)
            # print(downImg.shape)
            # print(downImg)

            # plt.imshow(downImg,"gray")
            # plt.show()
            df2 = pd.DataFrame(downImg)
            df2.to_csv('imgTest2.csv')
            # Save resized image
            io.imsave(os.path.join(saveDataDir,"mazes",dirName,imgName), downImg)

            maze = downImg
            start = (maze.shape[0]-1, 0)
            end = (0, maze.shape[1]-1)
            # downImg[start] = 0.5
            # downImg[end] = 0.5
            # plt.imshow(downImg,"gray")
            # plt.show()
            # print(f"{start}")
            # print(f"{end}")
            path = astar(maze, start, end, allow_diagonal_movement = True)
            print(path)
            pathImg = np.zeros([32,32,1],dtype=np.uint8)
            pathImg.fill(255)
            for p in path:
                maze[p] = 0.5
                pathImg[p] = 0.5

            # Show combined path and maze image
            # plt.imshow(maze, "gray")
            # plt.show()
            # Save path as separate image
            io.imsave(os.path.join(saveDataDir,"mazes",dirName+"_path",imgName), pathImg)

            # ../maps/mazes/train/149.png
            print(f"imgName: {imgName}")
            imgIndex = int(imgName.split('.')[0])
            print(f"imgIndex: {imgIndex}")
            # print(f"imgIndex Type: {type(imgIndex)}")
            # print(f"imgIndex +1: {imgIndex+1}")

            dfRow = pd.DataFrame ([[imgName, start, end, path]], columns = ['Image','Start','End','Path'], index = [int(imgIndex)])
            df = df.append(dfRow)
            # df = df.append({"Image": imgName, "Start": str(start), "End": str(end), "Path": str(path), index=imgIndex})
            # df = df.append({"Image": imgName, "Start": str(start), "End": str(end), "Path": str(path) }, ignore_index=True)
            # d = {"Image": imgName, "start": str(start), "end": str(end), "path": str(path) }
            # df = pd.DataFrame(data=d)

        df = df.sort_index()
        df.to_csv(os.path.join(saveDataDir,"mazes",'mazes_{}.csv'.format(dirName)))


if __name__ == '__main__':
    main()
