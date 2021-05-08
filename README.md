# About YOLO-V2

YOLO ("you only look once") is a popular deep learning-based algorithm with high accuracy and good real-time performance. In yolo, object detection is framed as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Yolo-v2 is the improvised version of the initially proposed Yolo detection method (YOLO-V1) and has better accuracies and speed compared to that of YOLO-V1.

The architecture of yolo-v2 CNN is as shown below.

![image](https://user-images.githubusercontent.com/83395271/117542689-43ac7e80-b037-11eb-855a-5fbd478d546d.png)

# Algorithm formulation:

In YOLO-V2, the input to the network is an image or video frame of shape (608, 608, 3) or (416, 416, 3). The encoded output is of shape (19,19,5,85) or (13,13,5,85). Each grid cell in the output is associated with 5 anchor boxes of different dimensions. These anchor boxes are used to detect multiple objects if their centers fall in the same grid. In this project, we have chosen the input image size of (608 x 608). So, the output is a (19x19) size grid cell map and each grid cell is associated with 5 bounding boxes of different sizes. Each bounding box is represented by 6 numbers (pc, bx, by, bh, bw, c). Here c is an 80-dimensional vector which represents the probability scores of each class, pc represents the confidence score for the presence of object in the bounding box, bx and by represent the x and y coordinates of bounding box, bh and bw represent the height and width of the bounding box. So, each bounding box is represented by 85 numbers.

So, we can think of the YOLO architecture as the following: 
IMAGE (608, 608, 3) -> DEEP CNN -> ENCODING (19, 19, 5, 85).

![image](https://user-images.githubusercontent.com/83395271/117542735-62127a00-b037-11eb-8a91-0520acd004be.png)

Now, for each box (of each cell) we will compute the following elementwise product and extract a probability that the box contains a certain class.

![image](https://user-images.githubusercontent.com/83395271/117542747-6c347880-b037-11eb-9a34-fa43389fc52e.png)

## Retaining the bounding boxes and class labels with maximum class probability

For each of the 19x19 grid cells, the maximum of the probability scores is found (taking a max across the 5 anchor boxes and across different classes).  If the bounding boxes with highest probability scores among classes are plotted, the output can be visualized as shown below:

![image](https://user-images.githubusercontent.com/83395271/117542762-83736600-b037-11eb-8a4d-9cc7df6348ae.png)

Since, there are still many number of bounding boxes, the algorithm's output has to be filtered down to a much smaller number of detected objects. This can be achieved by, first filtering the boxes with a threshold on class scores and then followed by non-maximum suppression.

## Filtering with a threshold on class scores

In this method, the boxes with confidence score less than a threshold are removed.

## Non-maximal suppression

In this method, if the intersection over union (IOU) area between the boxes with same class is greater than a threshold, then all the other boxes are removed keeping only the bounding box with highest probability score.

# Implementation on python using OpenCV

In this project, 3 types of files are used to build the network. They are:

### (a) yolov2.cfg file:
This is the configuration file containing the details of yolo CNN layer’s architecture and the width and height details of anchor boxes of each grid cell.

### (b) yolov2.weights file: 
This file contains the pretrained weights of the CNN used for yolo object detection  

### (c) coco_classes.txt file:
This file contains the labels of 80 different classes considered for training yolo model.

## Implementation steps:
1.	In OpenCV, the network is built by reading the config file and weights fille using cv2.dnn.readNetFromDarknet() command. This returns the pretrained yolo v2 network. 
2.	The input image or video frame is scaled to (0,1) range, resized to the shape 608 x 608 and fed as input to this network to get an output map of shape (1805,85). Here, 1805 is the total no. of bounding boxes obtained by considering 5 bounding boxes to each of the grid cell in 19 x 19 size output grid map. And 85 is the total no. of bounding box parameters per each box (i.e., pc, bx, by, bh, bw, c).
3.	Filter out the bounding boxes and class labels whose class probability scores are greater than a threshold.
4.	Apply non-maximal suppression on the bounding boxes obtained in the above step and filter them based on the IOU ratio between the boxes. In the code, cv2.dnn.NMSBoxes() command is used for non-maximal suppression.
5.	Using the bounding box parameters, class labels and class probability scores, boxes are drawn on the image or the video frame and the output image is displayed.

# Results

### Objects detected on sample image
![image](https://user-images.githubusercontent.com/83395271/117542883-1ad8b900-b038-11eb-893b-fd1f1753d3e1.png)

### Objects detected on Live video
![image](https://user-images.githubusercontent.com/83395271/117542914-35129700-b038-11eb-9547-a4dfeb40dea8.png)

