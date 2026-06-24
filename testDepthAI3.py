
import random
from typing import List

# Library imports
import pygame

# pymunk imports
import pymunk
import pymunk.pygame_util
import numpy as np
import cv2
import depthai as dai
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True


# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
depth = pipeline.create(dai.node.StereoDepth)
depth.setPostProcessingHardwareResources(3, 3)
# Properties
# monoLeftOut = monoLeft.requestOutput((640, 400))
monoLeftOut = monoLeft.requestFullResolutionOutput()
# monoLeft.setCamera("left")
# monoRightOut = monoRight.requestOutput((640, 400))
monoRightOut = monoRight.requestFullResolutionOutput()
# monoRight.setCamera("right")
depth.initialConfig.postProcessing.decimationFilter.decimationFactor = 2

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
# depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.initialConfig.setLeftRightCheck(lr_check)
depth.initialConfig.setConfidenceThreshold(100)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)
depth.initialConfig.postProcessing.spatialFilter.enable = True
depth.initialConfig.postProcessing.spatialFilter.holeFillingRadius = 4
depth.initialConfig.postProcessing.spatialFilter.numIterations = 1
depth.initialConfig.postProcessing.spatialFilter.alpha = 0.5
depth.initialConfig.postProcessing.spatialFilter.delta = 20
depth.initialConfig.postProcessing.temporalFilter.enable = True
depth.initialConfig.postProcessing.speckleFilter.enable = True
depth.initialConfig.postProcessing.speckleFilter.speckleRange = 5
# depth.initialConfig.postProcessing.thresholdFilter.minRange = 200
# depth.initialConfig.postProcessing.thresholdFilter.maxRange = 15000

# Linking
monoLeftOut.link(depth.left)
monoRightOut.link(depth.right)
depthQueue = depth.depth.createOutputQueue(maxSize=4, blocking=False)
maxdist = 2000
mindist = 250
with pipeline:
    pipeline.start()
    while pipeline.isRunning():
    
        # Progress time forward
        inDepth = depthQueue.get()  # blocking call, will wait until a new data has arrived
        img = inDepth.getFrame()
        bgimg = cv2.convertScaleAbs(img, alpha=(255./maxdist))
        bgimg[bgimg == 255] = 0
        bgimg[bgimg < 255.*mindist/maxdist] = 0
        print(bgimg.max())
        cv2.imshow('test',bgimg)
        cv2.waitKey(10)