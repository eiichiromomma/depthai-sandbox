'''
depth_post_processingを弄ってdepthとconfidenceMapを取得する例
'''

#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
# depthとconfidenceMap用にnodeは2つ作成してそれぞれ名前を付ける
xout = pipeline.create(dai.node.XLinkOut)
xoutConfMap = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("depth")
xoutConfMap.setStreamName("confidence_map")
# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)

config = depth.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 200
config.postProcessing.thresholdFilter.maxRange = 15000
config.postProcessing.decimationFilter.decimationFactor = 1
depth.initialConfig.set(config)

# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
# LinkはStereoDepthのpipelineに
depth.depth.link(xout.input)
depth.confidenceMap.link(xoutConfMap.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    while True:
        # queueEventsをdepthとconfidence_mapで発生させてそれぞれについて
        # getOutputQueue.tryGetAll()を呼ぶ。取得できてたらFIFOだからかpacketの最後を取得
        latestPacket = {}
        latestPacket['depth'] = None
        latestPacket['confidence_map'] = None
        frame = None
        queueEvents = device.getQueueEvents(('depth','confidence_map'))
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]
        # 取得できてたらデータを呼び出す。
        # confidence_mapはgetCvFrameで読んでそのままimshowに放る
        if latestPacket['depth'] is not None:
            frame = latestPacket['depth'].getFrame()
        if latestPacket['confidence_map'] is not None:
            frameC = latestPacket['confidence_map'].getCvFrame()
            cv2.imshow('confidence', frameC)

        if frame is not None:       
            binframe = frame.copy()
            binframe[binframe < 400] = 0
            binframe[binframe > 600] = 0
            binframe[binframe != 0] = 255
            binframe = binframe.astype(np.uint8)
            cv2.imshow('binframe', binframe)
            frame = frame/2000.
            frame = 0.999 - frame
            frame[frame < 0] = 0
            cv2.imshow("depth", frame)

        if cv2.waitKey(1) == ord('q'):
            break