'''
depth_post_processingを弄って特定距離の対象だけを2値画像で表示する
disparityではなくdepthを使うように変更している。
depthはuint16でmm単位
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
xout = pipeline.create(dai.node.XLinkOut)
# 出力のうちdepthを使う
xout.setStreamName("depth")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# Create a node that will produce the depth map
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
# xoutと出力のdepthをリンク
depth.depth.link(xout.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # depthを呼び出す
    q = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        inDepth = q.get()  # blocking call, will wait until a new data has arrived
        # depthは uint16(mm単位)        
        frame = inDepth.getFrame()
        # Normalization for better visualization
        binframe = frame.copy()
        # 400-600mmの範囲だけ255にする
        binframe[binframe < 400] = 0
        binframe[binframe > 600] = 0
        binframe[binframe != 0] = 255
        binframe = binframe.astype(np.uint8)
        cv2.imshow('binframe', binframe)
        # 2000mmで正規化
        frame = frame/2000.
        # 近い方を明るく，負値は0に
        frame = 0.999 - frame
        frame[frame < 0] = 0
        cv2.imshow("depth", frame)

        if cv2.waitKey(1) == ord('q'):
            break
# unlinkした方が繰り返し使うときにエラーが出にくい?
pipeline.unlink(monoLeft.out,depth.left)
pipeline.unlink(monoRight.out,depth.right)
pipeline.unlink(depth.depth,xout.input)