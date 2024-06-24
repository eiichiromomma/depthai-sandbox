'''
depthai pymunk pygame
'''


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
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
xout = pipeline.create(dai.node.XLinkOut)

xout.setStreamName("depth")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
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
depth.depth.link(xout.input)
class BouncyBalls(object):
    """
    This class implements a simple scene in which there is a static platform (made up of a couple of lines)
    that don't move. Balls appear occasionally and drop onto the platform. They bounce around.
    """

    def __init__(self) -> None:
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 900.0)

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.init()
        self._screen = pygame.display.set_mode((640, 400))
        # self._screen = pygame.display.set_mode((640, 400),pygame.FULLSCREEN)
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        # Balls that exist in the world
        self._balls: List[pymunk.Circle] = []

        # Execution control and time until the next ball spawns
        self._running = True
        self._ticks_to_next_ball = 10

        # depthai
        pygame.mouse.set_visible(False)
        self.hbfactor = 40

        self.hbs = []

        self.mindist = 500.0 # mm
        self.maxdist = 1000.0 # mm
        self.mirror = False


    def run(self) -> None:
        """
        The main loop of the game.
        :return: None
        """
        # Main loop
        with dai.Device(pipeline) as device:
            queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            while self._running:
            
                # Progress time forward
                for x in range(self._physics_steps_per_frame):
                    self._space.step(self._dt)
                inDepth = queue.get()  # blocking call, will wait until a new data has arrived
                img = inDepth.getFrame()  
                self._capture(img)
                self._process_events()
                self._update_balls()
                self._clear_screen()
                self._draw_objects()
                pygame.display.flip()
                # Delay fixed time between frames
                self._clock.tick(50)
                pygame.display.set_caption("fps: " + str(self._clock.get_fps()))
    def _capture(self,img):
      
        if self.mirror:
            img = np.fliplr(img)
        self.bgimg = cv2.convertScaleAbs(img, alpha=(255./self.maxdist))
        self.bgimg[self.bgimg == 255] = 0
        self.bgimg[self.bgimg < 255.*self.mindist/self.maxdist] = 0
        self.bgimg = cv2.resize(self.bgimg,
        (pygame.display.Info().current_w,pygame.display.Info().current_h))
        self.bgimg = cv2.merge((self.bgimg,self.bgimg,self.bgimg))
        img = img.astype(np.float32)
        fxy = 1./self.hbfactor
        img = cv2.resize(img,None,fx=fxy,fy=fxy, interpolation=cv2.INTER_NEAREST)
        self.hbr, self.hbc = np.where((self.mindist<img) & (img < self.maxdist))

    def _process_events(self) -> None:
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(self._screen, "bouncing_balls.png")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.mirror = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self.mirror = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                if self.mindist > 250:
                    self.mindist -= 250;
                    self.maxdist -= 250;
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                    self.mindist = 500;
                    self.maxdist = 1000;
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                if self.mindist < 2000:
                    self.mindist += 250;
                    self.maxdist += 250;

    def _update_balls(self) -> None:
        """
        Create/remove balls as necessary. Call once per frame only.
        :return: None
        """
        if not len(self.hbs) == 0:
            for s in self.hbs:
                self._space.remove(s,s.body)
            self.hbs = []
        if not len(self.hbr) == 0:
            for i in range(len(self.hbr)):
                b = pymunk.Body(body_type=pymunk.Body.STATIC)
                s = pymunk.Circle(b, self.hbfactor//2-1, (self.hbc[i]*self.hbfactor,self.hbr[i]*self.hbfactor))
                s.color = (128,128,128,20)
                s.elasticity = 0.95
                s.friction = 0.9
                self._space.add(b,s)
                self.hbs.append(s)
        self._ticks_to_next_ball -= 1
        if self._ticks_to_next_ball <= 0:
            self._create_ball()
            self._ticks_to_next_ball = 10
        # Remove balls that fall below 100 vertically
        balls_to_remove = [ball for ball in self._balls if ball.body.position.y > 500]
        for ball in balls_to_remove:
            self._space.remove(ball, ball.body)
            self._balls.remove(ball)

    def _create_ball(self) -> None:
        """
        Create a ball.
        :return:
        """
        mass = 10
        radius = random.randint(15,40)
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        x = random.randint(10, 630)
        body.position = x, random.randint(0,20)
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.color = (random.randint(0,255),random.randint(0,255),random.randint(0,255),255)
        shape.elasticity = 0.95
        shape.friction = 0.9
        self._space.add(body, shape)
        self._balls.append(shape)

    def _clear_screen(self) -> None:
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(pygame.Color("white"))

    def _draw_objects(self) -> None:
        """
        Draw the objects.
        :return: None
        """
        self.bgimg = self.bgimg.swapaxes(0,1)
        surf = pygame.surfarray.make_surface(self.bgimg)
        self._screen.blit(surf,(0,0))
        self._space.debug_draw(self._draw_options)
    def __del__(self):
        # self.depth_stream.stop()
        # openni2.unload()
        pipeline.unlink(monoLeft.out,depth.left)
        pipeline.unlink(monoRight.out,depth.right)
        pipeline.unlink(depth.depth,xout.input)

def main():
    game = BouncyBalls()
    game.run()


if __name__ == "__main__":
    main()
