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
# depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
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
        self._dt = 1.0 / 60.0
        self._physics_steps_per_frame = 1

        # pygame
        pygame.init()
        # pygame.mouse.set_visible(True)
        
        # 💡 解像度を固定変数として保持
        self.window_w = 640
        self.window_h = 400
        self._screen = pygame.display.set_mode((self.window_w, self.window_h))
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)
        self._balls: List[pymunk.Circle] = []

        self._running = True
        self._ticks_to_next_ball = 10

        # depthai
        pygame.mouse.set_visible(False)
        self.hbfactor = 40
        self.hbs = []
        self.hbr = []
        self.mindist = 500.0 
        self.maxdist = 1000.0 
        self.mirror = False
                self._bg_surface = pygame.Surface((self.window_w, self.window_h))

    def run(self) -> None:
        """
        The main loop of the game.
        """
        with pipeline:
            pipeline.start()
            while pipeline.isRunning() and self._running:
            
                for x in range(self._physics_steps_per_frame):
                    self._space.step(self._dt)
                
                inDepth = depthQueue.tryGet()  
                if inDepth is not None:
                    img = inDepth.getFrame()  
                    self._capture(img)
                    self._update_obstacles() # 新しいフレームが来た時だけCオブジェクトを更新

                self._process_events()
                self._spawn_and_clean_balls() # ボールの発生と削除だけを毎フレーム行う
                self._clear_screen()
                self._draw_objects()
                pygame.display.flip()
                
                self._clock.tick(50)
                pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

    def _capture(self, img):
        if self.mirror:
            img = np.fliplr(img)
        
        bg = np.clip(img * (255. / self.maxdist), 0, 255).astype(np.uint8)
        bg[bg == 255] = 0
        bg[bg < 255. * self.mindist / self.maxdist] = 0
        
        bg_rgb = np.repeat(bg[:, :, np.newaxis], 3, axis=2)
        
        bg_transposed = np.transpose(bg_rgb, (1, 0, 2))
        
        temp_surf = pygame.surfarray.make_surface(bg_transposed)
        pygame.transform.scale(temp_surf, (self.window_w, self.window_h), self._bg_surface)

        img_small = img[::self.hbfactor, ::self.hbfactor]
        
        self.hbr, self.hbc = np.where((self.mindist < img_small) & (img_small < self.maxdist))

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


    def _clear_screen(self) -> None:
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(pygame.Color("white"))

    def _update_obstacles(self) -> None:
        if len(self.hbs) > 0:
            for s in self.hbs:
                self._space.remove(s, s.body)
            self.hbs = []
            
        if len(self.hbr) > 0:
            for i in range(len(self.hbr)):
                b = pymunk.Body(body_type=pymunk.Body.STATIC)
                s = pymunk.Circle(b, self.hbfactor//2-1, (self.hbc[i]*self.hbfactor, self.hbr[i]*self.hbfactor))
                s.color = (128, 128, 128, 20)
                s.elasticity = 0.95
                s.friction = 0.9
                self._space.add(b, s)
                self.hbs.append(s)

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

    def _spawn_and_clean_balls(self) -> None:
        self._ticks_to_next_ball -= 1
        if self._ticks_to_next_ball <= 0:
            self._create_ball()
            self._ticks_to_next_ball = 10
            
        # 画面外（下部）に落ちたボールを削除
        balls_to_remove = [ball for ball in self._balls if ball.body.position.y > 500]
        for ball in balls_to_remove:
            self._space.remove(ball, ball.body)
            self._balls.remove(ball)

    def _draw_objects(self) -> None:
        self._screen.blit(self._bg_surface, (0, 0))
        self._space.debug_draw(self._draw_options)

def main():
    game = BouncyBalls()
    game.run()


if __name__ == "__main__":
    main()
