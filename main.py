import pygame
from numba import jit
from numba import cuda

import numpy as np
from time import time
import math

from settings import *
from helpers import *

@cuda.jit("void(intc[:], short, short, double, double, double, intc)")
def mandlebrot( pixels, width, height, C_x, C_y, zoom, maxDepth ):
    i = cuda.grid(1)
    if i < width*height :
        x = ( i / width - width/2 ) / zoom + C_x
        y = ( i % height - height/2 ) / zoom + C_y

        z = 0 + 0j
        c = x + y*1j
        depth = 0

        while True:
            if z.real**2 + z.imag**2 > 4:
                break
            elif depth > maxDepth:
                depth = -1
                break

            z = z*z*z*z + c
            depth += 1

        pixels[i] = depth

# Function to call the cuda kernal corresponding to
#   each pixel in order to get there respective mandlebrot depth
def getPixels(rawPixels, grid, block, position, zoom, accuracyBase): # TODO: Make more efficient
    mandlebrot[grid, block]( rawPixels, WIDTH, HEIGHT, position[0], position[1], zoom, int(20 * math.log(zoom, accuracyBase) ) )

def handleEvents(position, zoom, accuracyBase):
    res = [position, zoom, accuracyBase, False]
    # ======== Handle events ========
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # If click even was pressed
            res[3] = True # Set quit flag to true

    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_EQUALS]:
        res[1] *= 1.05

    if pressed[pygame.K_MINUS]:
        res[1] /= 1.05

    if pressed[pygame.K_d]:
        res[0][0] += 10/zoom

    if pressed[pygame.K_a]:
        res[0][0] -= 10/zoom

    if pressed[pygame.K_w]:
        res[0][1] -= 10/zoom

    if pressed[pygame.K_s]:
        res[0][1] += 10/zoom

    if pressed[pygame.K_UP]:
        res[2] -= 0.1

    if pressed[pygame.K_DOWN]:
        res[2] += 0.1



    res[2] = max(1.1, res[2])
    print(res[2])
    return res


# Main function
def __main__():
    position = [0, 0] # Position Of The Cammera
    zoom = 500 # The Zoom Factor
    accuracyBase = 2.7

    # Make sure the screem dimensions are valid
    if WIDTH.astype(int)*HEIGHT.astype(int) < 1024:
        print("To few pixels! :-(")
        exit()

    # Get the block and grid dimensions
    block = ceil(int(WIDTH)*int(HEIGHT)/1024)
    grid = 1024

    # Setup numpy arrays for the raw and colored pixel data
    rawPixels = np.array([0 for _ in range(WIDTH.astype(int)*HEIGHT.astype(int))], dtype=np.intc) # Will contain the mandlebrot depth of each pixel

    # Init pygame
    pygame.init()

    # Create a screen to draw to
    screen = pygame.display.set_mode([WIDTH, HEIGHT])

    quit = False # Flag to tell whether to quit out or not
    frame = 0 # The current frame count
    lastFrameRate = 60

    # Main function loop
    while not quit:
        start = time() # Frame start time

        # Handle quit, movement, and zoom events
        position, zoom, accuracyBase, quit = handleEvents(position, zoom, accuracyBase)

        # Set the new pixel values
        getPixels(rawPixels, grid, block, position, zoom, accuracyBase)

        # Update the screen
        pygame.pixelcopy.array_to_surface(screen, rawPixels.reshape(WIDTH, HEIGHT))
        pygame.display.flip()
        frame += 1

        end = time()

        lastFrameRate = 1/(end - start)

        if frame % 50 == 0 :
            print("Frame:", frame, "| FPS:", lastFrameRate)

__main__()
