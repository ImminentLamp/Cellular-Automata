
import pyglet
import random as rnd
import numpy as np
from scipy.fftpack import fft2, ifft2


class Neural:

    def __init__(self, window_width, window_height, cell_size):

        self.grid_width = int(window_width/cell_size)
        self.grid_height = int(window_height/cell_size)
        self.cell_size = cell_size
        self.cells = np.zeros((self.grid_width, self.grid_height))
        self.vertices = []
        self.generate_cells()

        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
        pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)

    def generate_cells(self):
        """ fill cells array with random floats between 0 and 1"""
        for row in range(0, self.grid_height):
            for col in range(0, self.grid_width):
                self.cells[row, col] = rnd.random()

    def run_rules(self):
        """compute next generation"""

        # convolution filter
        cfilter = np.array([[0.8, -0.85, 0.8],
                           [-0.85, -0.2, -0.85],
                           [0.8, -0.85, 0.8]])

        # convolute cells array with filter
        convolved = self.convolve(self.cells, cfilter)

        # pass all values through activation function
        activated = self.activation(convolved)

        # clip values between 0 and 1
        self.cells = np.clip(activated, 0, 1)

    @staticmethod
    def convolve(cells, cfilter):
        """perform matrix convolution with fourier transforms"""

        f = ifft2(fft2(cells, shape=cells.shape) * fft2(cfilter, shape=cells.shape)).real
        f = np.roll(f, (-((cfilter.shape[0] - 1) // 2), -((cfilter.shape[1] - 1) // 2)), axis=(0, 1))

        return f

    @staticmethod
    def activation(x):
        """activation function"""

        x = -1./(0.89*pow(x, 2.)+1.)+1.
        return x

    def draw(self):
        """draw grid of cells"""

        batch = pyglet.graphics.Batch()
        for row in range(0, self.grid_height):
            for col in range(0, self.grid_width):
                if self.cells[row, col] > 0:
                    # specify vertices of each cell to be drawn (any cell with value > 0)
                    vertex = [row * self.cell_size,                  col * self.cell_size,
                              row * self.cell_size,                  col * self.cell_size + self.cell_size,
                              row * self.cell_size + self.cell_size, col * self.cell_size + self.cell_size,
                              row * self.cell_size + self.cell_size, col * self.cell_size]
                    # add cell to batch with some colour (rgb) and alpha proportional to the cell value
                    batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', vertex), ('c4B', (255, 0, 0, int(255*self.cells[row, col]),
                                                                                     255, 0, 0, int(255*self.cells[row, col]),
                                                                                     255, 0, 0, int(255*self.cells[row, col]),
                                                                                     255, 0, 0, int(255*self.cells[row, col]))))

        pyglet.gl.glClear(pyglet.gl.GL_COLOR_BUFFER_BIT)
        # draw entire batch
        batch.draw()


