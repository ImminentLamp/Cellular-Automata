
import pyglet
from neural import Neural
from tqdm.auto import tqdm


class Window(pyglet.window.Window):

    def __init__(self):
        # set window size
        super().__init__(900, 900, 'Neural Cellular Automata')

        # pass in window size (x and y) and set cell size (must be a divisor of the window size)
        self.neural = Neural(self.get_size()[0],
                             self.get_size()[1],
                             5)
        self.i = 0
        pyglet.clock.schedule(self.update)

    def on_draw(self):
        # generation and fps labels
        label = pyglet.text.Label('Gen: ' + str(self.i), 'Calibri', 20, x=10, y=10, color=(0, 255, 0, 255), bold=True)
        fps = pyglet.text.Label('FPS: ' + str(pyglet.clock.get_fps()), 'Calibri', 20, x=self.get_size()[0] - 115,
                                y=self.get_size()[1] - 30, color=(0, 255, 0, 255), bold=True)

        # clear screen and draw updated grid
        self.clear()
        self.neural.draw()
        label.draw()
        fps.draw()

    def update(self, dt):
        """compute next generation of cells"""
        # set how many generations to compute at once by changing range
        # can get rid of tqdm, was just using it to track how long the iterations take
        for _ in tqdm(range(10), position=0, leave=True):
            # for x in range(2):
            self.i += 1
            self.neural.run_rules()


if __name__ == '__main__':
    window = Window()
    pyglet.app.run()










