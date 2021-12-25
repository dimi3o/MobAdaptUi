"""Real time plotting of Microphone level using kivy
"""
import random
from math import sin
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import Graph, MeshLinePlot
from kivy.core.window import Window
from kivy.clock import Clock

class Logic(BoxLayout):

    def __init__(self, **kwargs):
        super(Logic, self).__init__(**kwargs)
        self.plot = MeshLinePlot(color=[1, 0, 0, 1])
        self.x_axis = 0
        self.step = Window.width / 1000.

    def start(self):
        self.ids.graph.add_plot(self.plot)
        Clock.schedule_interval(self.get_value, 0.001)

    def stop(self):
        Clock.unschedule(self.get_value)

    def get_value(self, dt):
        #self.plot.points = [(i, j/5) for i, j in enumerate(levels)]
        self.plot.points.append((self.x_axis,random.randrange(0, 19, 1)/.2)) #[random.randrange(0, 19, 1)]
        # self.plot.points = [(x, 0.5) for x in range(-10, 10)]
        if self.x_axis - Window.width < 0:
            self.x_axis += self.step
        else:
            self.x_axis = 0
            self.ids.graph.remove_plot(self.plot)
            self.plot = MeshLinePlot(color=[1, 0, 0, 1])
            self.ids.graph.add_plot(self.plot)

class RealTimeMicrophone(App):
    # def build(self):
    #     return Builder.load_file("look.kv")

    def build(self):
        'create graph'
        # self.canvas.add(Color(1., 1., 0))
        self.graph = Graph(xlabel='X', ylabel='Y', x_ticks_minor=5,
                      x_ticks_major=25, y_ticks_major=1,
                      y_grid_label=True, x_grid_label=True, padding=5,
                      x_grid=True, y_grid=True, xmin=-0, xmax=1, ymin=-1, ymax=1)
        self.plot = MeshLinePlot(color=[1, 0, 0, 1])
        Clock.schedule_interval(self.get_value, 0.01)
        Clock.schedule_interval(self.plot_value, 1)
        #self.plot.points = [(x, sin(x / 10.)) for x in range(0, 101)]
        self.graph.add_plot(self.plot)
        self.x_axis = 0
        self.step = Window.width / 1000.
        self.points = []
        return self.graph

    def get_value(self, dt):
        self.x_axis += .1
        self.points.append((self.x_axis,sin(self.x_axis / 10.)))
        self.graph.xmax += .1

    def plot_value(self, dt):
        self.plot.points = [(x, y) for x, y in self.points]

if __name__ == "__main__":
    RealTimeMicrophone().run()
    
