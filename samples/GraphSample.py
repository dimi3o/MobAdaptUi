"""Real time plotting using kivy"""

from math import sin
from kivy.app import App
from kivy_garden.graph import Graph, MeshLinePlot
from kivy.core.window import Window
from kivy.clock import Clock

class GraphApp(App):
    def build(self):
        self.graph = Graph(xlabel='X', ylabel='Y', x_ticks_minor=5,
                           x_ticks_major=25, y_ticks_major=1,
                           y_grid_label=True, x_grid_label=True, padding=5,
                           x_grid=True, y_grid=True, xmin=-0, xmax=1, ymin=-1, ymax=1)
        self.plot = MeshLinePlot(color=[1, 0, 0, 1])
        Clock.schedule_interval(self.get_value, 0.01)
        Clock.schedule_interval(self.plot_value, 1)
        self.graph.add_plot(self.plot)
        self.x_axis = 0
        self.step = Window.width / 1000.
        self.points = []
        return self.graph

    def get_value(self, dt):
        self.x_axis += .1
        self.points.append((self.x_axis, sin(self.x_axis / 10.)))
        self.graph.xmax += .1

    def plot_value(self, dt):
        self.plot.points = [(x, y) for x, y in self.points]


if __name__ == "__main__":
    GraphApp().run()

