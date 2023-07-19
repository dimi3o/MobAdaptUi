import random
import math as m
from kivy.metrics import dp
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider
from kivy.uix.switch import Switch
from kivy.uix.spinner import Spinner
from kivy.uix.progressbar import ProgressBar
from kivy.uix.behaviors import TouchRippleBehavior
from kivy.uix.scatter import Scatter
from kivy.properties import ListProperty, BooleanProperty
from kivy.core.window import Window
from kivy.clock import Clock
from kivy_garden.graph import Graph, LinePlot
from colors import allcolors

widgets = ['Image','TextInput','Label','Button','CheckBox','Slider','Switch','Spinner','ProgressBar','FlyLabel','FlatButton']

class Widgets(object):
    @staticmethod
    def get_random_widget(name=''):
        ret = random.choice(widgets) if (name is '') else name
        if ret == 'Image': return Image(source='data/icons/bug1.png', allow_stretch=True, keep_ratio=True)
        elif ret == 'TextInput': return TextInput(text='textinput')
        elif ret == 'Label': return Label(text='label', color=random.choice(allcolors))
        elif ret == 'Button': return Button(text='button', background_color=random.choice(allcolors))
        elif ret == 'CheckBox': return CheckBox(active=True)
        elif ret == 'Slider': return Slider(min=1, max=10, value=1, step=1)
        elif ret == 'Switch': return Switch(active=True)
        elif ret == 'Spinner': return Spinner(text="Spinner", values=("Python", "Java", "C++", "C", "C#", "PHP"), background_color=(0.784,0.443,0.216,1))
        elif ret == 'ProgressBar': return ProgressBar(max=1000, value=750)
        elif ret == 'FlyLabel': return FlyLabel()
        elif ret == 'FlatButton': return FlatButton().build()
        return ''

    @staticmethod
    def get_app_icon(id=''):
        id = random.randint(1,41) if (id is '') else id
        return Image(source="data/icons/apps/a"+str(id)+".png", allow_stretch=True, keep_ratio=True)

    @staticmethod
    def get_food_icon(id=''):
        id = random.randint(1, 41) if (id is '') else id
        return Image(source="data/icons/foods/f" + str(id) + ".png", allow_stretch=True, keep_ratio=True)

    @staticmethod
    def get_widget(name):
        return Widgets.get_random_widget(name)

    @staticmethod
    def get_graph_widget(x_ticks_major=5, y_ticks_major=5, xmin=0, xmax=1, ymin=0, ymax=30, xlabel='Time',
                         WhiteBackColor=True):
        graph_theme = {
            'label_options': {
                'color': [0, 0, 0, 1],  # color of tick labels and titles
                'bold': False},
            'background_color': [1, 1, 1, 1],  # back ground color of canvas
            'tick_color': [0, 0, 0, 1],  # ticks and grid
            'border_color': [0, 0, 0, 1]}  # border drawn around each graph
        if WhiteBackColor:
            graph = Graph(**graph_theme)
        else:
            graph = Graph()
        if xlabel is not None: graph.xlabel = xlabel
        graph.x_ticks_minor = 0
        graph.x_ticks_major = x_ticks_major; graph.y_ticks_major = y_ticks_major
        graph.y_grid_label = True; graph.x_grid_label = True
        graph.padding = 5
        graph.x_grid = True; graph.y_grid = True
        graph.xmin = xmin; graph.xmax = xmax; graph.ymin = ymin; graph.ymax = ymax
        graph.tick_color = [0, 0, 0, 1]
        return graph

class FlyScatterV3(Scatter):#(TouchRippleBehavior, Scatter):
    velocity = ListProperty([2, 1])
    emulation = BooleanProperty(False)
    mode = 'Fly adapt'
    app = None
    agent = None
    env = None
    id = None
    raw_height = 0
    raw_width = 0
    raw_rotate = 0
    reduceW = -1
    reduceH = -1
    deltaposxy = 1
    doublesize = BooleanProperty(False)
    nx, ny, ns, nr = 0, 0, 0, 0
    taps = 0

    def __init__(self, **kwargs):
        super(FlyScatterV3, self).__init__(**kwargs)
        self.velocity[0] *= random.choice([-2, 2])
        self.velocity[1] *= random.choice([-2, 2])
        self.text = 'flyscatter'
        self.color = random.choice(allcolors)

    def on_touch_up(self, touch):
        if touch.grab_current == self:
            self.taps += 1
            self.children[0].text = f'taps: {self.taps}'
            self.calc_norm_values()
            print(self.id, self.taps, self.nx, self.ny, self.ns, self.nr)

    def toFixed(self,numObj, digits=0): return f"{numObj:.{digits}f}"

    def update_pos(self, *args):
        if self.mode ==  'Rotate adapt' or self.mode == 'Fly+Size+Rotate adapt':
            self.rotation += random.choice([-1, 1])
        if self.mode == 'Fly adapt' or self.mode == 'Fly+Size+Rotate adapt':
            self.x += self.deltaposxy*self.velocity[0]
            self.y += self.deltaposxy*self.velocity[1]
            if self.x < 0 or (self.x + 2*self.width//3) > Window.width:
                self.velocity[0] *= -1
            if self.y < 0 or (self.y + 2*self.height//3) > Window.height:
                self.velocity[1] *= -1
        if self.mode == 'Size adapt' or self.mode == 'Fly+Size+Rotate adapt':
            w = self.children[1].width
            h = self.children[1].height
            if w < self.raw_width // 3: self.reduceW = 1
            elif w > self.raw_width: self.reduceW = -1
            if h < self.raw_height // 3: self.reduceH = 1
            elif h > self.raw_height: self.reduceH = -1
            self.children[1].width = w + self.reduceW
            self.children[1].height = h + self.reduceH
        if self.mode == 'MARL adapt':
            e = self.env
            a, r = self.agent.step(e)
            self.change_pos_size(a)
            self.app.total_reward += r
            if e.is_done():
                self.calc_norm_values()
                print(self.id, self.taps, self.nx, self.ny, self.ns, self.nr)
                self.emulation = self.set_emulation(False)
                self.app.stop_emulation_async('MARL adapt is stopped. End of episode!', 'Adapt', self.agent.total_reward)
        self.calc_norm_values()
        self.children[0].text = f'{self.nx}, {self.ny}'

    def calc_norm_values(self):
        self.ns = self.toFixed(self.scale, 2)
        self.nx, self.ny = self.toFixed(self.x / Window.width, 2), self.toFixed(self.y / Window.height, 2)
        self.nr = self.toFixed(-m.sin(self.rotation / 180 * m.pi), 2)

    # 0 - left, 1 - right, 2 - up, 3 - down, 4 - more, 5 - less
    def change_pos_size(self, to=0, deltapos=1, deltascale=0.01):
        if to==0 and self.x>0: self.x -= deltapos
        elif to==1 and self.x+self.width<Window.width: self.x += deltapos
        elif to==2 and self.y+self.height<Window.height: self.y += deltapos
        elif to==3 and self.y>0: self.y -= deltapos
        elif to==4:
            self.scale += deltascale
            # if self.children[1].height < 3*self.raw_height:
            #     self.children[1].width += delta
            #     self.children[1].height += delta
        elif to==5:
            self.scale -= deltascale
            # if self.children[1].height > self.raw_height//2:
            #     self.children[1].width -= delta
            #     self.children[1].height -= delta
        elif to==6: self.rotation -= 1
        elif to==7: self.rotation += 1

    def change_emulation(self):
        self.emulation = self.set_emulation(True) if not(self.emulation) else self.set_emulation(False)

    def start_emulation(self):
        self.emulation = self.set_emulation(True)

    def stop_emulation(self):
        self.emulation = self.set_emulation(False)

    def set_emulation(self, mode=False):
        if mode:
            Clock.schedule_interval(self.update_pos, 1. / 60.)
            return True
        else:
            Clock.unschedule(self.update_pos)
            return False

    # def on_touch_move(self, touch):
        # if not self.doublesize:
        #     self.children[0].width *= 2
        #     self.children[0].height *= 2
        #     self.doublesize = True
        # else:
        #     self.children[0].width //= 2
        #     self.children[0].height //= 2
        #     self.doublesize = False

KV = """
<FlatButton>:
    ripple_color: 0, 0, 0, 0
    background_color: 0, 0, 0, 0
    color: root.primary_color

    canvas.before:
        Color:
            rgba: root.primary_color
        Line:
            width: 1
            rectangle: (self.x, self.y, self.width, self.height)

Screen:
    canvas:
        Color:
            rgba: 0.9764705882352941, 0.9764705882352941, 0.9764705882352941, 1
        # Rectangle:
        #     pos: (self.x, self.y)
        #     size: self.size
"""
class FlatButton(TouchRippleBehavior, Button):
    primary_color = [0.12941176470588237, 0.5882352941176471, 0.9529411764705882, 1]

    def build(self):
        screen = Builder.load_string(KV)
        btn = FlatButton(text="flatbutton", ripple_color=(0, 0, 0, 0))
        screen.add_widget(btn)
        return screen

    def on_touch_down(self, touch):
        collide_point = self.collide_point(touch.x, touch.y)
        if collide_point:
            touch.grab(self)
            self.ripple_show(touch)
            return True
        return False

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            self.ripple_fade()
            return True
        return False


class FlyLabel(TouchRippleBehavior, Label):
    velocity = ListProperty([2, 1])
    emulation = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(FlyLabel, self).__init__(**kwargs)
        self.velocity[0] *= random.choice([-2, 2])
        self.velocity[1] *= random.choice([-2, 2])
        self.text = 'flylabel'
        self.color = random.choice(allcolors)

    def update_pos(self, *args):
        parent = self.parent
        parent.x += self.velocity[0]
        parent.y += self.velocity[1]
        if parent.x < 0 or (parent.x + 2*parent.width//2) > Window.width:
            self.velocity[0] *= -1
        if parent.y < 0 or (parent.y + 2*parent.height//2) > Window.height:
            self.velocity[1] *= -1
        parent.pos = [parent.x, parent.y]

    def on_touch_down(self, touch):
        if self.emulation:
            self.emulation = False
            Clock.unschedule(self.update_pos)
            return
        self.emulation = True
        Clock.schedule_interval(self.update_pos, 1. / 60.)

class FlyScatter(TouchRippleBehavior, Scatter):
    velocity = ListProperty([2, 1])
    emulation = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(FlyScatter, self).__init__(**kwargs)
        self.velocity[0] *= random.choice([-2, 2])
        self.velocity[1] *= random.choice([-2, 2])
        self.text = 'flylabel'
        self.color = random.choice(allcolors)

    def update_pos(self, *args):
        parent = self
        parent.x += self.velocity[0]
        parent.y += self.velocity[1]
        if parent.x < 0 or (parent.x + 2*parent.width//3) > Window.width:
            self.velocity[0] *= -1
        if parent.y < 0 or (parent.y + 2*parent.height//3) > Window.height:
            self.velocity[1] *= -1
        parent.pos = [parent.x, parent.y]

    def on_touch_down(self, touch):
        if not(self.emulation):
            Clock.schedule_interval(self.update_pos, 1. / 60.)
            self.emulation = True
        else:
            Clock.unschedule(self.update_pos)
            self.emulation = False
