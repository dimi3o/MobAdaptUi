import random
import math as m
from agent import plot
from kivy.metrics import dp
from kivy.lang import Builder
from kivy.uix.widget import Widget
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
from kivy.properties import ListProperty, BooleanProperty, NumericProperty, StringProperty
from kivy.core.window import Window
from kivy.clock import Clock
from kivy_garden.graph import Graph, LinePlot
from colors import allcolors

widgets = ['Image','TextInput','Label','Button','CheckBox','Slider','Switch','Spinner','ProgressBar','FlyLabel','FlatButton']

class Widgets(object):
    @staticmethod
    def get_random_widget(name='', params=[]):
        ret = random.choice(widgets) if (name == '') else name
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
        elif ret == 'LineRectangle': return LineRectangle(line_width=2, line_rect=[params[0], params[1], params[2], params[3]], line_color=[1,0,0,1], label_text=params[4], label_pos=[params[2]//3, 0])
        return ''

    @staticmethod
    def get_app_icon(id=''):
        id = random.randint(1,41) if (id == '') else id
        return Image(source="data/icons/apps/a"+str(id)+".png", allow_stretch=True, keep_ratio=True)

    @staticmethod
    def get_food_icon(id=''):
        id = random.randint(1, 41) if (id == '') else id
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

    @staticmethod
    def target_ui(name='rus'):
        if name == "rus":
            return [[0.02, 0.31, 1.07, 0.12], [0.82, 0.66, 1.07, -0.07], [0.28, 0.27, 1.33, 0.03], [0.46, 0.80, 1.08, -0.07], [0.82, 0.37, 1.01, 0.00], [0.01, 0.00, 2.21, 0.48], [0.02, 0.22, 1.00, -0.28], [0.28, 0.19, 1.10, -0.05], [0.59, 0.81, 1.02, -0.05], [0.50, 0.18, 1.52, -0.09], [0.33, 0.66, 0.94, 0.05], [0.30, 0.84, 1.03, -0.16], [0.35, 0.75, 1.02, 0.14], [0.61, 0.70, 1.05, 0.16], [0.76, 0.78, 1.05, -0.02], [0.05, 0.42, 1.11, -0.02], [0.19, 0.23, 1.01, -0.14], [0.12, 0.19, 0.97, 0.00], [0.42, 0.66, 0.98, 0.29], [0.31, 0.00, 1.99, 0.72], [0.08, 0.66, 1.09, 0.05], [0.19, 0.37, 1.00, 0.16], [0.46, 0.31, 1.56, -0.34], [0.82, 0.82, 0.93, 0.05], [0.53, 0.71, 1.00, -0.17], [0.55, 0.83, 0.87, 0.02], [0.10, 0.71, 0.86, 0.10], [0.39, 0.79, 1.10, 0.22], [0.18, 0.67, 1.05, -0.33], [0.67, 0.00, 1.92, 0.40], [0.79, 0.66, 0.97, -0.10], [0.62, 0.67, 1.00, -0.21], [0.65, 0.24, 1.08, -0.21], [0.78, 0.21, 1.43, 0.24], [0.22, 0.78, 1.08, -0.17], [0.39, 0.42, 1.03, 0.05], [0.22, 0.75, 1.09, -0.05], [0.55, 0.66, 0.96, -0.14], [0.07, 0.82, 1.06, 0.09], [0.04, 0.75, 0.88, 0.28]]
        elif name == 'eur':
            return [[0.72, 0.19, 1.17, -0.06], [0.05, 0.76, 0.99, -0.00], [0.22, 0.77, 0.99, 0.02], [0.90, 0.85, 1.00, 0.02], [0.16, 0.32, 1.31, -0.03], [0.16, 0.27, 0.99, -0.02], [0.61, 0.05, 0.98, -0.02], [0.40, 0.80, 1.20, -0.04], [0.87, 0.87, 1.00, -0.02], [-0.01, 0.33, 1.16, -0.06], [0.89, 0.91, 1.01, -0.00], [0.89, 0.87, 0.99, 0.02], [0.77, 0.56, 1.27, -0.00], [0.43, 0.61, 1.25, 0.18], [0.16, 0.40, 1.29, -0.03], [0.79, 0.11, 1.00, -0.00], [0.86, 0.90, 1.00, 0.02], [0.00, 0.41, 0.99, -0.00], [0.25, 0.06, 1.00, -0.00], [0.07, 0.82, 1.35, -0.04], [0.92, 0.86, 1.00, -0.00], [0.79, 0.04, 0.98, 0.05], [0.32, 0.39, 1.57, -0.04], [0.63, 0.57, 1.00, -0.00], [0.61, 0.14, 1.00, -0.02], [0.90, 0.89, 1.00, 0.02], [0.65, 0.64, 1.32, 0.11], [0.08, 0.12, 1.00, -0.00], [0.06, 0.44, 1.35, 0.05], [0.04, 0.03, 1.42, 0.06], [0.91, 0.86, 1.00, -0.00], [0.75, 0.50, 1.00, -0.02], [0.20, 0.72, 0.99, 0.02], [0.26, 0.84, 1.01, -0.02], [0.91, 0.88, 1.00, 0.02], [0.92, 0.90, 1.00, -0.02], [0.45, 0.54, 1.33, 0.03], [0.92, 0.88, 0.97, -0.00], [0.90, 0.89, 0.99, -0.05], [0.37, 0.32, 1.01, 0.02]]
        elif name == 'asia':
            return [[0.67, 0.07, 1.00, -0.00], [0.22, 0.10, 1.31, 0.04], [0.73, 0.10, 1.00, -0.00], [0.59, 0.04, 1.00, -0.00], [0.24, 0.74, 1.33, -0.16], [0.56, 0.08, 1.00, -0.00], [0.75, 0.12, 1.00, -0.00], [0.70, 0.08, 1.00, -0.00], [0.68, 0.08, 1.00, -0.00], [0.65, 0.09, 1.00, -0.00], [0.63, 0.65, 2.11, 0.07], [0.22, 0.33, 1.00, -0.00], [0.73, 0.08, 1.00, -0.00], [0.69, 0.03, 1.00, -0.00], [0.20, 0.18, 1.00, -0.00], [0.08, 0.77, 1.17, -0.04], [0.68, 0.06, 1.00, -0.00], [0.22, 0.82, 1.65, 0.07], [0.60, 0.09, 1.00, -0.00], [0.66, 0.07, 1.00, -0.00], [0.64, 0.04, 1.00, -0.00], [0.09, 0.85, 1.00, -0.00], [0.07, 0.33, 1.00, -0.00], [0.75, 0.10, 1.00, -0.00], [0.73, 0.05, 1.00, -0.00], [0.80, 0.07, 1.00, -0.00], [-0.01, 0.13, 1.26, 0.11], [0.34, 0.17, 1.32, -0.02], [0.20, 0.24, 1.00, -0.00], [0.74, 0.09, 1.00, -0.00], [0.62, 0.46, 2.07, -0.09], [0.70, 0.11, 1.00, -0.00], [0.63, 0.10, 1.00, -0.00], [-0.02, 0.23, 1.29, 0.01], [0.58, 0.53, 2.70, -0.11], [0.67, 0.06, 1.00, -0.00], [0.33, 0.28, 1.20, 0.08], [0.02, 0.67, 1.49, -0.08], [0.39, 0.23, 1.00, -0.00], [0.44, 0.26, 3.25, 0.28]]

class FlyScatterV3(Scatter):#(TouchRippleBehavior, Scatter):
    velocity = ListProperty([2, 1])
    emulation = BooleanProperty(False)
    mode = 'Fly adapt'
    app = None
    strategy = None
    memory = None
    agent = None
    policy_net = None
    target_net = None
    optimizer = None
    env = None
    id = 1
    grid_rect = None
    raw_height = 0
    raw_width = 0
    raw_rotate = 0
    reduceW = -1
    reduceH = -1
    deltaposxy = 1
    doublesize = BooleanProperty(False)
    nx, ny, ns, nr = 0, 0, 0, 0
    taps = 0
    steps_learning = 0
    vect_state = []

    def __init__(self, **kwargs):
        super(FlyScatterV3, self).__init__(**kwargs)
        self.velocity[0] *= random.choice([-2, 2])
        self.velocity[1] *= random.choice([-2, 2])
        self.text = 'flyscatter'
        self.color = random.choice(allcolors)
        if self.env: self.set_vect_state()

    def tap_event(self):
        self.taps += 1
        self.children[0].text = f'taps: {self.taps}'
        self.calc_norm_values()
        r = self.MARL_core()
        print(r)

    def toFixed(self,numObj, digits=0): return f"{numObj:.{digits}f}"

    def update_pos(self, *args):
        if self.mode == 'MARL adapt':
            r = self.MARL_core()
            #self.change_pos_size(a)
            #self.app.total_reward += self.app.y_discount**self.app.rewards_count * r # DISCOUNTED REWARD 2
            #self.app.total_reward = r # SINGLE AGENT REWARD
            #self.app.total_reward += r  # SUMM REWARD
            self.app.rewards_count += 1
            self.app.reward_data[int(self.id)-1] = self.agent.reward_data[-1]
            self.app.cumulative_reward_data[int(self.id) - 1] += self.agent.reward_data[-1]
            self.app.loss_data[int(self.id)-1] = self.agent.loss_data[-1]
            if self.env.is_done():
                self.emulation = self.set_emulation(False)
                self.app.stop_emulation_async('MARL adapt is stopped. End of episode!', 'Adapt',
                                              self.agent.total_reward)
                #print(self.id, self.taps, self.nx, self.ny, self.ns, self.nr)
            self.children[0].text = f'{self.nx}, {self.ny}'
        elif self.mode ==  'Rotate adapt' or self.mode == 'Fly+Size+Rotate adapt': self.rotation += random.choice([-1, 1])
        elif self.mode == 'Fly adapt' or self.mode == 'Fly+Size+Rotate adapt':
            self.x += self.deltaposxy*self.velocity[0]
            self.y += self.deltaposxy*self.velocity[1]
            if self.x < 0 or (self.x + 2*self.width//3) > Window.width:
                self.velocity[0] *= -1
            if self.y < 0 or (self.y + 2*self.height//3) > Window.height:
                self.velocity[1] *= -1
        elif self.mode == 'Size adapt' or self.mode == 'Fly+Size+Rotate adapt':
            w = self.children[1].width
            h = self.children[1].height
            if w < self.raw_width // 3: self.reduceW = 1
            elif w > self.raw_width: self.reduceW = -1
            if h < self.raw_height // 3: self.reduceH = 1
            elif h > self.raw_height: self.reduceH = -1
            self.children[1].width = w + self.reduceW
            self.children[1].height = h + self.reduceH

    def MARL_core(self):
        a, r = self.agent.step(self.env)

        # target_net_state_dict = self.target_net.state_dict()
        # policy_net_state_dict = self.policy_net.state_dict()
        # for key in policy_net_state_dict:
        #     target_net_state_dict[key] = policy_net_state_dict[key] * self.app.TAU + target_net_state_dict[key] * (1 - self.app.TAU)
        # self.target_net.load_state_dict(target_net_state_dict)

        if self.env.steps_left % self.app.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # print('--- target_net update ---')
        return r

    def set_vect_state(self):
        self.calc_norm_values()
        self.vect_state = [int(self.id), int(self.taps), float(self.nx), float(self.ny), float(self.ns), float(self.nr)]
        self.env.vect_state = self.vect_state

    def calc_norm_values(self):
        self.ns = self.toFixed(self.scale, 2)
        self.nx, self.ny = self.toFixed(self.x / Window.width, 2), self.toFixed(self.y / Window.height, 2)
        self.nr = self.toFixed(-m.sin(self.rotation / 180 * m.pi), 2)

    # 0 - left, 1 - right, 2 - up, 3 - down, 4 - more, 5 - less
    def change_pos_size(self, to=0, deltapos=1, deltascale=0.01):
        r = -1
        if to==0 and self.x>0: self.x -= deltapos; r = .01
        elif to==1 and self.x+self.width<Window.width: self.x += deltapos; r = .01
        elif to==2 and self.y+self.height<Window.height: self.y += deltapos; r = .01
        elif to==3 and self.y>0: self.y -= deltapos; r = .01
        elif to==4 and self.scale<2.: self.scale += deltascale; r = .01
        elif to==5 and self.scale>0.4: self.scale -= deltascale; r = .01
        elif to==6: self.rotation -= 1; r = .01 # and (self.rotation>265 or self.rotation==0.)
        elif to==7: self.rotation += 1; r = .01 # self.rotation<110
        self.set_vect_state()
        return r


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

    def on_touch_up(self, touch):
        if touch.grab_current == self: self.tap_event()
        #print(self.id, self.taps, self.nx, self.ny, self.ns, self.nr)
        self.app.do_current_ui_vect([self.id, self.taps, self.nx, self.ny, self.ns, self.nr])
        super(FlyScatterV3, self).on_touch_up(touch)


    # def on_touch_down(self, touch):
    #     if touch.grab_current == self: self.tap_event()
    #     super(FlyScatterV3, self).on_touch_down(touch)

    # def on_touch_move(self, touch):
        # if not self.doublesize:
        #     self.children[0].width *= 2
        #     self.children[0].height *= 2
        #     self.doublesize = True
        # else:
        #     self.children[0].width //= 2
        #     self.children[0].height //= 2
        #     self.doublesize = False

class AsyncConsoleScatter(Scatter):
    console = None

    def __init__(self, **kwargs):
        super(AsyncConsoleScatter, self).__init__(**kwargs)

    def start_emulation(self, console_widget):
        self.console = console_widget
        Clock.schedule_interval(self.update_console, 1. / 2.)

    def update_console(self, *args):
        self.console.text += '.'

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

# Builder.load_string('''
# <LineRectangle>:
#     canvas:
#         Color:
#             rgba: .1, .1, 1, .9
#         Line:
#             width: 2.
#             rectangle: (self.x, self.y, self.width, self.height)
#     Label:
#         center: root.center
#         text: 'Rectangle'
# ''')
#
# class LineRectangle(Widget):
#     pass

Builder.load_string('''
<LineRectangle>:
    canvas:
        Color:
            rgba: root.line_color
        Line:
            width: root.line_width
            rectangle: root.line_rect
    Label:
        text: root.label_text
        pos: root.label_pos
        color: root.line_color
''')

class LineRectangle(Widget):
    line_color = ListProperty([])
    line_rect = ListProperty([])
    line_width = NumericProperty()
    label_text = StringProperty('')
    label_pos = ListProperty([])

    def __init__(self, **kwargs):
        self.line_color = kwargs.pop('line_color', [.1, .1, 1, .9])
        self.line_rect = kwargs.pop('line_rect', [0, 0, 50, 50])
        self.line_width = kwargs.pop('line_width', 1)
        self.label_text = kwargs.pop('label_text', 'Rectangle')
        self.label_pos = kwargs.pop('label_pos', [0, 0])
        super(LineRectangle, self).__init__()
#
# self.bbox1 = LineRectangle(line_wdth=2, line_rect=[100, 100, 100, 100], line_color=[1,0,0,1], label_text='bbox1', label_pos=[100, 100])

