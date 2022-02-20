import random
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
from kivy.properties import ListProperty, StringProperty
from kivy.core.window import Window
from kivy.clock import Clock
from colors import allcolors

widgets = ['Image','TextInput','Label','Button','CheckBox','Slider','Switch','Spinner','ProgressBar','TransLabel','FlatButton']


class Widgets:
    @staticmethod
    def get_random_widget():
        ret = random.choice(widgets)
        if ret=='Image': return Image(source='data/icons/bug1.png')
        elif ret=='TextInput': return TextInput(text='textinput') #(font_size = 20, size_hint_y = None, height = 50)
        elif ret=='Label': return Label(text='label', color=random.choice(allcolors))
        elif ret=='Button': return Button(text='button', background_color=random.choice(allcolors)) #on_press=self.on_btn_click))
        elif ret=='CheckBox': return CheckBox(active=True)
        elif ret=='Slider': return Slider(min=1, max=10, value=1, step=1)
        elif ret=='Switch': return Switch(active=True)
        elif ret=='Spinner': return Spinner(text="Spinner", values=("Python", "Java", "C++", "C", "C#", "PHP"), background_color=(0.784,0.443,0.216,1))
        elif ret=='ProgressBar': return ProgressBar(max=1000, value=750)
        elif ret=='TransLabel': return TransLabel()
        elif ret=='FlatButton': return FlatButton().build()
        return


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


class TransLabel(Label):
    velocity = ListProperty([2, 1])

    def __init__(self, **kwargs):
        super(TransLabel, self).__init__(**kwargs)
        Clock.schedule_interval(self.updatePOS, 1. / 60.)
        self.velocity[0] *= random.choice([-1, 1])
        self.velocity[1] *= random.choice([-1, 1])

        self.text = 'translabel'
        self.color = random.choice(allcolors)

    def updatePOS(self, *args):
        self.x += self.velocity[0]
        self.y += self.velocity[1]

        if self.x < 0 or (self.x + self.width) > Window.width:
            self.velocity[0] *= -1

        if self.y < 0 or (self.y + self.height) > Window.height:
            self.velocity[1] *= -1

        self.pos = [self.x, self.y]