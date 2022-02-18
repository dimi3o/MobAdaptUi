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
from colors import allcolors


class Widgets:
    @staticmethod
    def get_random_widget():
        widgets = [Image(source='data/icons/bug1.png'),
                   TextInput(text='textinput'),#(font_size = 20, size_hint_y = None, height = 50)
                   Label(text='label', color=random.choice(allcolors)),
                   Button(text='button', background_color=random.choice(allcolors)), #on_press=self.on_btn_click))
                   CheckBox(active=True),
                   Slider(min=1, max=10, value=1, step=1),
                   Switch(active=True),
                   Spinner(text="Spinner", values=("Python", "Java", "C++", "C", "C#", "PHP"), background_color=(0.784,0.443,0.216,1)),
                   ProgressBar(max=1000, value=750)]
                   #RectangleFlatButton().build()]
        return random.choice(widgets)


KV = """
<RectangleFlatButton>:
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
        Rectangle:
            pos: self.pos
            size: self.size
"""


class RectangleFlatButton(TouchRippleBehavior, Button):
    primary_color = [
        0.12941176470588237,
        0.5882352941176471,
        0.9529411764705882,
        1
    ]

    def build(self):
        screen = Builder.load_string(KV)
        btn = RectangleFlatButton(
                    text="flatbutton",
                    # pos_hint={"center_x": 0.5, "center_y": 0.5},
                    # size_hint=(None, None),
                    # size=(dp(110), dp(35)),
                    ripple_color=(0, 0, 0, 0),
                )
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
