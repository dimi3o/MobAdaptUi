import random
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider
from kivy.uix.switch import Switch
from kivy.uix.spinner import Spinner
from kivy.uix.progressbar import ProgressBar
from colors import allcolors, redclr, greenclr

class Widgets():
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
        return random.choice(widgets)