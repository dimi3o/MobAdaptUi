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
        widgets = ['Image', 'TextInput', 'Label', 'Button', 'CheckBox', 'Slider', 'Switch', 'Spinner', 'ProgressBar']
        selected_name = random.choice(widgets)
        if selected_name == 'Image': return Image(source='data/icons/bug1.png')
        elif selected_name == 'TextInput': return TextInput(text='textinput')#(font_size = 20, size_hint_y = None, height = 50)
        elif selected_name == 'Label': return Label(text='label', color=random.choice(allcolors))
        elif selected_name == 'Button': return Button(text='button', background_color=random.choice(allcolors))#on_press=self.on_btn_click))
        elif selected_name == 'CheckBox': return CheckBox(active = True)
        elif selected_name == 'Slider': return Slider(min=1, max=10, value=1, step=1)
        elif selected_name == 'Switch': return Switch(active = True)
        elif selected_name == 'Spinner': return Spinner(text ="Spinner", values =("Python", "Java", "C++", "C", "C#", "PHP"), background_color =(0.784, 0.443, 0.216, 1))
        elif selected_name == 'ProgressBar': return ProgressBar(max = 1000, value = 750)