import random
from kivy.app import App
from kivy.properties import NumericProperty, BooleanProperty, StringProperty, ListProperty
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider
from kivy.uix.switch import Switch
from kivy.uix.spinner import Spinner
from kivy.uix.progressbar import ProgressBar
from kivy.graphics import Ellipse
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle
from colors import allcolors, redclr, greenclr

WhiteBackColor = True
__version__ = '0.0.2.0'

def get_textcolor():
    return [0, 0, 0, 1] if WhiteBackColor else [1, 1, 1, 1]

def get_hor_boxlayout(orientation='horizontal', padding=10, spacing=10):
    return BoxLayout(orientation=orientation, padding=padding, spacing=spacing)

def get_random_widget():
    widgets = ['Image', 'TextInput', 'Label', 'Button', 'CheckBox', 'Slider', 'Switch', 'Spinner', 'ProgressBar']
    selected_name = random.choice(widgets)
    if selected_name == 'Image': return Image(source='data/icons/bug1.png')
    elif selected_name == 'TextInput': return TextInput(text='textinput')#(font_size = 20, size_hint_y = None, height = 50)
    elif selected_name == 'Label': return Label(text='label', color=get_textcolor())
    elif selected_name == 'Button': return Button(text='button', background_color=random.choice(allcolors))#on_press=self.on_btn_click))
    elif selected_name == 'CheckBox': return CheckBox(active = True)
    elif selected_name == 'Slider': return Slider(min=1, max=10, value=1, step=1)
    elif selected_name == 'Switch': return Switch(active = True)
    elif selected_name == 'Spinner': return Spinner(text ="Spinner", values =("Python", "Java", "C++", "C", "C#", "PHP"), background_color =(0.784, 0.443, 0.216, 1))
    elif selected_name == 'ProgressBar': return ProgressBar(max = 1000, value = 750)

class MainApp(App):
    sm = ScreenManager()
    rows = NumericProperty(5)
    cols = NumericProperty(5)

    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'Adaptive Mobile UI'

    def build(self):
        # PROPERTIES LAYOUT
        self.root2 = BoxLayout(orientation='vertical', padding=10)
        self.root2.add_widget(Label(text='Image, TextInput, Label, Button, CheckBox, Slider, Switch, Spinner, ProgressBar', color=get_textcolor()))
        self.rows_slider = Slider(min=1, max=10, value=5, step=1)
        self.cols_slider = Slider(min=1, max=10, value=5, step=1)
        self.root2.add_widget(self.rows_slider)
        self.root2.add_widget(self.cols_slider)
        self.root2.add_widget(Button(text='main screen', size_hint_y=None, height='30dp', on_press=self.to_scr1_btn_click))
        # MAIN SCREEN ROOT LAYOUT
        self.root1 = BoxLayout(orientation='vertical', padding=10)
        self.root1.add_widget(Button(text='rebuild of layout', size_hint_y=None, height='30dp', on_press=self.rebuild_btn_click))
        self.widgets_layout = BoxLayout(orientation='vertical')
        self.rebuild_btn_click(self)
        self.root1.add_widget(self.widgets_layout)
        self.root1.add_widget(Button(text='properties screen', size_hint_y=None, height='30dp', on_press=self.to_scr2_btn_click))
        scr1 = Screen(name='scr1')
        scr1.add_widget(self.root1)
        scr2 = Screen(name='scr2')
        scr2.add_widget(self.root2)
        self.sm.add_widget(scr1)
        self.sm.add_widget(scr2)
        if WhiteBackColor:
            self.sm.bind(size=self._update_rect, pos=self._update_rect)
            with self.sm.canvas.before:
                Color(1, 1, 1)
                self.rect = Rectangle(size=self.sm.size, pos=self.sm.pos)
        return self.sm

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def rebuild_btn_click(self, instance):
        self.widgets_layout.clear_widgets()
        for i in range(self.rows_slider.value):
            hor = get_hor_boxlayout()
            for j in range(random.randint(1,self.cols_slider.value-1)):
                hor.add_widget(get_random_widget())
            self.widgets_layout.add_widget(hor)

    def to_scr1_btn_click(self, instance):
        self.sm.transition.direction = 'right'
        self.sm.current = 'scr1'

    def to_scr2_btn_click(self, instance):
        self.sm.transition.direction = 'left'
        self.sm.current = 'scr2'

    def on_btn_click(self, instance):
        freq = int(instance.text) + 1
        instance.text = str(freq)

if __name__ == "__main__":
    app = MainApp()
    app.run()
