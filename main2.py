import random
from kivywidgets import Widgets
from kivy.app import App
from kivy.properties import NumericProperty
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.scatter import Scatter
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle

WhiteBackColor = True
__version__ = '0.0.2.1'

def get_textcolor():
    return [0, 0, 0, 1] if WhiteBackColor else [1, 1, 1, 1]

def get_hor_boxlayout(orientation='horizontal', padding=10, spacing=10):
    return BoxLayout(orientation=orientation, padding=padding, spacing=spacing)

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
        self.root2.add_widget(Label(text='Image, TextInput, Label,\nButton, CheckBox, Slider,\nSwitch, Spinner, ProgressBar', color=get_textcolor()))
        self.rows_slider = Slider(min=1, max=10, value=5, step=1)
        self.cols_slider = Slider(min=1, max=10, value=5, step=1)
        hor = get_hor_boxlayout()
        hor.add_widget(Label(text='ROWS:', color=get_textcolor(), size_hint_x=None, width='50dp'))
        hor.add_widget(self.rows_slider)
        self.root2.add_widget(hor)
        hor = get_hor_boxlayout()
        hor.add_widget(Label(text='COLUMNS:', color=get_textcolor(), size_hint_x=None, width='50dp'))
        hor.add_widget(self.cols_slider)
        self.root2.add_widget(hor)
        self.root2.add_widget(Button(text='main screen', size_hint_y=None, height='30dp', on_press=self.to_scr1_btn_click))
        self.add_screen('properties', self.root2)

        # SANDBOX LAYOUT
        self.root3 = BoxLayout(orientation='vertical', padding=10)
        self.root3.add_widget(Button(text='rebuild', size_hint_y=None, height='30dp', on_press=self.sandbox_rebuild_btn_click))
        self.sandbox_widgets = BoxLayout(orientation='vertical', padding=0, spacing=0)
        self.sandbox_rebuild_btn_click(self)
        self.root3.add_widget(self.sandbox_widgets)
        self.root3.add_widget(Button(text='mainscreen', size_hint_y=None, height='30dp', on_press=self.to_scr1_btn_click_left))
        self.add_screen('sandbox', self.root3)

        # MAINSCREEN LAYOUT
        self.root1 = BoxLayout(orientation='vertical', padding=10)
        self.root1.add_widget(Button(text='rebuild', size_hint_y=None, height='30dp', on_press=self.mainscreen_rebuild_btn_click))
        self.mainscreen_widgets = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.mainscreen_rebuild_btn_click(self)
        self.root1.add_widget(self.mainscreen_widgets)
        hor = BoxLayout(orientation='horizontal', padding=0, spacing=0, size_hint_y=None, height='10dp')
        hor.add_widget(Button(text='sandbox', size_hint_y=None, height='30dp', on_press=self.to_scr3_btn_click))
        hor.add_widget(Button(text='properties', size_hint_y=None, height='30dp', on_press=self.to_scr2_btn_click))
        self.root1.add_widget(hor)
        self.add_screen('mainscreen', self.root1)
        self.to_scr1_btn_click(self)

        if WhiteBackColor:
            self.sm.bind(size=self._update_rect, pos=self._update_rect)
            with self.sm.canvas.before:
                Color(1, 1, 1)
                self.rect = Rectangle(size=self.sm.size, pos=self.sm.pos)
        return self.sm

    def add_screen(self, name, widget):
        scr = Screen(name=name)
        scr.add_widget(widget)
        self.sm.add_widget(scr)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def mainscreen_rebuild_btn_click(self, instance):
        self.mainscreen_widgets.clear_widgets()
        for i in range(self.rows_slider.value):
            hor = get_hor_boxlayout()
            for j in range(random.randint(1,self.cols_slider.value-1)):
                hor.add_widget(Widgets.get_random_widget())
            self.mainscreen_widgets.add_widget(hor)

    def sandbox_rebuild_btn_click(self, instance):
        self.sandbox_widgets.clear_widgets()
        for i in range(self.rows_slider.value):
            hor = get_hor_boxlayout('horizontal', 0, 0)
            for j in range(self.cols_slider.value):
                s = Scatter(do_rotation=False, do_scale=False, auto_bring_to_front=False)
                hor.add_widget(s)
                w = Widgets.get_random_widget()
                w.height = f'{300/self.cols_slider.value}dp'
                s.add_widget(w)
            self.sandbox_widgets.add_widget(hor)

    def to_scr1_btn_click(self, instance):
        self.sm.transition.direction = 'right'
        self.sm.current = 'mainscreen'

    def to_scr1_btn_click_left(self, instance):
        self.sm.transition.direction = 'left'
        self.sm.current = 'mainscreen'

    def to_scr2_btn_click(self, instance):
        self.sm.transition.direction = 'left'
        self.sm.current = 'properties'

    def to_scr3_btn_click(self, instance):
        self.sm.transition.direction = 'right'
        self.sm.current = 'sandbox'

    def on_btn_click(self, instance):
        freq = int(instance.text) + 1
        instance.text = str(freq)

if __name__ == "__main__":
    app = MainApp()
    app.run()
