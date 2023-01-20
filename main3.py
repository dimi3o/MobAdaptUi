import random
from kivywidgets import Widgets, FlyScatter
from kivy.app import App
from kivy.properties import NumericProperty
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.uix.slider import Slider
from kivy.uix.checkbox import CheckBox
from kivy.uix.scatter import Scatter
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle

WhiteBackColor = True
__version__ = '0.0.3.0'

class MainApp(App):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
    def build(self):
        # Root Layout
        self.root = BoxLayout(orientation='vertical', padding=10)  # ,size_hint_y=None)
        btn = Button(text='adapt ui', size_hint_y=None, height='30dp', on_press=lambda null: self.show_popup("MARLMUI starting...", "Info"))
        self.root.add_widget(btn)
        if WhiteBackColor:
            self.root.bind(size=self._update_rect, pos=self._update_rect)
            with self.root.canvas.before:
                Color(1, 1, 1)
                self.rect = Rectangle(size=self.root.size, pos=self.root.pos)
        return self.root

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def show_popup(self, text='', title='Popup Window'):
        popup = Popup(title=title, size_hint=(None, None),
                      size=(Window.width / 2, Window.height / 4))
        layout = BoxLayout(orientation='vertical', padding=10)
        layout.add_widget(Label(text=text))
        layout.add_widget(Button(text='OK', on_press=popup.dismiss))
        popup.content = layout
        popup.open()

if __name__ == "__main__":
    app = MainApp()
    app.run()