from kivy.app import App
from kivy.properties import NumericProperty, BooleanProperty, StringProperty, ListProperty
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout

__version__ = '0.0.2.0'

class MainApp(App):
    sm = ScreenManager()
    def build(self):
        self.root1 = BoxLayout(orientation='vertical', padding=10)
        self.root1.add_widget(Label(text='FIRST SCREEN'))
        self.root1.add_widget(Button(text='To second screen',on_press=self.to_scr2_btn_click))
        scr1 = Screen(name='scr1')
        scr1.add_widget(self.root1)
        self.root2 = BoxLayout(orientation='vertical', padding=10)
        self.root2.add_widget(Label(text='SECOND SCREEN'))
        self.root2.add_widget(Button(text='To first screen', on_press=self.to_scr1_btn_click))
        scr2 = Screen(name='scr2')
        scr2.add_widget(self.root2)
        self.sm.add_widget(scr1)
        self.sm.add_widget(scr2)
        return self.sm

    def to_scr1_btn_click(self, instance):
        self.sm.transition.direction = 'right'
        self.sm.current = 'scr1'

    def to_scr2_btn_click(self, instance):
        self.sm.transition.direction = 'left'
        self.sm.current = 'scr2'

if __name__ == "__main__":
    app = MainApp()
    app.run()
