import random
from colors import MyColors
from kivywidgets import Widgets, FlyScatter
from kivy.app import App
from kivy.properties import NumericProperty
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.uix.spinner import Spinner
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
        self.title = 'MARL Mobile User Interface v.'+__version__
        self.text_color = MyColors.get_textcolor(WhiteBackColor)
        self.modes = ("mode1", "mode2", "mode3", "mode4", "mode5")

    def build(self):
        # Root Layout
        self.root = BoxLayout(orientation='vertical', padding=10)  # ,size_hint_y=None)
        self.mainscreen_widgets = BoxLayout(orientation='vertical', padding=0, spacing=0)
        self.mainscreen_rebuild_btn_click()
        self.root.add_widget(self.mainscreen_widgets)
        self.modespinner = Spinner(text="mode1", values=self.modes, background_color=(0.127,0.854,0.561,1))
        btn = Button(text='adapt ui', size_hint_y=None, height='30dp', on_press=self.adapt_ui)#on_press=lambda null: self.show_popup('MARLMUI starting... '+self.modespinner.text, 'Info'))
        lbl = Label(text='Select mode and press "adapt ui":', color=(0, 0, 1, 1)) #, size_hint_x=None, width='150dp')
        self.panel = self.init_hor_boxlayout([lbl,self.modespinner, btn])
        self.root.add_widget(self.panel)
        self.panel.canvas.before.add(Color(0.827, 0.827, 0.827, 1.))
        self.panel.canvas.before.add(Rectangle(size=(1000,45), pos=self.panel.pos))
        if WhiteBackColor:
            self.root.bind(size=self._update_rect, pos=self._update_rect)
            with self.root.canvas.before:
                MyColors.get_textcolor(WhiteBackColor) #Color(1, 1, 1)
                self.rect = Rectangle(size=self.root.size, pos=self.root.pos)
        return self.root

    def adapt_ui(self, instance):
        self.show_popup('MARLMUI starting... ' + self.modespinner.text, 'Info')
        self.mainscreen_rebuild_btn_click()

    def mainscreen_rebuild_btn_click(self):
        self.mainscreen_widgets.clear_widgets()
        for i in range(random.randint(5, 10)):
            hor = BoxLayout(orientation='horizontal', padding=10, spacing=10, )
            for j in range(random.randint(1, 5)):
                hor.add_widget(Widgets.get_random_widget())
            self.mainscreen_widgets.add_widget(hor)

    def init_hor_boxlayout(self,widjets):
        hor = BoxLayout(orientation='horizontal', padding=0, spacing=0, size_hint_y=None, height='30dp')
        for w in widjets: hor.add_widget(w)
        return hor

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