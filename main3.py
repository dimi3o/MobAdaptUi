import random

import colors
from colors import MyColors
from kivywidgets import Widgets, FlyScatter, FlyScatterV3
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
    FlyScatters = []

    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'MARL Mobile User Interface v.'+__version__
        self.text_color = MyColors.get_textcolor(WhiteBackColor)
        self.modes = ("Fly adapt", "MARL adapt", "GAN adapt")
        self.width_height = ("1", "2", "3", "4", "5", "6")

    def build(self):
        # Root Layout
        self.root = BoxLayout(orientation='vertical', padding=10)  # ,size_hint_y=None)

        #HEAD PANEL
        self.widthspinner = Spinner(text="2", values=self.width_height, background_color=(0.527,0.154,0.861,1))
        self.heightspinner = Spinner(text="2", values=self.width_height, background_color=(0.527,0.154,0.861,1))
        lbl = Label(text='Width/Height":', color=(0, 0, 1, 1))
        btn = Button(text='rebuild', size_hint_y=None, height='30dp', on_press=self.mainscreen_rebuild_btn_click)
        self.headpanel = self.init_hor_boxlayout([lbl, self.widthspinner, self.heightspinner, btn])
        self.root.add_widget(self.headpanel)
        self.headpanel.bind(size=self._update_rect_headpanel, pos=self._update_rect_headpanel)

        with self.headpanel.canvas.before:
            Color(0.827, 0.827, 0.827, 1.)
            self.rect_headpanel = Rectangle()

        #MAIN CONTENT
        self.mainscreen_widgets = BoxLayout(orientation='vertical', padding=0, spacing=0)
        self.mainscreen_rebuild_btn_click(self)
        self.root.add_widget(self.mainscreen_widgets)

        #FOOT PANEL
        self.modespinner = Spinner(text="Fly adapt", values=self.modes, background_color=(0.127,0.854,0.561,1))
        btn = Button(text='adapt ui', size_hint_y=None, height='30dp', on_press=self.adapt_ui) #on_press=lambda null: self.show_popup('MARLMUI starting... '+self.modespinner.text, 'Info'))
        lbl = Label(text='Select mode and press "adapt ui":', color=(0, 0, 1, 1)) #, size_hint_x=None, width='150dp')
        self.footpanel = self.init_hor_boxlayout([lbl,self.modespinner, btn])
        self.root.add_widget(self.footpanel)
        self.footpanel.bind(size=self._update_rect_footpanel, pos=self._update_rect_footpanel)
        with self.footpanel.canvas.before:
            Color(0.827, 0.827, 0.827, 1.)
            self.rect_footpanel = Rectangle()
        # self.panel.canvas.before.add(Color(0.827, 0.827, 0.827, 1.))
        # self.panel.canvas.before.add(Rectangle(size=(1000, 45), pos=self.panel.pos))

        if WhiteBackColor:
            self.root.bind(size=self._update_rect, pos=self._update_rect)
            with self.root.canvas.before:
                Color(1, 1, 1, 1)
                self.rect = Rectangle(size=self.root.size, pos=self.root.pos)

        return self.root

    def adapt_ui(self, instance):
        for s in self.FlyScatters:
            s.change_emulation()
            mode = 'starting' if s.emulation else 'stopped'
        # self.show_popup('adapt ui '+mode+'...', self.modespinner.text)

    def mainscreen_rebuild_btn_click(self, instance):
        self.mainscreen_widgets.clear_widgets()
        self.FlyScatters.clear()
        width = int(self.widthspinner.text)
        height = int(self.heightspinner.text)
        for i in range(height):#random.randint(5, 10)):
            hor = BoxLayout(orientation='horizontal', padding=10, spacing=10, )
            for j in range(width):#random.randint(1, 5)):
                s = FlyScatterV3(do_rotation=True, do_scale=True, auto_bring_to_front=False)
                hor.add_widget(s)
                w = Widgets.get_random_widget()
                w.width = f'{550 // width}dp'
                w.height = f'{300 // height}dp'
                s.add_widget(w)
                self.FlyScatters.append(s)
                #hor.add_widget(Widgets.get_random_widget())
            self.mainscreen_widgets.add_widget(hor)

    def init_hor_boxlayout(self,widjets):
        hor = BoxLayout(orientation='horizontal', padding=0, spacing=0, size_hint_y=None, height='30dp')
        for w in widjets: hor.add_widget(w)
        return hor

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def _update_rect_headpanel(self, instance, value):
        self.rect_headpanel.pos = (instance.pos[0]-5, instance.pos[1]-5)
        self.rect_headpanel.size = (instance.size[0]+10, instance.size[1]+10)

    def _update_rect_footpanel(self, instance, value):
        self.rect_footpanel.pos = (instance.pos[0]-5, instance.pos[1]-5)
        self.rect_footpanel.size = (instance.size[0]+10, instance.size[1]+10)

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