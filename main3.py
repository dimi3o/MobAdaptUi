import random

import agent
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
__version__ = '0.0.3.1'

class MainApp(App):
    FlyScatters = []
    IdsPngs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'MARL Mobile User Interface v.'+__version__
        self.text_color = MyColors.get_textcolor(WhiteBackColor)
        self.modes = ('Fly adapt', 'Size adapt', 'Rotate adapt', 'Fly+Size+Rotate adapt','MARL adapt', 'GAN adapt')
        self.cols_rows = ('1х1', '2х2', '3х3', '4х4', '5х5', '6х6', '8x5')
        self.objects = ('Apps', 'Foods', 'Widgets')
        self.episodes = ('20', '200', '2000', '20000')

    def build(self):
        # Root Layout
        self.root = BoxLayout(orientation='vertical', padding=10)  # ,size_hint_y=None)

        #HEAD PANEL
        self.colrowspinner = Spinner(text=self.cols_rows[6], values=self.cols_rows, background_color=(0.527, 0.154, 0.861, 1))
        self.colrowspinner.bind(text=self.colrowspinner_selected_value)
        self.objectspinner = Spinner(text=self.objects[1], values=self.objects, background_color=(0.027, 0.954, 0.061, 1))
        self.objectspinner.bind(text=self.colrowspinner_selected_value)
        lbl = Label(text='Size/Objs:', color=(0, 0, 1, 1))
        btn = Button(text='rebuild', size_hint_y=None, height='30dp', on_press=self.mainscreen_rebuild_btn_click)
        self.headpanel = self.init_hor_boxlayout([lbl, self.colrowspinner, self.objectspinner, btn])
        self.root.add_widget(self.headpanel)
        self.headpanel.bind(size=self._update_rect_headpanel, pos=self._update_rect_headpanel)

        with self.headpanel.canvas.before:
            Color(0.827, 0.827, 0.827, 1.)
            self.rect_headpanel = Rectangle()

        #MAIN CONTENT
        self.mainscreen_widgets = BoxLayout(orientation='vertical', padding=0, spacing=0)
        self.episodespinner = Spinner(text=self.episodes[1], values=self.episodes, size_hint_x=None, width='50dp', background_color=(0.225, 0.155, 0.564, 1))
        self.mainscreen_rebuild_btn_click(self)
        self.root.add_widget(self.mainscreen_widgets)

        #FOOT PANEL
        self.modespinner = Spinner(text="MARL adapt", values=self.modes, background_color=(0.127,0.854,0.561,1))
        btn = Button(text='ADAPT UI', size_hint_y=None, height='30dp', background_color=(1, 0, 0, 1), on_press=self.adapt_ui) #on_press=lambda null: self.show_popup('MARLMUI starting... '+self.modespinner.text, 'Info'))
        lbl = Label(text='Adaptation:', color=(0, 0, 1, 1)) #, size_hint_x=None, width='150dp')
        self.lblOnOff = Label(text='OFF', size_hint_x=None, width='30dp', color=(1, 0, 0, 1))
        self.total_reward = Label(text='0', size_hint_x=None, width='40dp', color=(0, 0, 1, 1))
        self.footpanel = self.init_hor_boxlayout([lbl,self.modespinner, self.episodespinner, btn, self.lblOnOff, self.total_reward])
        self.root.add_widget(self.footpanel)
        self.footpanel.bind(size=self._update_rect_footpanel, pos=self._update_rect_footpanel)
        with self.footpanel.canvas.before:
            Color(0.827, 0.827, 0.827, 1.)
            self.rect_footpanel = Rectangle()

        if WhiteBackColor:
            self.root.bind(size=self._update_rect, pos=self._update_rect)
            with self.root.canvas.before:
                Color(1, 1, 1, 1)
                self.rect = Rectangle(size=self.root.size, pos=self.root.pos)

        return self.root

    def adapt_ui(self, instance):
        m = self.modespinner.text
        if m == 'GAN adapt':
            self.show_popup('This adapt ui in the pipeline...', self.modespinner.text)
            return
        if m == 'MARL adapt' and self.lblOnOff.text == 'OFF':
            self.total_reward.text = '0'
        for s in self.FlyScatters:
            s.change_emulation()
            s.mode = self.modespinner.text
            if m == 'MARL adapt' and s.emulation:
                s.agent.total_reward = 0
                s.env.steps_left = int(self.episodespinner.text)
            self.lblOnOff.text = 'ON' if s.emulation else 'OFF'

    def stop_emulation_async(self,Text='Stop emulation!', Header='Adapt', par=0):
        if self.modespinner.text == 'MARL adapt':
            self.total_reward.text = str(int(self.total_reward.text)+int(par))
        if self.lblOnOff.text == 'ON':
            self.lblOnOff.text = 'OFF'
            self.show_popup(Text, Header)

    def colrowspinner_selected_value(self, spinner, text):
        self.mainscreen_rebuild_btn_click(self)

    def mainscreen_rebuild_btn_click(self, instance):
        self.mainscreen_widgets.clear_widgets()
        self.FlyScatters.clear()
        TextSize = self.colrowspinner.text
        Objects = self.objectspinner.text
        rows = int(TextSize[0])
        cols = int(TextSize[2])
        random.shuffle(self.IdsPngs)
        for i in range(rows):
            hor = BoxLayout(orientation='horizontal', padding=10, spacing=10, )
            for j in range(cols):#random.randint(1, 5)):
                s = FlyScatterV3(do_rotation=True, do_scale=True, auto_bring_to_front=False)
                s.app = self
                s.env = agent.Environment(int(self.episodespinner.text))
                s.agent = agent.Agent()
                hor.add_widget(s)
                ids = self.IdsPngs[i*cols+j]
                w = Widgets.get_app_icon(ids) if Objects=='Apps' else Widgets.get_food_icon(ids) if Objects=='Foods' else Widgets.get_random_widget()
                w.width = f'{360 // cols}dp'#f'{Window.width//cols}dp'
                w.height = f'{800 // rows}dp'#f'{Window.height//(rows+3)}dp'#
                s.add_widget(w)
                wi = Widgets.get_random_widget('Label')
                s.add_widget(wi)
                wi.text = str(ids)
                s.raw_width = w.width
                s.raw_height = w.height
                self.FlyScatters.append(s)

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