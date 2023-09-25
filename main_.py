import time
from dqnvianumpy.q_learning import train_main
from dqnvianumpy.q_learning import test_main
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.graphics import Color, Rectangle

WhiteBackColor = True
__version__ = '0.0.1.1'

class MainApp(App):

    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'DQN via NumPy v.' + __version__
        self.modeargs = ('train', 'test')
        self.r_modeargs = ('map', 'weights', 'stats')

    def build(self):

        self.root = BoxLayout(orientation='vertical', padding=10)
        if WhiteBackColor: self.root.bind(size=self._update_rect, pos=self._update_rect)

        self.console = TextInput(password=False, multiline=True)
        self.root.add_widget(self.console)
        self.dqnmodespinner = Spinner(text=self.modeargs[0], values=self.modeargs, background_color=(0.027, 0.954, 0.061, 1))
        self.dqnr_modespinner = Spinner(text=self.r_modeargs[0], values=self.r_modeargs, background_color=(0.027, 0.125, 0.061, 1))
        self.test_dqn_btn = Button(text='STEP', size_hint_y=None, height='30dp', background_color=(1, 0, 0, 1), on_press=self.step_btn)
        self.root.add_widget(self.init_hor_boxlayout([Label(text='Mode:', color=(1, 0, 1, 1)), self.dqnmodespinner, self.dqnr_modespinner, self.test_dqn_btn]))


        # create content and add to the popup
        popupbtn1 = Button(text='START!\nProcess in progress after click... ')
        popupbtn2 = Button(text='CANCEL')
        self.popup = Popup(title='Emulation',content=self.init_hor_boxlayout([popupbtn1,popupbtn2],Window.height / 7), auto_dismiss=False, size_hint=(None, None),
                      size=(2*Window.width / 3, Window.height / 4))
        popupbtn1.bind(on_press=lambda null: self.test_dqn(self))
        popupbtn2.bind(on_press=self.popup.dismiss)


        if WhiteBackColor:
            with self.root.canvas.before:
                Color(1, 1, 1)
                self.rect = Rectangle(size=self.root.size, pos=self.root.pos)
        return self.root

    def step_btn(self, instance):
        self.popup.open()

    def test_dqn(self, instance):
        #self.cclear()
        self.cwriteline('RUNNING...')
        if self.dqnmodespinner.text == "train":
            train_main(self.dqnr_modespinner.text, self)
        else:
            test_main('model.pkl', self)
        self.popup.dismiss()

    def cwriteline(self, string = ''):
        self.console.text += '\n'+string

    def cwrite(self, string = ''):
        self.console.text += string

    def cclear(self, string=''):
            self.console.text = ''

    def init_hor_boxlayout(self, widjets, height='30dp'):
        hor = BoxLayout(orientation='horizontal', padding=0, spacing=0, size_hint_y=None, height=height)
        for w in widjets: hor.add_widget(w)
        return hor

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

if __name__ == "__main__":
    app = MainApp()
    app.run()