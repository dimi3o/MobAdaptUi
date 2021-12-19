from kivy.app import App
from kivy.properties import NumericProperty, BooleanProperty, StringProperty, ListProperty
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

__version__ = '0.0.2'

class MainScreen(Screen):
    fullscreen = BooleanProperty(False)
    current_title = StringProperty()
    def add_widget(self, *args, **kwargs):
        if 'content' in self.ids:
            return self.ids.content.add_widget(*args, **kwargs)
        return super(MainScreen, self).add_widget(*args, **kwargs)

class MainApp(App):
    index = NumericProperty(-1)
    time = NumericProperty(0)
    emulation = BooleanProperty(False)
    screen_names = ListProperty([])
    hierarchy = ListProperty([])

    def build(self):
        self.screens = {}
        self.screen_names = ['MainScreen']
        self.available_screens = ['mainscreen.kv']
        self.screen = self.load_screen(0)
        self.content = self.screen.ids.content
        sm = self.root.ids.sm
        sm.switch_to(self.screen, direction='right')
        self.current_title = self.screen.name
        self.text_to_display("\nMinimalistic Display")

    def load_screen(self,index=0):
        self.display = Label()
        #self.display.bind(size=self.display.setter('text_size'))
        self.title = 'Adaptive UI'
        screen = Builder.load_file(self.available_screens[index])
        screen.add_widget(self.display)
        return screen

    def go_hierarchy_previous(self):
        pass

    def text_to_display(self,text=""):
        self.display.text = "\n"+text

    def toggle_emulation(self):
        self.text_to_display("\nemulation")

    def on_toggle_left_right(self):
        self.text_to_display("\nleft/right")

    def on_toggle_top_bottom(self):
        self.text_to_display("\ntop/bottom")

if __name__ == "__main__":
    app = MainApp()
    app.run()
