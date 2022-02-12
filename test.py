from kivy.app import App

from kivy.uix.scatter import Scatter
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout

class TutorialApp(App):
    def build(self):
        b = BoxLayout(orientation="vertical")
        f0 = FloatLayout()
        for i in range(5):
            f = FloatLayout()
            s = Scatter()
            l = Label(text=f"Hello{i}!",
                      font_size=50)
            f.add_widget(s)
            s.add_widget(l)
            f0.add_widget(f)
        return f0

if __name__ == "__main__":
    TutorialApp().run()