from kivy.app import App
from kivy.lang import Builder
from kivy.animation import Animation
from kivy.uix.button import Button
from kivy.properties import ObjectProperty
from kivy.uix.gridlayout import GridLayout

from os import listdir
kv_path = '../kv/'
for kv in listdir(kv_path):
    Builder.load_file(kv_path+kv)


class AddButton(Button):
    pass

class SubtractButton(Button):
    pass

class SimpleButton(Button):
    pass

class Container(GridLayout):
    display = ObjectProperty()

    def add_one(self):
        value = int(self.display.text)
        self.display.text = str(value+1)

    def add_one(self):
        value = int(self.display.text)
        self.display.text = str(value+1)

    def subtract_one(self):
        value = int(self.display.text)
        self.display.text = str(value-1)

    def animate(self):
        # create an animation object. This object could be stored
        # and reused each call or reused across different widgets.
        # += is a sequential step, while &= is in parallel
        animation = Animation(pos=(100, 100), t='out_bounce')
        animation += Animation(pos=(200, 100), t='out_bounce')
        animation &= Animation(size=(500, 500))
        animation += Animation(size=(100, 50))

        self.btn.text = str(int(self.btn.text)+1)

        # apply the animation on the button, passed in the "instance" argument
        # Notice that default 'click' animation (changing the button
        # color while the mouse is down) is unchanged.
        animation.start(self.btn)

class MainApp(App):

    def build(self):
        self.title = 'Adaptive Ui App!!!'
        return Container() # Simle animate v1

if __name__ == "__main__":
    app = MainApp()
    app.run()