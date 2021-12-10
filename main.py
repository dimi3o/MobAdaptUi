import random

from kivy.app import App
from kivy.lang import Builder
from kivy.animation import Animation
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.core.window import Window

from os import listdir
kv_path = './kv/'
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

red = [1,0,0,1]
gray = [.10, .35, .41, .85]
green = [0,1,0,1]
lgreen = [0,1,1,1]
blue = [0,0,1,1]
purple = [1,0,1,1]
colors = [red, green, blue, lgreen, purple]

class MainApp(App):

    def build(self):
        #return Container() # Simle animate v1
        self.title = 'Adaptive Ui App!!!'
        self.root = BoxLayout(orientation="vertical", padding=10)  #,size_hint_y=None)
        self.root.add_widget(Label(text="ADAPTIVE UI",size=(200, 50),size_hint=(1, None)))
        for i in range(4):
            hor = BoxLayout(orientation="horizontal", padding=10, spacing=10) #,size_hint=(None, None))
            for i in range(4):
                btn = Button(
                    text="0",background_color=gray) #size=(200, 50),size_hint=(None, None))
                btn.bind(on_press=self.on_btn_click)
                hor.add_widget(btn)
            self.root.add_widget(hor)
        return self.root

    def on_btn_click(self, instance):
        freq = int(instance.text) + 1
        if freq % 3 == 0: self.adapt_row(instance,freq)
        else: instance.text = str(freq)

    def adapt_row(self,instance,freq):
        adapt = False
        row, col, list, maxfreq = self.get_idx_children(instance)
        if row >= 0 and col >= 0 and list != None:
            if int(freq) > maxfreq and col > 0:
                x = instance.x
                animation = Animation(pos=(x, instance.y), t='out_bounce')
                animation += Animation(pos=(Window.width-instance.width-20, instance.y), t='out_bounce')
                animation.start(instance)

                adapt = True
                l1 = len(list.children)
                list.remove_widget(instance)
                l2 = len(list.children)
                if (l1 - l2 > 0):
                    list.add_widget(instance)
        instance.text = str(freq)
        if adapt:
            instance.background_color = random.choice(colors)
            #self.show_popup(f"Frequency click: {freq}\n\nRight shift >>>",f"Row Adapt ({row},{col})")

    def get_idx_children(self, instance):
        row = -1; col = -1; maxfreq = 0; list = None
        for child in self.root.children:
            row += 1
            if len(child.children)>0:
                try: col = child.children.index(instance)
                except: col = -1
                if col >= 0:
                    list = child;
                    for btn in child.children:
                        if int(btn.text)>maxfreq: maxfreq=int(btn.text)
                    break
        return row, col, list, maxfreq

    def show_popup(self, text="", title="Popup Window"):
        popupWindow = Popup(title=title, content=Label(text=text), size_hint=(None, None),
                            size=(Window.width / 2, Window.height / 4))
        popupWindow.open()

if __name__ == "__main__":
    app = MainApp()
    app.run()