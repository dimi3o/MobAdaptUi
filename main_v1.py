import random

from kivy.app import App
from kivy.animation import Animation
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.properties import NumericProperty, BooleanProperty
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.clock import Clock
from colors import allcolors, favcolors #, somecolors

class MainApp(App):
    emulation = BooleanProperty(False)
    reward = NumericProperty(0)
    def build(self):
        Clock.schedule_interval(self._update_clock, 1 / 60.)
        self.title = 'Adaptive UI'
        self.shift_padding = 20
        self.shift_spacing = 10
        self.root = BoxLayout(orientation="vertical", padding=10)  #,size_hint_y=None)
        self.display = Label()
        self.reward_lbl = Label(text='\n\nReward: 0')
        self.text_to_display("\n\nMinimalistic Display")
        hor = BoxLayout(orientation="horizontal", padding=10, spacing=10, size_hint_y=None)
        hor.add_widget(self.display)
        hor.add_widget(self.reward_lbl)
        self.root.add_widget(hor)
        hor = BoxLayout(orientation="horizontal", padding=10, spacing=10,size_hint_y=None)  # ,size_hint=(None, None))
        text = ["left","right","top","bottom"]
        for i in range(4):
            hor.add_widget(ToggleButton(size_hint_y=None, height='48dp', text=text[i],
                                        state="down" if i % 2 != 0 else "normal", group="g1" if i < 2 else "g2",on_release=self.on_toggle_click))
        hor.add_widget(ToggleButton(size_hint_y=None, height='48dp',text="EMU",state="normal",on_release=self.toggle_emulation))
        self.root.add_widget(hor)
        self.hor_shift = "right"; self.ver_shift = "bottom"
        self.topcol = 0; self.toprow = 0
        self.rows = 6; self.cols = 5
        for i in range(self.rows):
            hor = BoxLayout(orientation="horizontal", padding=10, spacing=10) #,size_hint=(None, None))
            for i in range(self.cols):
                btn = Button(
                    text="0",background_color=random.choice(allcolors)) #size=(200, 50),size_hint=(None, None))
                btn.bind(on_press=self.on_btn_click)
                hor.add_widget(btn)
            self.root.add_widget(hor)
        return self.root

    def toggle_emulation(self, instance):
        self.emulation = not self.emulation

    def _update_clock(self, dt):
        if self.emulation:
            rand_row = random.randint(0, self.rows - 1)
            rand_col = random.randint(0, self.cols - 1)
            self.on_btn_click(self.get_button_instance(rand_row,rand_col))
            self.text_to_display("\n\nEMULATION: row=" + str(rand_row) + ", col=" + str(rand_col))

    def get_button_instance(self, row=0, col=0):
        return self.root.children[row].children[col]

    def text_to_display(self,text=""):
        self.display.text = text

    def on_toggle_click(self, instance):
        if instance.group == 'g1': self.hor_shift = instance.text
        else: self.ver_shift = instance.text
        instance.state="down"
        self.topcol = 0 if self.hor_shift == "right" else self.cols - 1
        self.toprow = 0 if self.ver_shift == "bottom" else self.rows - 1

    def on_btn_click(self, instance):
        freq = int(instance.text) + 1
        if freq % 3 == 0:
            self.adapt_ui(instance, freq)
        instance.text = str(freq)

    def swap_toprow(self, freq, row, col):
        if row == self.toprow: return False
        maxfreq = 0;
        for r in range(self.rows):
            f = int(self.root.children[r].children[self.topcol].text)
            if f>maxfreq: maxfreq = f
        return True if freq>maxfreq else False

    def swap_topcol(self, freq, row, col):
        if col == self.topcol: return False
        max_freq = 0
        for c in range(self.cols):
            f = int(self.root.children[row].children[c].text)
            if f > max_freq: max_freq = f
        return True if freq > max_freq else False

    def swap_row(self, freq, row, col):
        c = col
        if self.hor_shift == "left":
            col_start = self.cols - 1
            col_end = 0
            step = -1
        else:
            col_start = 0
            col_end = self.cols - 1
            step = 1
        let_shift = False
        for c in range(col_start, col_end, step):
            f = int(self.root.children[row].children[c].text)
            if freq > f:
                let_shift = True
                break
        return c if let_shift and c != col else col

    def swap_col(self, freq, row, col):
        r = row
        if self.ver_shift == "top":
            row_start = self.toprow-1 if col == self.topcol else self.toprow
            row_end = -1
            step = -1
        else:
            row_start = self.toprow+1 if col == self.topcol else self.toprow
            row_end = self.rows
            step = 1
        for r in range(row_start, row_end, step):
            f = int(self.root.children[r].children[col].text)
            if freq > f: break
        return r if r != row else row

    def shift_from_to(self, instance, row, col, to_row, to_col):
        if row == to_row and col == to_col: return
        self.root.children[row].children[col] = self.root.children[to_row].children[to_col]
        self.root.children[to_row].children[to_col] = instance
        self.reward -= abs(row-to_row)+abs(col-to_col)
        self.update_reward()

    def adapt_ui(self, instance, freq):
        row, col = self.get_idx_children(instance)
        pos_x = instance.x; pos_y = instance.y
        anim = Animation(pos=(instance.x, instance.y), t='out_bounce', d=.1)
        if row >= 0 and col >= 0 and (row != self.toprow or col != self.topcol):
            # Column adapt
            if self.swap_topcol(freq, row, col):
                self.shift_from_to(instance, row, col, row, self.topcol)
                pos_x = self.animate_toprow(instance)
                col = self.topcol
            else:
                to_pos_col = self.swap_row(freq, row, col)
                if col != to_pos_col:
                    self.shift_from_to(instance, row, col, row, to_pos_col)
                    pos_x = self.animate_row(instance, to_pos_col)
                    col = to_pos_col

            # Row adapt
            if self.swap_toprow(freq, row, col):
                self.shift_from_to(instance, row, self.topcol, self.toprow, self.topcol)
                pos_y = self.animate_topcol(instance)
                row = self.toprow
            elif row != self.toprow:
                to_pos_row = self.swap_col(freq, row, col)
                if row != to_pos_row:
                    self.shift_from_to(instance, row, col, to_pos_row, col)
                    pos_y = self.animate_col(instance, to_pos_row)
                    row = to_pos_row

        if pos_x != instance.x or pos_y != instance.y:
            anim += Animation(pos=(pos_x, pos_y), t='out_bounce', d=.1)
            anim.start(instance)
            if not self.emulation:
                self.text_to_display("\n\nSWAP: row=" + str(row) + ", col=" + str(col))
        else:
            self.reward += 1
            self.update_reward()

    def update_reward(self):
        self.reward_lbl.text = "\n\nReward: " + str(self.reward)

    def get_idx_children(self, instance):
        row = -1; col = -1
        for child in self.root.children:
            row += 1
            if len(child.children)>0:
                try: col = child.children.index(instance)
                except: col = -1
                if col >= 0:
                    break
        return row, col

    def animate_toprow(self, instance):
        if self.hor_shift == "right":
            pos_x = Window.width - instance.width - self.shift_padding
        else:
            pos_x = self.shift_padding
        return pos_x

    def animate_topcol(self, instance):
        if self.ver_shift == "bottom":
            pos_y = self.shift_padding
        else:
            pos_y = instance.height*(self.rows-1)+self.shift_padding*self.rows
        return pos_y

    def animate_col(self, instance, row):
        return self.shift_padding*(row+1)+instance.height*row

    def animate_row(self, instance, col):
        return instance.width*(self.cols - col - 1) + self.shift_spacing*(self.cols - col) + 10

    def show_popup(self, text="", title="Popup Window"):
        popup = Popup(title=title, size_hint=(None, None),
                      size=(Window.width / 2, Window.height / 4))
        layout = BoxLayout(orientation="vertical", padding=10)
        layout.add_widget(Label(text=text))
        layout.add_widget(Button(text="OK",on_press=popup.dismiss))
        popup.content = layout
        popup.open()

if __name__ == "__main__":
    app = MainApp()
    #app.show_popup("The frequency of clicks adapts the interface", "Info")
    app.run()
