__version__ = '0.0.1.6'

import random
from kivy.app import App
from kivy.animation import Animation
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
from kivy_garden.graph import Graph, LinePlot  # , MeshLinePlot
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.slider import Slider
from kivy.core.window import Window
from kivy.clock import Clock
from colors import allcolors, redclr, greenclr
from strategy import AdaptStrategy


def get_hor_boxlayout(orientation='horizontal', padding=10, spacing=10):
    return BoxLayout(orientation=orientation, padding=padding, spacing=spacing)


class MainApp(App):
    emulation = BooleanProperty(False)
    adapt_frequency = NumericProperty(0)
    reward = NumericProperty(0)
    tb_strategy = ListProperty([])
    shift_spacing = NumericProperty(10)
    shift_padding = NumericProperty(20)
    rows = NumericProperty(7)
    cols = NumericProperty(7)

    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'Adaptive Mobile UI'
        self.hor_shift = 'right'
        self.ver_shift = 'bottom'
        self.adapt_strategy = AdaptStrategy.RightBottom
        self.top_col = 0
        self.top_row = 0
        self.freq_points = []
        self.reward_points = []

    def build(self):
        # Root Layout
        self.root = BoxLayout(orientation='vertical', padding=10)  # ,size_hint_y=None)

        # Plot
        hor = BoxLayout(orientation='horizontal', padding=10, size_hint_y=None, height='10dp')
        hor.add_widget(Label(text='Adapt Frequency'))
        hor.add_widget(Label(text='Reward'))
        self.root.add_widget(hor)
        self.freq_graph = Graph(x_ticks_minor=0,  # xlabel='T'
                                x_ticks_major=5, y_ticks_major=10,
                                y_grid_label=True, x_grid_label=True, padding=5,
                                x_grid=True, y_grid=True, xmin=0, xmax=1, ymin=0, ymax=30)
        self.freq_plot = LinePlot(line_width=2, color=[1, 0, 0, 1])
        self.freq_graph.add_plot(self.freq_plot)
        self.reward_graph = Graph(x_ticks_minor=0,
                                  x_ticks_major=5, y_ticks_major=100,
                                  y_grid_label=True, x_grid_label=True, padding=5,
                                  x_grid=True, y_grid=True, xmin=0, xmax=1, ymin=0, ymax=1)
        self.reward_plot = LinePlot(line_width=2, color=[1, 0, 0, 1])
        self.reward_graph.add_plot(self.reward_plot)
        graph_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height='200dp')
        graph_layout.add_widget(self.freq_graph)
        graph_layout.add_widget(self.reward_graph)
        self.root.add_widget(graph_layout)

        # Labels Panel
        self.adapt_freq_lbl = Label(text='F:0', color=greenclr)
        self.display = Label(text='Action')
        self.reward_lbl = Label(text='R: 0', color=redclr)
        hor = get_hor_boxlayout()
        hor.add_widget(self.display)
        hor.add_widget(self.adapt_freq_lbl)
        hor.add_widget(self.reward_lbl)
        hor.add_widget(ToggleButton(text='reset', state='normal', on_release=self.on_reset, size_hint_x=None))
        self.root.add_widget(hor)
        hor = get_hor_boxlayout()
        self.adapt_request_slider = Slider(min=1, max=15, value=3, step=1)
        self.adapt_request_slider.bind(value=self.on_move_adapt_request)
        self.adapt_request_lbl = Label(text='Adapt request: ' + str(self.adapt_request_slider.value))
        hor.add_widget(self.adapt_request_lbl)
        hor.add_widget(self.adapt_request_slider)
        self.root.add_widget(hor)

        # Strategy Panel
        hor = get_hor_boxlayout()
        self.tb_strategy.append(ToggleButton(text='left', state='normal', group='g0', on_release=self.on_change_adapt_strategy))
        self.tb_strategy.append(ToggleButton(text='right', state='down', group='g0', on_release=self.on_change_adapt_strategy))
        self.tb_strategy.append(ToggleButton(text='top', state='normal', group='g1', on_release=self.on_change_adapt_strategy))
        self.tb_strategy.append(ToggleButton(text='bottom', state='down', group='g1', on_release=self.on_change_adapt_strategy))
        self.tb_strategy.append(ToggleButton(text='center', state='normal', group='g2', on_release=self.on_change_adapt_strategy))
        self.tb_strategy.append(ToggleButton(text='rand', state='normal', group='g2', on_release=self.on_change_adapt_strategy))
        self.tb_strategy.append(ToggleButton(text='EMU', state='normal', on_release=self.toggle_emulation))
        for tb in self.tb_strategy: hor.add_widget(tb)
        self.root.add_widget(hor)

        # Buttons Panel
        for i in range(self.rows):
            hor = get_hor_boxlayout()
            for j in range(self.cols):
                hor.add_widget(Button(text='0',
                                      background_color=random.choice(allcolors),
                                      on_press=self.on_btn_click))
            self.root.add_widget(hor)
        return self.root

    def on_reset(self, instance):
        instance.state = 'normal'
        self.tb_strategy[6].state = 'normal'
        Clock.unschedule(self._update_clock)
        Clock.unschedule(self._update_clock_sec)
        self.emulation = False
        for i in range(0, self.rows):
           for j in range(0, self.cols):
               self.root.children[i].children[j].text = '0'
        self.adapt_frequency = 0
        self.adapt_freq_lbl.text = 'F: 0'
        self.reward = 0
        self.reward_lbl.text = 'R: 0'
        self.freq_graph.remove_plot(self.freq_plot)
        self.freq_plot = LinePlot(line_width=2, color=[1, 0, 0, 1])
        self.freq_graph.add_plot(self.freq_plot)
        self.freq_points = []
        self.freq_graph.x_ticks_major = 5; self.freq_graph.y_ticks_major = 10
        self.freq_graph.xmin = 0; self.freq_graph.xmax = 1; self.freq_graph.ymin = 0; self.freq_graph.ymax = 30
        self.reward_graph.remove_plot(self.reward_plot)
        self.reward_plot = LinePlot(line_width=2, color=[1, 0, 0, 1])
        self.reward_graph.add_plot(self.reward_plot)
        self.reward_points = []
        self.reward_graph.x_ticks_major = 5; self.reward_graph.y_ticks_major = 100
        self.reward_graph.xmin = 0; self.reward_graph.xmax = 1; self.reward_graph.ymin = 0; self.reward_graph.ymax = 1

    def on_change_adapt_strategy(self, instance):
        if self.adapt_strategy.value == 4 or self.adapt_strategy.value == 5:
            if instance.group == 'g0' or instance.group == 'g1':
                instance.state = 'normal'
                return
        self.adapt_strategy, self.top_col, self.top_row = AdaptStrategy.change_strategy(instance, self.adapt_strategy,
                                                                                        self.cols, self.rows)
        if instance.group == 'g0':
            self.hor_shift = instance.text
            instance.state = 'down'
        elif instance.group == 'g1':
            self.ver_shift = instance.text
            instance.state = 'down'
        else:
            if instance.state == 'down':
                for i in range(0, 4): self.tb_strategy[i].state = 'normal'
            elif instance.state == 'normal':
                # LeftTop=0 LeftBottom=1 RightTop=2 RightBottom=3 Center=4 Rand=5
                if self.adapt_strategy.value == 0:
                    i = 0; j = 2
                elif self.adapt_strategy.value == 1:
                    i = 0; j = 3
                elif self.adapt_strategy.value == 2:
                    i = 1; j = 2
                else:
                    i = 1; j = 3
                self.tb_strategy[i].state = 'down'
                self.tb_strategy[j].state = 'down'
                self.hor_shift = self.tb_strategy[i].text
                self.ver_shift = self.tb_strategy[j].text

    def on_move_adapt_request(self, instance, value):
        self.adapt_request_lbl.text = 'Adapt request: {} '.format(int(self.adapt_request_slider.value))

    def toggle_emulation(self, instance):
        self.emulation = not self.emulation
        if self.emulation:
            Clock.schedule_interval(self._update_clock, 1 / 60.)
            Clock.schedule_interval(self._update_clock_sec, 1)
        else:
            Clock.unschedule(self._update_clock)
            Clock.unschedule(self._update_clock_sec)

    def _update_clock_sec(self, dt):
        # Adapt Freq Plot
        if self.freq_graph.ymax < self.adapt_frequency:
            self.freq_graph.ymax = self.adapt_frequency
        if self.adapt_frequency > self.freq_graph.y_ticks_major * 5: self.freq_graph.y_ticks_major *= 2
        if self.freq_graph.xmax > self.freq_graph.x_ticks_major * 5: self.freq_graph.x_ticks_major *= 4
        self.freq_points.append((self.freq_graph.xmax - 1, self.adapt_frequency))
        self.freq_plot.points = [(x, y) for x, y in self.freq_points]
        self.freq_graph.xmax += 1

        # Reward Plot
        if self.reward_graph.ymax < self.reward:
            self.reward_graph.ymax = self.reward
        elif self.reward_graph.ymin > self.reward:
            self.reward_graph.ymin = self.reward
        if abs(self.reward) > self.reward_graph.y_ticks_major * 5: self.reward_graph.y_ticks_major *= 2
        if self.reward_graph.xmax > self.reward_graph.x_ticks_major * 5: self.reward_graph.x_ticks_major *= 4
        self.reward_points.append((self.reward_graph.xmax - 1, self.reward))
        self.reward_plot.points = [(x, y) for x, y in self.reward_points]
        self.reward_graph.xmax += 1

        # Adapt Freq Label
        self.adapt_freq_lbl.text = 'F: ' + str(self.adapt_frequency)  # 'Adapt freq.(in 1 sec): '
        self.adapt_frequency = 0

    def _update_clock(self, dt):
        if self.emulation:
            rand_row = random.randint(0, self.rows - 1)
            rand_col = random.randint(0, self.cols - 1)
            self.on_btn_click(self.get_button_instance(rand_row, rand_col))
            self.text_to_display('row=' + str(rand_row) + ', col=' + str(rand_col))

    def get_button_instance(self, row=0, col=0):
        return self.root.children[row].children[col]

    def text_to_display(self, text=''):
        self.display.text = text

    def on_btn_click(self, instance):
        freq = int(instance.text) + 1
        if freq % self.adapt_request_slider.value == 0:
            self.adapt_ui(instance, freq)
        instance.text = str(freq)

    def swap_toprow(self, freq, row, col):
        if row == self.top_row: return False
        maxfreq = 0
        for r in range(self.rows):
            f = int(self.root.children[r].children[self.top_col].text)
            if f > maxfreq: maxfreq = f
        return True if freq > maxfreq else False

    def swap_topcol(self, freq, row, col):
        if col == self.top_col: return False
        max_freq = 0
        for c in range(self.cols):
            f = int(self.root.children[row].children[c].text)
            if f > max_freq: max_freq = f
        return True if freq > max_freq else False

    def swap_row(self, freq, row, col):
        c = col
        if self.hor_shift == 'left':
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
        if self.ver_shift == 'top':
            row_start = self.top_row - 1 if col == self.top_col else self.top_row
            row_end = -1
            step = -1
        else:
            row_start = self.top_row + 1 if col == self.top_col else self.top_row
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
        self.reward -= abs(row - to_row) + abs(col - to_col)
        self.update_reward()

    def swap_to_center(self, freq, row, col):
        if row == self.top_row and col == self.top_col: return row, col

        extr = 0;
        extc = 0
        while self.top_row - extr >= 0 and self.top_col - extc >= 0:
            for r in range(self.top_row - extr, self.top_row + extr + (self.rows % 2)):
                if r < 0 or r > self.rows - 1: break
                for c in range(self.top_col - extc, self.top_col + extc + (self.cols % 2)):
                    if c < 0 or c > self.cols - 1: break
                    f = int(self.root.children[r].children[c].text)
                    if f < freq: return r, c
            extc += 1
            extr += 1
            if abs(row - self.top_row) > abs(col - self.top_col):
                if self.top_row - extr == row or self.top_row + extr == row: break
            else:
                if self.top_col - extc == col or self.top_col + extc == col: break
        return row, col

    def adapt_ui(self, instance, freq):
        row, col = self.get_idx_children(instance)
        pos_x = instance.x
        pos_y = instance.y
        anim = Animation(pos=(instance.x, instance.y), t='out_bounce', d=.02)
        if row >= 0 and col >= 0 and (row != self.top_row or col != self.top_col):
            if self.adapt_strategy == AdaptStrategy.Center:
                to_pos_row, to_pos_col = self.swap_to_center(freq, row, col)
                self.shift_from_to(instance, row, col, to_pos_row, to_pos_col)
                row = to_pos_row;
                col = to_pos_col
                pos_x = self.animate_row(instance, to_pos_col)
                pos_y = self.animate_col(instance, to_pos_row)
            elif self.adapt_strategy == AdaptStrategy.Rand:
                to_pos_row = random.randint(0, self.rows - 1)
                to_pos_col = random.randint(0, self.cols - 1)
                self.shift_from_to(instance, row, col, to_pos_row, to_pos_col)
                row = to_pos_row;
                col = to_pos_col
                pos_x = self.animate_row(instance, to_pos_col)
                pos_y = self.animate_col(instance, to_pos_row)
            else:
                # Column adapt
                if self.swap_topcol(freq, row, col):
                    self.shift_from_to(instance, row, col, row, self.top_col)
                    pos_x = self.animate_toprow(instance)
                    col = self.top_col
                else:
                    to_pos_col = self.swap_row(freq, row, col)
                    if col != to_pos_col:
                        self.shift_from_to(instance, row, col, row, to_pos_col)
                        pos_x = self.animate_row(instance, to_pos_col)
                        col = to_pos_col

                # Row adapt
                if self.swap_toprow(freq, row, col):
                    self.shift_from_to(instance, row, self.top_col, self.top_row, self.top_col)
                    pos_y = self.animate_topcol(instance)
                    row = self.top_row
                elif row != self.top_row:
                    to_pos_row = self.swap_col(freq, row, col)
                    if row != to_pos_row:
                        self.shift_from_to(instance, row, col, to_pos_row, col)
                        pos_y = self.animate_col(instance, to_pos_row)
                        row = to_pos_row

        if pos_x != instance.x or pos_y != instance.y:
            if pos_x > 0 and pos_x < Window.width and pos_y > 0 and pos_y < Window.height:
                anim += Animation(pos=(pos_x, pos_y), t='out_bounce', d=.02)
                anim.start(instance)
            self.adapt_frequency += 1
            if not self.emulation:
                self.text_to_display('row=' + str(row) + ', col=' + str(col))
        else:
            self.reward += 1
            self.update_reward()

    def update_reward(self):
        self.reward_lbl.text = 'R: ' + str(self.reward)

    def get_idx_children(self, instance):
        row = -1;
        col = -1
        for child in self.root.children:
            row += 1
            if len(child.children) > 0:
                try:
                    col = child.children.index(instance)
                except:
                    col = -1
                if col >= 0:
                    break
        return row, col

    def animate_toprow(self, instance):
        if self.hor_shift == 'right':
            pos_x = Window.width - instance.width - self.shift_padding
        else:
            pos_x = self.shift_padding
        return pos_x

    def animate_topcol(self, instance):
        if self.ver_shift == 'bottom':
            pos_y = self.shift_padding
        else:
            pos_y = instance.height * (self.rows - 1) + self.shift_padding * self.rows
        return pos_y

    def animate_col(self, instance, row):
        return self.shift_padding * (row + 1) + instance.height * row

    def animate_row(self, instance, col):
        return instance.width * (self.cols - col - 1) + self.shift_spacing * (self.cols - col) + 10

    def show_popup(self, text='', title='Popup Window'):
        popup = Popup(title=title, size_hint=(None, None),
                      size=(Window.width / 2, Window.height / 4))
        layout = BoxLayout(orientation='vertical', padding=10)
        layout.add_widget(Label(text=text))
        layout.add_widget(Button(text='OK', on_press=popup.dismiss))
        popup.content = layout
        popup.open()


if __name__ == '__main__':
    app = MainApp()
    # app.show_popup('The frequency of clicks adapts the interface', 'Info')
    app.run()
