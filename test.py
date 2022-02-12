import random
from kivywidgets import Widgets
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scatter import Scatter

def get_hor_boxlayout(orientation='horizontal', padding=10, spacing=10):
    return BoxLayout(orientation=orientation, padding=padding, spacing=spacing)

class TutorialApp(App):
    def build(self):
        self.root = BoxLayout(orientation='vertical', padding=0)
        self.root.add_widget(Button(text='rebuild of layout', size_hint_y=None, height='30dp', on_press=self.rebuild_btn_click))
        self.widgets_layout = BoxLayout(orientation='vertical')
        self.rebuild_btn_click(self)
        self.root.add_widget(self.widgets_layout)
        return self.root

    def rebuild_btn_click(self, instance):
        self.widgets_layout.clear_widgets()
        for i in range(5):
            hor = get_hor_boxlayout('horizontal', 0, 0)
            for j in range(5):
                s = Scatter(do_rotation=False, do_scale=False, auto_bring_to_front=False)
                hor.add_widget(s)
                s.add_widget(Widgets.get_random_widget())
            self.widgets_layout.add_widget(hor)
        # for j in range(5):
        #     b = BoxLayout(orientation='horizontal', padding=0)
        #     for i in range(5):
        #         f = FloatLayout()
        #         s = Scatter(do_rotation=False, do_scale=False, auto_bring_to_front=False)
        #         f.add_widget(s)
        #         w = Widgets.get_random_widget()
        #         s.add_widget(w)
        #         b.add_widget(f)
        #     b0.add_widget(b)

if __name__ == "__main__":
    TutorialApp().run()