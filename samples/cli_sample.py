from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
text_proc = ""

class MainApp(App):
    def build(self):
        layout = BoxLayout(orientation="vertical",padding=10)#,size_hint_y=None)
        #Main Buttons
        HorBL = BoxLayout(orientation="horizontal",padding=0,size_hint=(None, None))
        options = ['Process','Step','Clear']
        functions = [self.on_press_button_proc,self.on_press_button_step,self.on_clear_display]
        for i in range(3):
            btn = Button(text=options[i],size=(200, 50), size_hint=(None, None))
            btn.bind(on_press=functions[i])
            HorBL.add_widget(btn)
        layout.add_widget(HorBL)
        self.display = Label(
            text='Minimalistic Display',
            font_size='12sp',
            size_hint=(1, 1),halign="left",valign="top")
        self.display.bind(size=self.display.setter('text_size'))
        self.display.text = text_proc
        layout.add_widget(self.display)
        self.root = layout
        return self.root

    def on_press_button_proc(self, instance):
        self.add_text("process..")

    def on_press_button_step(self, instance):
        self.add_text("step..")

    def on_clear_display(self, instance):
        self.display.text = ""

    def add_text(self,text=""):
        self.display.text += "\n"+text

if __name__ == "__main__":
    app = MainApp()
    app.run()