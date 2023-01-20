from kivy.app import App
from kivy_garden.mapview import MapMarkerPopup, MapView
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder

class City():
    def __init__(self, name, lat, lon, info):
        self.name = name
        self.lon = lon
        self.lat = lat
        self.info = info

moscow = City('Moscow', 55.75371378848573, 37.62416076660156, 'инфа о москве')
saintp = City('Saint P', 59.9386, 30.3141, 'инфа о питере')
sochi = City('Sochi', 43.5992, 39.7257, 'инфа о сочи')
orel = City('Orel', 52.9651, 36.0785, 'Население Орла 318 642')
orenburg = City('Orenburg', 51.76665, 55.10045, 'Население Оренбурга 555 420')
cities = [moscow, saintp, sochi, orel, orenburg]

class MyLabel(Label):
    def on_size(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0, 0.05, 0.64, 0.25)
        Rectangle(pos=self.pos, size=self.size)


class MainScr(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vl = BoxLayout(orientation='vertical', pos_hint={'center_x': 0.5, 'center_y': 0.5})
        dropdown = DropDown()
        for i in range(len(cities)):
            btn = Button(text=cities[i].name, size_hint_y=None, height=40)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)
        self.mainbutton = Button(text='Выберите город', size_hint=(1, None), pos=(350, 300))
        self.mainbutton.bind(on_release=dropdown.open)
        dropdown.bind(on_select=self.change_city)
        vl.add_widget(self.mainbutton)
        self.mymap = MapView(lat=55.75371378848573, lon=37.62416076660156, zoom=10, double_tap_zoom=True)
        vl.add_widget(self.mymap)
        for i in cities:
            marker = MapMarkerPopup(lat=i.lat, lon=i.lon, source="data/icons/bug1.png")
            marker.add_widget(MyLabel(text=i.info, color='black'))
            self.mymap.add_widget(marker)
        self.add_widget(vl)

    def change_city(self, instance, x):
        setattr(self.mainbutton, 'text', x)
        for i in cities:
            if i.name == x:
                self.mymap.center_on(i.lat, i.lon)
                self.mymap.zoom = 10

class MainApp(App):
    def build(self):
        self.title = 'Map Demo Page'
        sm = ScreenManager()
        sm.add_widget(MainScr(name='main'))
        return sm

MainApp().run()
