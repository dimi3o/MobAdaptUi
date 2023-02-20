from enum import Enum
from kivy.graphics import Color

redclr = [1,0,0,1]
grayclr = [.10, .35, .41, .85]
greenclr = [0,1,0,1]
lgreenclr = [0,1,1,1]
blueclr = [0,0,1,1]
lblueclr = [0.12941176470588237, 0.5882352941176471, 0.9529411764705882, 1]
purpleclr = [1,0,1,1]
yellowclr = [.9, 1, .01, 1]
favcolors = [redclr, greenclr, blueclr, lgreenclr, purpleclr,yellowclr]
allcolors = [redclr, grayclr, greenclr, blueclr, lgreenclr, purpleclr,yellowclr]
LightGrayColor = Color(0.827, 0.827, 0.827, 1.)

class MyColors(Enum):
    Red = [1,0,0,1]
    Gray = [.10, .35, .41, .85]
    Green = [0,1,0,1]
    LightGreen = [0,1,1,1]
    Blue = [0,0,1,1]
    LightBlue = [0.12941176470588237, 0.5882352941176471, 0.9529411764705882, 1]
    Purple = [1,0,1,1]
    Yellow = [.9, 1, .01, 1]

    @staticmethod
    def get_textcolor(while_flag=False):
        return [0, 0, 0, 1] if while_flag else [1, 1, 1, 1]

# somecolors = []
# x = float(.01); y = float(.01); z = float(.01)
# while (x <= 1.):
#     while (y <= 1.):
#         while (z <= 1.):
#             clr = [x,y,z,1]
#             somecolors.append(clr)
#             z += .1
#         y += .1
#     x += .1