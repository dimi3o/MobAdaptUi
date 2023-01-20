redclr = [1,0,0,1]
grayclr = [.10, .35, .41, .85]
greenclr = [0,1,0,1]
lgreenclr = [0,1,1,1]
blueclr = [0,0,1,1]
purpleclr = [1,0,1,1]
yellowclr = [.9, 1, .01, 1]
favcolors = [redclr, greenclr, blueclr, lgreenclr, purpleclr,yellowclr]
allcolors = [redclr, grayclr, greenclr, blueclr, lgreenclr, purpleclr,yellowclr]

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