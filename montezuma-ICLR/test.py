from image_processing import *
import pickle
rec = Recognizer()
from matplotlib import pyplot as plt

C = [(79, 101),
 (83, 120),
 (128, 159),
 (110, 118),
 (136, 127),
 (78, 81),
 (120, 81),
 (33, 82),
 (35.0, 171.0),
 (25,129)]

O = [(12,120),(29,84),(127,83)]
G = C + O

img = rec.base_img
color = (0,0,255)
for g in C:
	g = (int(g[0]),int(g[1]))
	img = draw_circle(img, g, 4, color)

color = (255,100,0)
for g in O:
	g = (int(g[0]),int(g[1]))
	img = draw_circle(img, g, 4, color)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()








