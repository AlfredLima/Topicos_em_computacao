from shapely.geometry import Polygon
import pygame
import sys
from pygame.locals import *
import math


def createPoints2Poly(side, center, radius):
    angle = 2*math.pi/side
    points = []
    for i in range(side):
        x = center[0] + radius*math.cos((i)*angle)
        y = center[1] + radius*math.sin((i)*angle)
        points.append((x, y))

    return tuple(points)


# inicia o pygame
pygame.init()

# inicia a janela
windowSurface = pygame.display.set_mode((625, 625), 0, 32)
# inicia as cores utilizadas
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
# inicia as fontes
basicFont = pygame.font.SysFont(None, 48)
# desenha o fundo branco
windowSurface.fill(GREEN)
# desenha um poligono verde na superficie

city = [(0, 0), (0, 625), (625, 625), (625, 0)]

BSc = [(219, 287), (219, 393), (312, 234), (312, 234),
       (312, 340), (312, 448), (405, 287), (405, 393)]

Bs = []
for B in BSc:
    Bs.append(createPoints2Poly(4, B, 10))

for B in Bs:
    pygame.draw.polygon(windowSurface, (127, 127, 127),
                        B)

for i in range(len(BSc)):
    Bs[i] = Polygon(Bs[i])

print(Bs[0].intersection(Bs[1]).area)

# desenha a janela na tela
pygame.display.update()
# roda o loop do jogo
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
