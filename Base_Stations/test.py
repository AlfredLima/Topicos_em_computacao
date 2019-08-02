import pygame
import sys
from pygame.locals import *
import math


def createPoints2Poly(side, center, radius):
    angle = 2*math.pi/side
    points = []
    for i in range(side):
        x = center[0] + radius*math.cos((i+0.5)*angle)
        y = center[1] + radius*math.sin((i+0.5)*angle)
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
windowSurface.fill(WHITE)
# desenha um poligono verde na superficie
# pygame.draw.polygon(windowSurface, GREEN, ((146, 0), (291, 106), (236, 277),
#                                            (56, 277), (0, 106)))

pygame.draw.polygon(windowSurface, GREEN,
                    createPoints2Poly(6, (625/2, 625/2), 100))

# desenha a janela na tela
pygame.display.update()
# roda o loop do jogo
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
