import pygame as pg
import sys
import random
pg.init()

#maze displayer
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
ASTAR_RED = (255, 165, 143)
BFS_GREEN = (87, 255, 140)
DFS_PINK = (251, 145, 255)
DIJKSTRA_BLUE = (150, 197, 255)
GREEN = (0,255,0)
GRAY = (150,150,150)
AQUA = (26, 138, 132)
PURPLE = (93,0,255)
g_width = 10
g_height = 10
DFS_pf = False

#maze generator
wall = "1"
cell = "0"
unvisited = "u"
height = 50
width = 50
maze = []
walls = [] #initialise walls array


#user input
myFont = pg.font.SysFont("verdana",50)

