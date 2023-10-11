#imports
#import random
from constants import *
import numpy as np
import heapq

FPS = 1000

#functions/procedures
#outputs readable format of maze
def output_maze(maze):
    for i in range(0, height):
        for j in range(0, width):
            print(str(maze[i][j]), end=" ")
        print('\n')

#calculates the number of surrounding cells to a specific wall
def surrounding_cells(r_wall):
    s_cells = 0
    if (maze[r_wall[0]-1][r_wall[1]] == "0"):
        s_cells += 1
    if (maze[r_wall[0]+1][r_wall[1]] == "0"):
        s_cells += 1
    if (maze[r_wall[0]][r_wall[1]-1] == "0"):
        s_cells +=1
    if (maze[r_wall[0]][r_wall[1]+1] == "0"):
        s_cells += 1

    return s_cells

#draw rectangle
def draw_rect(color,a,b,fps2):
    """(colour, x padding, y padding"""
    pg.draw.rect(WINDOW,color,[a*g_width,b*g_height,g_width,g_height])
    pg.display.update()
    fpsClock.tick(fps2)
    pass
        
#upper wall check
def upper(maze,r_wall):
    if r_wall[0] != 0:
        if maze[r_wall[0]-1][r_wall[1]] != "0":
            maze[r_wall[0]-1][r_wall[1]] = "1"
            
        if [r_wall[0]-1,r_wall[1]] not in walls:
            walls.append([r_wall[0]-1,r_wall[1]]); draw_rect(BLACK,r_wall[1],r_wall[0]-1,FPS)

#lower wall check
def lower(maze,r_wall):
    if r_wall[0] != height-1: #lower
        if maze[r_wall[0]+1][r_wall[1]] != "0":
            maze[r_wall[0]+1][r_wall[1]] = "1"
           
        if [r_wall[0]+1,r_wall[1]] not in walls:
            walls.append([r_wall[0]+1,r_wall[1]]); draw_rect(BLACK,r_wall[1],r_wall[0]+1,FPS)

#right most wall check
def right(maze,r_wall):
    if r_wall[1] != width-1: #right most
        if maze[r_wall[0]][r_wall[1]+1] != "0":
            maze[r_wall[0]][r_wall[1]+1] = "1"

        if [r_wall[0],r_wall[1]+1] not in walls:
            walls.append([r_wall[0],r_wall[1]+1]); draw_rect(BLACK,r_wall[1]+1,r_wall[0],FPS)
            
#left most wall check
def left(maze,r_wall):
    if r_wall[1] != 0: #left most
        if maze[r_wall[0]][r_wall[1]-1] != "0":
            maze[r_wall[0]][r_wall[1]-1] = "1"
           
        if [r_wall[0],r_wall[1]-1] not in walls:
            walls.append([r_wall[0],r_wall[1]-1]); draw_rect(BLACK,r_wall[1]-1,r_wall[0],FPS)

#delete wall
def delete(r_wall):
    for wall in walls:
        if (wall[0] == r_wall[0] and
            wall[1] == r_wall[1]):
            walls.remove(wall)
            
#is the randomly selected wall a left wall?
def is_left(r_wall):
    if r_wall[1] != 0:
        if (maze[r_wall[0]][r_wall[1]-1] == "u" and
            maze[r_wall[0]][r_wall[1]+1] == "0"):
            return True

#is the randomly selected wall an upper wall?
def is_upper(r_wall):
    if r_wall[0] != 0:
        if (maze[r_wall[0]-1][r_wall[1]] == "u" and
            maze[r_wall[0]+1][r_wall[1]] == "0"):
            return True

#is the randomly selected wall a lower wall?
def is_lower(r_wall):
    if r_wall[0] != height-1:
        if (maze[r_wall[0]+1][r_wall[1]] == "u" and
            maze[r_wall[0]-1][r_wall[1]] == "0"):
            return True
        
#is the randomly selected wall a right wall?
def is_right(r_wall):
    if r_wall[1] != width-1:
        if (maze[r_wall[0]][r_wall[1]+1] == "u" and
            maze[r_wall[0]][r_wall[1]-1] == "0"):
            return True

#is the number of surrounding cells to the randomly selected left wall less than 2?
def sc_lt_2(r_wall):
    s_cells = surrounding_cells(r_wall)
    if s_cells < 2:
        maze[r_wall[0]][r_wall[1]] = "0"; draw_rect(WHITE,r_wall[1],r_wall[0],FPS)
        return True

######################################################################################################################################################################

#Generate maze

######################################################################################################################################################################

def Generate():
    global ei,ej
    #set all cells in grid to unvisited
    for i in range(0, height):
        line = []
        for j in range(0, width):
                line.append("u")
        maze.append(line)

    #random starting point that isn't in one of the four corners
    s_height = int(random.random()*height)
    s_width = int(random.random()*width)
    
    if (s_height == 0):
        s_height += 1
            
    if (s_height == height-1): 
        s_height -= 1
            
    if (s_width == 0): 
        s_width += 1
            
    if (s_width == width-1):
        s_width -= 1	

    #set the randomised start point as a cell "0"
    maze[s_height][s_width] = "0"; draw_rect(WHITE,s_width,s_height,FPS)
    walls.append([s_height-1, s_width])
    walls.append([s_height, s_width-1])
    walls.append([s_height, s_width+1])
    walls.append([s_height+1, s_width])
    
    #set these surrouding walls ^ as walls
    maze[s_height-1][s_width] = "1"; draw_rect(BLACK,s_width,s_height-1,FPS)
    maze[s_height][s_width-1] = "1"; draw_rect(BLACK,s_width-1,s_height,FPS)
    maze[s_height][s_width+1] = "1"; draw_rect(BLACK,s_width+1,s_height,FPS)
    maze[s_height+1][s_width] = "1"; draw_rect(BLACK,s_width,s_height+1,FPS)
    
    while walls:
        r_wall = walls[int(random.random()*len(walls))-1]
        
        if is_left(r_wall):
            if sc_lt_2(r_wall):
                upper(maze,r_wall); lower(maze,r_wall); left(maze,r_wall)
            delete(r_wall)     
            continue

        if is_upper(r_wall):
            if sc_lt_2(r_wall):
                upper(maze,r_wall); left(maze,r_wall); right(maze,r_wall)
            delete(r_wall)    
            continue

        if is_lower(r_wall):
            if sc_lt_2(r_wall):
                lower(maze,r_wall); left(maze,r_wall); right(maze,r_wall)
            delete(r_wall)
            continue

        if is_right(r_wall):
            if sc_lt_2(r_wall):
                right(maze,r_wall); lower(maze,r_wall); upper(maze,r_wall)
            delete(r_wall)
            continue
        
        #remove walls from walls anyway
        delete(r_wall)   

    #set all unvisited cells to walls
    for i in range(0,height):
        for j in range(0,width):
            if maze[i][j] == "u":
                maze[i][j] = "1"; draw_rect(BLACK,j,i,FPS)

    #set maze entrance and exit
    for i in range(0,width):
        if maze[1][i] == "0":
            global start
            start = 0,i; si,sj = 0,i
            maze[0][i] = "0"; draw_rect(GREEN,i,0,FPS)         
            break

    for i in range(width-1,0,-1):
        if maze[height-2][i] == "0":
            global end
            end = height-1,i; ei,ej = height-1,i
            maze[height-1][i] = "0"; draw_rect(RED,i,height-1,FPS)
            break
    return (si,sj)

######################################################################################################################################################################

#Breadth First Search

######################################################################################################################################################################

def make_step(k,color):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == k:
                if i>0 and matrix[i-1][j] == 0 and a[i-1][j] == 0:
                    matrix[i-1][j] = k + 1; draw_rect(color,j,i-1,60)
                    
                if j>0 and matrix[i][j-1] == 0 and a[i][j-1] == 0:
                    matrix[i][j-1] = k + 1; draw_rect(color,j-1,i,60)
                    
                if i<len(matrix)-1 and matrix[i+1][j] == 0 and a[i+1][j] == 0:
                    matrix[i+1][j] = k + 1; draw_rect(color,j,i+1,60)
                    
                if j<len(matrix[i])-1 and matrix[i][j+1] == 0 and a[i][j+1] == 0:
                    matrix[i][j+1] = k + 1; draw_rect(color,j+1,i,60)

def BFS(color):
    for i in range(len(a)):
        matrix.append([])
        for j in range(len(a[i])):
            matrix[-1].append(0) #fill with 0's
            
    #place start point
    i,j = start
    matrix[i][j] = 1; draw_rect(AQUA,j,i,60)

    k = 0
    while matrix[end[0]][end[1]] == 0:
        k += 1
        make_step(k,color)

    i,j = end
    k = matrix[i][j]
    
    #find path through maze, and append to path array
    #append the end point of the maze (start point of path backtracking)
    BFS_pf = [(i,j)]
    while k > 1:
        if i > 0 and matrix[i - 1][j] == k-1:
            i, j = i-1, j
            BFS_pf.append((i, j))
            k-=1
        elif j > 0 and matrix[i][j - 1] == k-1:
            i, j = i, j-1
            BFS_pf.append((i, j))
            k-=1
        elif i < len(matrix) - 1 and matrix[i + 1][j] == k-1:
            i, j = i+1, j
            BFS_pf.append((i, j))
            k-=1
        elif j < len(matrix[i]) - 1 and matrix[i][j + 1] == k-1:
            i, j = i, j+1
            BFS_pf.append((i, j))
            k -= 1
    
    #draw path by backtracking
    for i in range(len(BFS_pf)):
        draw_rect(AQUA,BFS_pf[i][1],BFS_pf[i][0],60)

######################################################################################################################################################################

#Depth First Search Recursive

######################################################################################################################################################################

def DFS(i,j):
    global ei,ej,a,DFS_pf
    if i < 0 or j < 0 or i > len(a)-1 or j > len(a[0])-1:
        return
    if (i,j) in path or a[i][j] > 0:
        return
    path.append((i,j))
    
    #mark visited cells as 2
    #if this statment is not here, because the function
    #is recursive, after finding the path, the colours
    #will still be drawn onto the grid, which is what we don't want
    if DFS_pf != True:
        a[i][j] = 2; draw_rect(DFS_PINK,j,i,60)
        
    #path is found 
    if (i,j) == (ei,ej):
        path.pop()
        DFS_pf = True
        
        #draw path by backtracking
        for i in range(len(path)):
            draw_rect(AQUA,path[i][1],path[i][0],60)
        return
    else:
        #upper check
        DFS(i-1,j)
        #lower check
        DFS(i+1,j)
        #right check
        DFS(i,j+1)
        #left check
        DFS(i,j-1)
    path.pop()
    return

######################################################################################################################################################################

#A star algorithm

######################################################################################################################################################################

#heuristic function
def h(a,b):
    """find manhattan distance between two points (a,b)"""
    (x1,y1) = a
    (x2,y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def astar(maze, start, end):
    #neighours list
    #list of possible neighbours
    neighbours = [(0,1),(0,-1),(1,0),(-1,0)]

    #positions that do not need considering again
    close_set = set()

    #route paths taken in each iteration, parent nodes
    parents = {}

    #Dictionary to contain g value
    g = {start:0}

    #Dictionary to contain f value
    f = {start:h(start, end)}

    #all positions that are being considered
    open_set = []

    #push start node and f value of start node to the open set
    heapq.heappush(open_set, (f[start], start))

 
    while open_set:
        draw_rect(GREEN,start[1],start[0],FPS)
        draw_rect(RED,end[1],end[0],FPS)
        #c_node = current node
        #find position of neighbour with smallest f value
        c_node = heapq.heappop(open_set)[1]; draw_rect(PURPLE, c_node[1], c_node[0],60)

        #check if the current node is at the end node
        #if so, draw out the shortest path and return it
        if c_node == end:
            path = []
            while c_node in parents:
                path.append(c_node)
                c_node = parents[c_node]
        
            #draw path 
            for i in range(len(path)):
                draw_rect(AQUA,path[i][1],path[i][0],60)

            print(path)
            return path

        #if not, add current node to closed set
        close_set.add(c_node); draw_rect(ASTAR_RED, c_node[1], c_node[0],60)

        #calculate g value for all possible neighbours
        for i, j in neighbours:
            neighbour = c_node[0] + i, c_node[1] + j
            #not confirmed g value therefore, indefinite
            indefinite_g = g[c_node] + h(c_node, neighbour)

            #if neighbour is outside the maze, ignore and contine the loop
            if 0 <= neighbour[0] < maze.shape[0]:
                if 0 <= neighbour[1] < maze.shape[1]:                
                    if maze[neighbour[0]][neighbour[1]] == 1: continue
                else: continue
            else: continue
 
            if neighbour in close_set and indefinite_g >= g.get(neighbour, 0):
                continue
 
            #update set
            if indefinite_g < g.get(neighbour, 0) or neighbour not in [i[1]for i in open_set]:
                parents[neighbour] = c_node
                g[neighbour] = indefinite_g
                f[neighbour] = indefinite_g + h(neighbour, end)
                heapq.heappush(open_set, (f[neighbour], neighbour))
 
    
    return False


######################################################################################################################################################################

#main procedure

######################################################################################################################################################################

def main():
    alg = int(input("What algorithm do you want to visualise?\nBFS (1)\nDFS (2)\nA* (3)\nDijkstra (4)\n"))
    global WINDOW,a,matrix,path,fpsClock,times
    WINDOW = pg.display.set_mode([(width*g_width),(height*g_height)])
    WINDOW.fill(GRAY)
    fpsClock = pg.time.Clock()
    
    running = True
    done = False

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        while not done:
            if alg == 1: #bfs
                Generate()
                a = [list(map(int,i)) for i in maze]
                matrix = []
                BFS(BFS_GREEN)
            elif alg == 2: #dfs
                si,sj = Generate()
                a = [list(map(int,i)) for i in maze]
                path = []
                DFS(si,sj)
            elif alg == 3: #a*
                Generate()
                a = [list(map(int,i)) for i in maze]
                a = np.array(a)
                astar(a,start,end)
            elif alg == 4: #dijkstra
                #since unweighted maze, BFS = Dijkstra
                Generate()
                a = [list(map(int,i)) for i in maze]
                matrix = []
                BFS(DIJKSTRA_BLUE)
            
            #redraw start and end point
            draw_rect(GREEN,start[1],start[0],FPS)
            draw_rect(RED,end[1],end[0],FPS)

            done = True
    pg.quit()
