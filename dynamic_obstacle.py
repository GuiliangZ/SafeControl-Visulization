import random
import numpy as np
import math

def l2( xy0, xy1 ):
    ox = xy1[0]
    oy = xy1[1]
    dx = xy0[0] - xy1[0]
    dy = xy0[1] - xy1[1]
    dist = math.sqrt( (dx * dx) + (dy * dy) )
    if (xy1[0] < -0.9):
        warp_dx = xy0[0] - (1 + (xy1[0] + 1))
        dist1 = math.sqrt( (warp_dx * warp_dx) + (dy * dy) )
        if (dist1 < dist):
            ox = (1 + (xy1[0] + 1))
            dist = dist1
            #print(f"case1")
    elif (xy1[0] > 0.9):
        warp_dx = xy0[0] - (-1 + (xy1[0] - 1))
        dist1 = math.sqrt( (warp_dx * warp_dx) + (dy * dy) )
        if (dist1 < dist):
            ox = (-1 + (xy1[0] - 1))
            dist = dist1
            #print(f"case2")
    return dist, ox, oy


class Obstacle():

    def __init__(self,
                 a_x, b_x, c_x,  
                 a_y, b_y, c_y, obs_color, t_start):
        self.a_x = a_x / 5
        self.b_x = b_x
        self.c_x = c_x
        self.a_y = a_y / 5
        self.b_y = b_y
        self.c_y = c_y
        self.obs_color = obs_color
        self.t_start = t_start

    @property
    def params(self):
        return { 'a_x':     self.a_x,
                 'b_x':     self.b_x,
                 'c_x':     self.c_x,
                 'a_y':     self.a_y,
                 'b_y':     self.b_y,
                 'c_y':     self.c_y,
                 'obs_color':     self.obs_color,
                 't_start': self.t_start }

    def x(self, t):
        t_shifted = t - self.t_start
        x = ((self.a_x * t_shifted * t_shifted)
             + (self.b_x * t_shifted)
             + self.c_x)
        return x

    def y(self, t):
        t_shifted = t - self.t_start
        y = ((self.a_y * t_shifted * t_shifted)
             + (self.b_y * t_shifted)
             + self.c_y)
        return y

    def v_x(self, t):
        t_shifted = t - self.t_start
        v_x = ((2 * self.a_x * t_shifted) + self.b_x)
        return v_x

    def v_y(self, t):
        t_shifted = t - self.t_start
        v_y = ((2 * self.a_y * t_shifted) + self.b_y)
        return v_y

    def obs_color(self):
        return self.obs_color

FIELD_X_BOUNDS = (-0.95, 0.95)
FIELD_Y_BOUNDS = (-0.95, 1.0)

class ObstacleField(object):
    
    def __init__(self,
                 a_x, b_x, c_x,
                 a_y, b_y, c_y, obs_color, t_start):
        self.a_x = a_x / 5
        self.b_x = b_x
        self.c_x = c_x
        self.a_y = a_y / 5
        self.b_y = b_y
        self.c_y = c_y
        self.obs_color = obs_color
        self.t_start = t_start

    @property
    def params(self):
        return { 'a_x':     self.a_x,
                 'b_x':     self.b_x,
                 'c_x':     self.c_x,
                 'a_y':     self.a_y,
                 'b_y':     self.b_y,
                 'c_y':     self.c_y,
                 'obs_color':     self.obs_color,                 
                 't_start': self.t_start }

    def x(self, t):
        t_shifted = t - self.t_start
        x = ((self.a_x * t_shifted * t_shifted)
             + (self.b_x * t_shifted)
             + self.c_x)
        return x

    def y(self, t):
        t_shifted = t - self.t_start
        y = ((self.a_y * t_shifted * t_shifted)
             + (self.b_y * t_shifted)
             + self.c_y)
        return y

    def v_x(self, t):
        t_shifted = t - self.t_start
        v_x = ((2 * self.a_x * t_shifted) + self.b_x)
        return v_x

    def v_y(self, t):
        t_shifted = t - self.t_start
        v_y = ((2 * self.a_y * t_shifted) + self.b_y)
        return v_y

    def obs_color(self):
        return self.obs_color
    

FIELD_X_BOUNDS = (-0.95, 0.95)
FIELD_Y_BOUNDS = (-0.95, 1.0)

class ObstacleField(object):
    
    def __init__(self,
                 x_bounds = FIELD_X_BOUNDS,
                 y_bounds = FIELD_Y_BOUNDS):

        self.random_init()
        #self.adversarial_init()
        #self.stationary_set_init()
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    def random_init(self):
        obstacles = []
        color_choices = ['yellow', 'gold', 'orange',  'maroon', 'violet', 'magenta', 'purple', 
                                    'navy', 'blue', 'skyblue', 'cyan', 'turquoise', 'lightgreen', 'green', 'darkgreen', 
                                    'chocolate', 'brown']
        obs_color = color_choices.pop()
        # for i in range(100):
        #     obstacles.append(self.random_init_obstacle(t = -100))

        # for i in range(50):
        #     obstacles.append(self.stationary_random_init_obstacle(t = -100))

        # x = 0
        # y = -0.5
        # for i in range(4):
        #     obstacles.append(self.stationary_set_init_obstacle(x, y, t = -100))
        #     x += 0.1
        #     y -= 0.05
        # x = 0
        # y = -0.5
        # for i in range(4):
        #     x -= 0.1
        #     y -= 0.05
        #     obstacles.append(self.stationary_set_init_obstacle(x, y, t = -100))
        # x = 0
        # y = -0.5
        # for i in range(4):
        #     obstacles.append(self.stationary_set_init_obstacle(x, y, t = -100))
        #     x -= 0.05
        #     y += 0.05
        # x = 0
        # y = -0.5
        # for i in range(4):
        #     obstacles.append(self.stationary_set_init_obstacle(x, y, t = -100))
        #     x += 0.05
        #     y += 0.05
        # x = 0.25
        # y = -0.4
        # for i in range(4):
        #     obstacles.append(self.stationary_set_init_obstacle(x, y, t = -100))
        #     x -= 0.05
        #     y += 0.05
        # x = 0.15
        # y = -0.4
        # for i in range(4):
        #     x -= 0.1
        #     y += 0.1
        #     obstacles.append(self.stationary_set_init_obstacle(x, y, t = -100))
        # obstacles.append(self.stationary_set_init_obstacle(x = 0, y = -0.8, t = -100))
        # obstacles.append(self.stationary_set_init_obstacle(x = -0.1, y = -0.5, t = -100))
        # obstacles.append(self.stationary_set_init_obstacle(x = 0.18, y = -0.48, t = -100))  

        obstacles.append(self.designate_init_obstacle(x = 0, y = 0.3, b_x = 0, b_y = 0.005 * -1, a_x = 0, a_y = 0, obs_color = obs_color))
        obs_color = color_choices.pop()
        obstacles.append(self.designate_init_obstacle(x = 0.25, y = 0.3, b_x = -0.007, b_y = 0.007 * -1, a_x = 0, a_y = 0, obs_color = obs_color))        
        obs_color = color_choices.pop()
        obstacles.append(self.designate_init_obstacle(x = -0.25, y = 0.3, b_x = 0.007, b_y = 0.007 * -1, a_x = 0, a_y = 0, obs_color = obs_color))          
        obs_color = color_choices.pop()
        obstacles.append(self.designate_init_obstacle(x = 0.07, y = 0.4, b_x = 0, b_y = 0.006 * -1, a_x = 0, a_y = 0, obs_color = obs_color))    
        obs_color = color_choices.pop()
        obstacles.append(self.designate_init_obstacle(x = -0.07, y = 0.4, b_x = 0, b_y = 0.006 * -1, a_x = 0, a_y = 0, obs_color = obs_color))
        obs_color = color_choices.pop()
        obstacles.append(self.designate_init_obstacle(x = 0.17, y = 0.35, b_x = -0.006, b_y = 0.006 * -1, a_x = 0, a_y = 0, obs_color = obs_color))    
        obs_color = color_choices.pop()
        obstacles.append(self.designate_init_obstacle(x = -0.17, y = 0.35, b_x = 0.006, b_y = 0.006 * -1, a_x = 0, a_y = 0, obs_color = obs_color))      
        obs_color = color_choices.pop()
        obstacles.append(self.designate_init_obstacle(x = 0, y = -0.2, b_x = 0, b_y = 0.005 * 1, a_x = 0, a_y = 0, obs_color = obs_color))
        obs_color = color_choices.pop()
        obstacles.append(self.designate_init_obstacle(x = 0.25, y = -0.1, b_x = -0.005, b_y = 0.005 * 1, a_x = 0, a_y = 0, obs_color = obs_color))        
        obs_color = color_choices.pop()
        obstacles.append(self.designate_init_obstacle(x = -0.25, y = -0.1, b_x = 0.005, b_y = 0.005 * 1, a_x = 0, a_y = 0, obs_color = obs_color))  
        obs_color = color_choices.pop()

        obstacles.append(self.designate_init_obstacle(x = 0.3, y = 0.1, b_x = -0.008, b_y = 0, a_x = 0, a_y = 0, obs_color = obs_color))
        obs_color = color_choices.pop()
        obstacles.append(self.designate_init_obstacle(x = -0.3, y = 0.1, b_x = 0.008, b_y = 0, a_x = 0, a_y = 0, obs_color = obs_color))
        obs_color = color_choices.pop()
        # obstacles.append(self.designate_init_obstacle(x = 0.3, y = 0.15, b_x = -0.008, b_y = 0, a_x = 0, a_y = 0))
        # obstacles.append(self.designate_init_obstacle(x = -0.3, y = 0.15, b_x = 0.008, b_y = 0, a_x = 0, a_y = 0))
        # obstacles.append(self.designate_init_obstacle(x = 0.3, y = 0.05, b_x = -0.008, b_y = 0, a_x = 0, a_y = 0))
        # obstacles.append(self.designate_init_obstacle(x = -0.3, y = 0.05, b_x = 0.008, b_y = 0, a_x = 0, a_y = 0))

        obstacles.append(self.designate_init_obstacle(x = 0.2, y = -0.2, b_x = -0.006, b_y = 0.006, a_x = 0, a_y = 0, obs_color = obs_color))
        obs_color = color_choices.pop()
        obstacles.append(self.designate_init_obstacle(x = -0.2, y = -0.2, b_x = 0.006, b_y = 0.006, a_x = 0, a_y = 0, obs_color = obs_color))  

        # obstacles.append(self.designate_init_obstacle(x = 0.1, y = -0.2, b_x = -0.0025, b_y = 0.005, a_x = 0, a_y = 0))
        # obstacles.append(self.designate_init_obstacle(x = -0.1, y = -0.2, b_x = 0.0025, b_y = 0.005, a_x = 0, a_y = 0)) 
        self.obstacles = obstacles
        return


    
    def designate_init_obstacle(self, x, y, b_x, b_y, a_x, a_y, obs_color, t=-3, vehicle_x = 0, vehicle_y = -1, min_dist = 0.1):
        dist = math.sqrt((vehicle_x-x)**2 + (vehicle_y-y)**2)


        while (dist < min_dist):
            x = random.uniform(FIELD_X_BOUNDS[0],FIELD_X_BOUNDS[1])
            y = random.uniform(FIELD_Y_BOUNDS[0],FIELD_Y_BOUNDS[1])
            dist = math.sqrt((vehicle_x-x)**2 + (vehicle_y-y)**2)
        return Obstacle(a_x, b_x, x, a_y, b_y, y, obs_color, t)


    def random_init_obstacle(self, t, vehicle_x = 0, vehicle_y = -1, min_dist = 0.1):
        dist = -1
        x = y = -1
        while (dist < min_dist):
            x = random.uniform(FIELD_X_BOUNDS[0],FIELD_X_BOUNDS[1])
            y = random.uniform(FIELD_Y_BOUNDS[0],FIELD_Y_BOUNDS[1])
            dist = math.sqrt((vehicle_x-x)**2 + (vehicle_y-y)**2)
        b_x = random.uniform(1e-3, 1e-2) * random.choice([1,-1])
        b_y = random.uniform(1e-3, 1e-2) * random.choice([1,-1])
        a_x = random.uniform(5e-6, 1e-4) * random.choice([1,-1])
        a_y = random.uniform(5e-6, 1e-4) * random.choice([1,-1])
        color_choices = ['yellow', 'gold', 'orange',  'maroon', 'violet', 'magenta', 'purple', 
                                    'navy', 'blue', 'skyblue', 'cyan', 'turquoise', 'lightgreen', 'green', 'darkgreen', 
                                    'chocolate', 'brown']
        obs_color = random.choice(color_choices)
        return Obstacle(a_x, b_x, x, a_y, b_y, y, obs_color, t)

    def stationary_random_init_obstacle(self, t, vehicle_x = 0, vehicle_y = -1, min_dist = 0.1):
        dist = -1
        x = y = -1
        while (dist < min_dist):
            x = random.uniform(FIELD_X_BOUNDS[0],FIELD_X_BOUNDS[1])
            y = random.uniform(FIELD_Y_BOUNDS[0],FIELD_Y_BOUNDS[1])
            dist = math.sqrt((vehicle_x-x)**2 + (vehicle_y-y)**2)
        b_x = 0
        b_y = 0
        a_x = 0
        a_y = 0
        color_choices = ['yellow', 'gold', 'orange',  'maroon', 'violet', 'magenta', 'purple', 
                                    'navy', 'blue', 'skyblue', 'cyan', 'turquoise', 'lightgreen', 'green', 'darkgreen', 
                                    'chocolate', 'brown']
        obs_color = random.choice(color_choices)
        return Obstacle(a_x, b_x, x, a_y, b_y, y, obs_color, t)

    def stationary_set_init_obstacle(self, x, y, t, vehicle_x = 0, vehicle_y = -1, min_dist = 0.1):
        dist = math.sqrt((vehicle_x-x)**2 + (vehicle_y-y)**2)
        while (dist < min_dist):
            x = random.uniform(FIELD_X_BOUNDS[0],FIELD_X_BOUNDS[1])
            y = random.uniform(FIELD_Y_BOUNDS[0],FIELD_Y_BOUNDS[1])
            dist = math.sqrt((vehicle_x-x)**2 + (vehicle_y-y)**2)
        b_x = 0
        b_y = 0
        a_x = 0
        a_y = 0
        color_choices = ['yellow', 'gold', 'orange',  'maroon', 'violet', 'magenta', 'purple', 
                                    'navy', 'blue', 'skyblue', 'cyan', 'turquoise', 'lightgreen', 'green', 'darkgreen', 
                                    'chocolate', 'brown']
        obs_color = random.choice(color_choices) 
        return Obstacle(a_x, b_x, x, a_y, b_y, y, obs_color, t)




    # def adversarial_init(self, robot_posX, robot_posY):
    #     obstacles = []
    #     for i in range(1):
    #         obstacles.append(self.adversarial_init_obstacle(robot_posX, robot_posY,t = -100))
    #     self.obstacles = obstacles
    #     return 

    # def adversarial_init_obstacle(self, t, vehicle_x = 0, vehicle_y = -1, min_dist = 0.1):
    #     dist = -1
    #     x = y = -1
    #     while (dist < min_dist):
    #         x = random.uniform(FIELD_X_BOUNDS[0],FIELD_X_BOUNDS[1])
    #         y = random.uniform(FIELD_Y_BOUNDS[0],FIELD_Y_BOUNDS[1])
    #         dist = math.sqrt((vehicle_x-x)**2 + (vehicle_y-y)**2)
        
    #     b_x = random.uniform(1e-3, 1e-2) * random.choice([1,-1])
    #     b_y = random.uniform(1e-3, 1e-2) * random.choice([1,-1])

    #     a_x = random.uniform(5e-6, 1e-4) * random.choice([1,-1])
    #     a_y = random.uniform(5e-6, 1e-4) * random.choice([1,-1])
    #     return Obstacle( a_x, b_x, x, a_y, b_y, y, t)

    # def adversarial_obs_control(self, robot_posX, robot_posY, ):
    #     robot_pos = [[robot_posX],[robot_posY]]
    #     pos = np.array([[x],[y]])
    #     vel = np.array([[b_x],[b_y]])
    #     kp = np.array([[2, 0],[0, 1]])
    #     kd = np.array([[2, 0],[0, 1]])
    #     acc = np.dot(kp, (robot_pos-pos))-np.dot(kd,(vel))
    #     a_x = acc[0][0]
    #     a_y = acc[1][0]
    #     return a_x, a_y

    #def adversarial_steer(self, a_x, a_y):





    def obstacle_locations(self, t, vehicle_x, vehicle_y, min_dist):
        """
        Returns (i, x, y) tuples indicating that the i-th obstacle is at location (x,y).
        """
        locs = []
        for i, a in enumerate(self.obstacles):
            #print(a.obs_color)
            if self.x_bounds[0] <= a.x(t) <= self.x_bounds[1] and self.y_bounds[0] <= a.y(t) <= self.y_bounds[1]:
                locs.append((i, a.x(t), a.y(t), a.obs_color))
            else:
                self.obstacles[i] = self.random_init_obstacle(t, vehicle_x, vehicle_y, min_dist)
                locs.append((i, a.x(t), a.y(t), a.obs_color))
        return locs

    def all_obstacle_states(self, t, vehicle_x, vehicle_y, min_dist):
        """
        Returns (i, x, y) tuples indicating that the i-th obstacle is at location (x,y).
        """
        states = []
        for i, a in enumerate(self.obstacles):
            if self.x_bounds[0] <= a.x(t) <= self.x_bounds[1] and self.y_bounds[0] <= a.y(t) <= self.y_bounds[1]:
                states.append(( a.x(t), a.y(t),a.v_x(t), a.v_y(t), a.a_x, a.a_y))
            else:
                self.obstacles[i] = self.random_init_obstacle(t, vehicle_x, vehicle_y, min_dist)
                states.append(( a.x(t), a.y(t),a.v_x(t), a.v_y(t), a.a_x, a.a_y))
        return states

    def unsafe_obstacle_locations(self, t, cx, cy, min_dist):    
        locs =  [ (i, a.x(t), a.y(t), a.v_x(t), a.v_y(t), a.a_x, a.a_y, a.obs_color)
                  for i,a in enumerate(self.obstacles)]
        unsafe_obstacles = [] 
        for i,x,y,x_v,y_v,x_a,y_a, obs_color  in locs:
            if self.x_bounds[0] <= x <= self.x_bounds[1] and self.y_bounds[0] <= y <= self.y_bounds[1]:
                dist, ox, oy = l2([cx,cy], [x,y])
                if dist < min_dist:
                    unsafe_obstacles.append([i,(ox,oy,x_v,y_v,x_a,y_a), obs_color])
        return  unsafe_obstacles
