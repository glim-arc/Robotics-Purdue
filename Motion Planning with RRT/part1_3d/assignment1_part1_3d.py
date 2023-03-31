from __future__ import division
import pybullet as p
import pybullet_data
import numpy as np
import time
import argparse
import math
import os
import sys

UR5_JOINT_INDICES = [0, 1, 2]


def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)


def draw_sphere_marker(position, radius, color):
   vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
   marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
   return marker_id


def remove_marker(marker_id):
   p.removeBody(marker_id)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--birrt', action='store_true', default=False)
    parser.add_argument('--smoothing', action='store_true', default=False)
    args = parser.parse_args()
    return args

#your implementation starts here
#refer to the handout about what the these functions do and their return type
###############################################################################
class RRT_Node:
    def __init__(self, conf):
        self.conf = conf
        self.parent = None
        self.children = []

    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)

def sample_conf(conf_space):
    idx = np.random.choice(len(conf_space))
    s_conf = conf_space[idx]

    gamma = 0.23

    if abs(goal_conf[0] - s_conf[0]) <= gamma and abs(goal_conf[1] - s_conf[1]) <= gamma and abs(goal_conf[2] - s_conf[2]) <= gamma:
        return s_conf , True, idx
    
    return s_conf , False, idx


def find_nearest(rand_node, node_list):
    min_node = None
    mindist = float('inf')

    for cur in node_list:
        curdist = abs(rand_node.conf[0] - cur.conf[0]) + abs(rand_node.conf[1] - cur.conf[1])

        if mindist > curdist:
            mindist = curdist
            min_node = cur

    return min_node
        
def steer_to(rand_node, nearest_node):
    dist = abs(rand_node.conf[0] - nearest_node.conf[0]) + abs(rand_node.conf[1] - nearest_node.conf[1])+ abs(rand_node.conf[2] - nearest_node.conf[2])
    step = 0.05
    n = dist/step
    n = int(n)

    if n == 0:
        return False

    step0 = (rand_node.conf[0] - nearest_node.conf[0])/n
    step1 = (rand_node.conf[1] - nearest_node.conf[1])/n
    step2 = (rand_node.conf[2] - nearest_node.conf[2])/n

    curconf = nearest_node.conf

    for i in range(n):
        curconf = (curconf[0] + step0, curconf[1] + step1, curconf[2] + step2)

        if collision_fn(curconf):
            return False

    return True
    
def steer_to_until(rand_node, nearest_node):
    dist = abs(rand_node.conf[0] - nearest_node.conf[0]) + abs(rand_node.conf[1] - nearest_node.conf[1]) + abs(
        rand_node.conf[2] - nearest_node.conf[2])
    step = 0.05
    n = dist / step

    step0 = (rand_node.conf[0] - nearest_node.conf[0]) / n
    step1 = (rand_node.conf[1] - nearest_node.conf[1]) / n
    step2 = (rand_node.conf[2] - nearest_node.conf[2]) / n

    curconf = nearest_node.conf

    n = int(n)

    if n == 0:
        return nearest_node

    flag = False
    collfree = None

    for i in range(n):
        curconf = (curconf[0] + step0, curconf[1] + step1, curconf[2] + step2)


        if collision_fn(curconf):
            flag = True
            break
        else:
            collfree = curconf

    if collfree == None:
        return nearest_node

    if flag == True:
        before_col = RRT_Node(collfree)
        nearest_node.add_child(before_col)
        before_col.set_parent(nearest_node)

        return before_col
    else :
        nearest_node.add_child(rand_node)
        rand_node.set_parent(nearest_node)
        return rand_node

def RRT():
    spacing = 30
    conf_space = []

    step = math.pi * 2 / spacing

    for i in range(spacing*2-4):
        for j in range(spacing*2-4):
            for k in range(spacing-4):
                conf_space.append((-math.pi * 2 + i*step, -math.pi * 2 + j*step, -math.pi + k * step))


    q_start = RRT_Node(start_conf)
    T = [q_start]
    N = (spacing*spacing*spacing)**2

    q_last = RRT_Node(goal_conf)
    
    for i in range(N):
        curconfig, isgoal, idx = sample_conf(conf_space)
        conf_space.pop(idx)

        if collision_fn(curconfig) == True:
            continue

        q_rand = RRT_Node(curconfig)
        q_nearest = find_nearest(q_rand, T)

        flag = steer_to(q_rand, q_nearest)

        if flag:
            q_nearest.add_child(q_rand)
            q_rand.set_parent(q_nearest)
            T.append(q_rand)

            if isgoal == True:
                q_last = q_rand
                break

    path = [goal_conf]
    path.insert(0,q_last.conf)
    cur = q_last.parent

    while cur != None:
        path.insert(0, cur.conf)
        cur = cur.parent

    #add intermediate steering path in between the nodes
    full_path = []
    for i in range(len(path)-1):
        fromconf = path[i]
        toconf = path[i+1]
        dist = abs(fromconf[0] - toconf[0]) + abs(fromconf[1] - toconf[1]) + abs(
            fromconf[2] - toconf[2])
        step = 0.05
        n = dist / step
        n = int(n)

        if n == 0:
            continue

        step0 = (toconf[0] - fromconf[0]) / n
        step1 = (toconf[1] - fromconf[1]) / n
        step2 = (toconf[2] - fromconf[2]) / n

        curconf = fromconf
        full_path.append(fromconf)

        for i in range(n):
            curconf = (curconf[0] + step0, curconf[1] + step1, curconf[2] + step2)
            full_path.append(curconf)

    return path


def BiRRT():
    spacing = 30
    conf_space = []

    step = math.pi * 2 / spacing

    for i in range(spacing * 2 - 4):
        for j in range(spacing * 2 - 4):
            for k in range(spacing - 4):
                conf_space.append((-math.pi * 2 + i * step, -math.pi * 2 + j * step, -math.pi + k * step))

    q_start = RRT_Node(start_conf)
    q_goal = RRT_Node(goal_conf)
    T_a = [q_start]
    T_b = [q_goal]
    N = (spacing * spacing * spacing) ** 2

    a_last = None
    b_last = None

    for i in range(N):
        curconfig, isgoal, idx = sample_conf(conf_space)
        conf_space.pop(idx)  # remove from the conf_space as the sample collides

        if collision_fn(curconfig) == True:
            continue

        q_rand = RRT_Node(curconfig)
        a_nearest = find_nearest(q_rand, T_a)

        q_new = steer_to_until(q_rand, a_nearest)
        T_a.append(q_new)

        if q_new != None:
            #Connect
            b_nearest = find_nearest(q_new,T_b)
            connect = steer_to(q_new, b_nearest)

            if connect:
                b_last = b_nearest
                a_last = q_new
                break

        temp = T_a
        T_a = T_b
        T_b = temp

    #Find which tree has the start node to connect the two trees
    find_root = a_last

    while find_root.parent != None:
        find_root = find_root.parent

    if find_root != q_start:
        temp = a_last
        a_last = b_last
        b_last = temp

    path = [a_last.conf]
    cur = a_last.parent

    while cur != None:
        path.insert(0, cur.conf)
        cur = cur.parent

    cur = b_last

    while cur != None:
        path.append(cur.conf)
        cur = cur.parent

    # add intermediate path in between the nodes
    full_path = []
    for i in range(len(path) - 1):
        fromconf = path[i]
        toconf = path[i + 1]
        dist = abs(fromconf[0] - toconf[0]) + abs(fromconf[1] - toconf[1]) + abs(
            fromconf[2] - toconf[2])
        step = 0.05
        n = dist / step

        if n == 0:
            continue

        step0 = (toconf[0] - fromconf[0]) / n
        step1 = (toconf[1] - fromconf[1]) / n
        step2 = (toconf[2] - fromconf[2]) / n

        curconf = fromconf
        full_path.append(fromconf)

        n = int(n)
        if n == 0:
            n = 1

        for i in range(n):
            curconf = (curconf[0] + step0, curconf[1] + step1, curconf[2] + step2)
            full_path.append(curconf)

    return path

def BiRRT_smoothing():
    path = BiRRT()

    origin = len(path)
    print(len(path))
    for i in range(100):
        if len(path) <= 3:
            break

        idxs = np.random.choice(len(path), 2)
        idxa = idxs[0]
        idxb = idxs[1]

        if idxa > idxb:
            temp = idxa
            idxa = idxb
            idxb = temp

        na = RRT_Node(path[idxa])
        nb = RRT_Node(path[idxb])

        flag = steer_to(na, nb)

        if flag == True:
            temp_a = path[:idxa+1]
            temp_b = path[idxb:]

            path = temp_a + temp_b

    return path

###############################################################################
#your implementation ends here

if __name__ == "__main__":
    args = get_args()

    # set up simulator
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200, cameraTargetPosition=(0.0, 0.0, 0.0))

    # load objects
    plane = p.loadURDF("plane.urdf")
    ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)
    obstacle1 = p.loadURDF('assets/block.urdf',
                           basePosition=[1/4, 0, 1/2],
                           useFixedBase=True)
    obstacle2 = p.loadURDF('assets/block.urdf',
                           basePosition=[2/4, 0, 2/3],
                           useFixedBase=True)
    obstacles = [plane, obstacle1, obstacle2]

    # start and goal
    start_conf = (-0.813358794499552, -0.37120422397572495, -0.754454729356351)
    start_position = (0.3998897969722748, -0.3993956744670868, 0.6173484325408936)
    goal_conf = (0.7527214782907734, -0.6521867735052328, -0.4949270744967443)
    goal_position = (0.35317009687423706, 0.35294029116630554, 0.7246701717376709)
    goal_marker = draw_sphere_marker(position=goal_position, radius=0.02, color=[1, 0, 0, 1])
    set_joint_positions(ur5, UR5_JOINT_INDICES, start_conf)

    
		# place holder to save the solution path
    path_conf = None

    # get the collision checking function
    from collision_utils import get_collision_fn
    collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

    if args.birrt:
        if args.smoothing:
            # using birrt with smoothing
            path_conf = BiRRT_smoothing()
        else:
            # using birrt without smoothing
            path_conf = BiRRT()
    else:
        # using rrt
        path_conf = RRT()

    if path_conf is None:
        # pause here
        input("no collision-free path is found within the time budget, finish?")
    else:
        # execute the path
        while True:
            for q in path_conf:
                if q == goal_conf:
                    q = goal_conf
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                time.sleep(0.4)
            time.sleep(1)