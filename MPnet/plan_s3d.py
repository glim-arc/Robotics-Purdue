import numpy as np

def IsInCollision(x,obc):
    cubesize = [[5, 5, 10], [5, 10, 5], [5, 10, 10], [10, 5, 5], [10, 5, 10], [10, 10, 5], [10, 10, 10], [5, 5, 5],
                [10, 10, 10], [5, 5, 5]]

    if (x[0] > 20 or x[0] < -20) or \
            (x[1] > 20 or x[1] < -20) or \
            (x[2] > 20 or x[2] < -20):
        return True

    s=np.zeros(3,dtype=np.float32)
    s[0]=x[0]
    s[1]=x[1]
    s[2]=x[2]

    for i, size in enumerate(cubesize):
        collision = True
        for j in range(0,3):
            if abs(obc[i][j] - s[j]) > size[j]/2.0:
                collision = False
                break
        if collision == True:
            return True
    return False