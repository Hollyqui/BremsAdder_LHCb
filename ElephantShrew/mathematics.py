import numpy as np
class Mathematics:
    def get_angle(v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        """ Returns the angle in radians between vectors 'v1' and 'v2'    """
        cosang = np.einsum('ij, ij->i', v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        return np.arctan2(sinang, cosang)

    '''finds distance between two points'''
    def get_distance(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
