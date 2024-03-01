import numpy as np

def boundcon(HazeImg, A, C0, C1, sz):
    if len(A) == 1:
        A = np.full((3, 1), A)
    if len(C0) == 1:
        C0 = np.full((3, 1), C0)
    if len(C1) == 1:
        C1 = np.full((3, 1), C1)
    
    HazeImg = np.double(HazeImg)

    t_r = np.maximum((A[0] - HazeImg[:, :, 0]) / (A[0] - C0[0]), (HazeImg[:, :, 0] - A[0]) / (C1[0] - A[0]))
    t_g = np.maximum((A[1] - HazeImg[:, :, 1]) / (A[1] - C0[1]), (HazeImg[:, :, 1] - A[1]) / (C1[1] - A[1]))
    t_b = np.maximum((A[2] - HazeImg[:, :, 2]) / (A[2] - C0[2]), (HazeImg[:, :, 2] - A[2]) / (C1[2] - A[2]))

    t_b = np.maximum.reduce([t_r, t_g, t_b], axis=0)
    t_b = np.minimum(t_b, 1)
    t_bdcon = t_b

    return t_bdcon
