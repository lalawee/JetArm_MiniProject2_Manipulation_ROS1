import numpy as np

def JetArm_Jacobian(q1, q2, q3, q4):
    """
    JetArm Jacobian
    Outputs:
    J_v = linear velocity Jacobian
    J_w = angular velocity Jacobian
    Inputs:
    q1, q2, q3, q4 - joint angles in radians
    """
    
    # Linear Velocity Jacobian (J_v)
    J_v = np.array([
        [-(3 * np.sin(q1) * (1415 * np.sin(q2 + q3 + q4) + 2157 * np.sin(q2 + q3) + 2157 * np.sin(q2))) / 50000,
         (3 * np.cos(q1) * (1415 * np.cos(q2 + q3 + q4) + 2157 * np.cos(q2 + q3) + 2157 * np.cos(q2))) / 50000,
         (6471 * np.cos(q1 + q2 + q3)) / 100000 + (6471 * np.cos(q2 - q1 + q3)) / 100000 + (849 * np.cos(q1 + q2 + q3 + q4)) / 20000 + (849 * np.cos(q2 - q1 + q3 + q4)) / 20000,
         (849 * np.cos(q1 + q2 + q3 + q4)) / 20000 + (849 * np.cos(q2 - q1 + q3 + q4)) / 20000,
         0,
         0],
        
        [(3 * np.cos(q1) * (1415 * np.sin(q2 + q3 + q4) + 2157 * np.sin(q2 + q3) + 2157 * np.sin(q2))) / 50000,
         (3 * np.sin(q1) * (1415 * np.cos(q2 + q3 + q4) + 2157 * np.cos(q2 + q3) + 2157 * np.cos(q2))) / 50000,
         (6471 * np.sin(q1 + q2 + q3)) / 100000 - (6471 * np.sin(q2 - q1 + q3)) / 100000 + (849 * np.sin(q1 + q2 + q3 + q4)) / 20000 - (849 * np.sin(q2 - q1 + q3 + q4)) / 20000,
         (849 * np.sin(q1 + q2 + q3 + q4)) / 20000 - (849 * np.sin(q2 - q1 + q3 + q4)) / 20000,
         0,
         0],
        
        [0,
         - (849 * np.sin(q2 + q3 + q4)) / 10000 - (6471 * np.sin(q2 + q3)) / 50000 - (6471 * np.sin(q2)) / 50000,
         - (849 * np.sin(q2 + q3 + q4)) / 10000 - (6471 * np.sin(q2 + q3)) / 50000,
         - (849 * np.sin(q2 + q3 + q4)) / 10000,
         0,
         0]
    ])
    
    # Angular Velocity Jacobian (J_w)
    J_w = np.array([
        [0, -np.sin(q1), -np.sin(q1), -np.sin(q1), np.sin(q1 + q2 + q3 + q4) / 2 + np.sin(q2 - q1 + q3 + q4) / 2, np.sin(q1 + q2 + q3 + q4) / 2 + np.sin(q2 - q1 + q3 + q4) / 2],
        [0,  np.cos(q1),  np.cos(q1),  np.cos(q1), np.cos(q2 - q1 + q3 + q4) / 2 - np.cos(q1 + q2 + q3 + q4) / 2, np.cos(q2 - q1 + q3 + q4) / 2 - np.cos(q1 + q2 + q3 + q4) / 2],
        [1,         0,         0,         0,                                   np.cos(q2 + q3 + q4),                                   np.cos(q2 + q3 + q4)]
    ])
    
    return J_v, J_w