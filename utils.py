import math
import numpy as np

first_bin = 0
last_bin = 15
pavlov_to_go = 1.5
# from niko shared in slacks and servo.py . I just added "ang"
def parse_data(data):
    ang = (data[1] * 256 + data[0] - 0x200) * math.radians(300.0) / 1024.0
    position = (data[0] + data[1] * 256 - 512) * 5 * math.pi / 3069
    speed = (data[2] + data[3] * 256) * 5 * math.pi / 3069
    load = (data[5] & 3) * 256 + data[4] * (1 - 2 * bool(data[5] & 4))
    voltage = data[6] / 10
    temperature = data[7]

    return [ang, position, speed, load, voltage, temperature]

def is_approx_equal(a,b,degree = 1e-2):
    is_app_eq = (abs(a - b) <= max(1e-4 * max(abs(a), abs(b)), degree))
    return is_app_eq

def cummlant_positive(state):
    if state == last_bin:
        return 0.0
    else:
        return 1.0

def cummlant_negative(state):
    if state == first_bin:
        return 0.0
    else:
        return 1.0


def back_forth_policy(state, action):
    return 1.0

def go_to_neg_policy(state, action):
    if action == -1:
        return 1.0
    else:
        return 0.0

def go_to_pos_policy(state, action):
    if action == 1:
        return 1.0
    else:
        return 0.0

def gamma_to_neg(state):
    if state == first_bin:
        return 0.0
    else:
        return 1.0

def gamma_to_pos(state):
    if state == last_bin:
        return 0.0
    else:
        return 1.0

def feature_vector_on(state):
    fvector = np.zeros(last_bin+1)
    fvector[state] = 1.0
    return fvector

def feature_vector_off(state):
    fvector = np.zeros(last_bin+1)
    fvector[state] = 1.0
    return fvector

def read_data(servo):
    read_all = [0x02, 0x24, 0x08]
    data = servo.send_instruction(read_all, servo.servo_id)
    return parse_data(data)

def policy_robot(servo,ang,dir):
    if is_approx_equal(ang,1.5):
        servo.move_angle(-1.5, blocking=False)
        dir = -1
    elif is_approx_equal(ang,-1.5):
        servo.move_angle(1.5, blocking=False)
        dir = 1
    return dir

def pavlov(servo,ang):
    global pavlov_to_go
    servo.move_angle(pavlov_to_go, blocking=False)
    pavlov_to_go = -1* pavlov_to_go

def get_angle_bin_on(ang,dir,bins):
    if dir == 1:
        ang_f_bin = ang + 1.5
    elif dir == -1:
        ang_f_bin = 4.5 - ang

    return np.digitize(ang_f_bin, bins)

def get_angle_bin_off(ang,dir,bins):

    ang_f_bin = ang + 1.5
    return np.digitize(ang_f_bin, bins)
