import numpy as np
from shapely.geometry import Polygon

def get_corners(pos, size, yaw):
        
        dx =  size[0]/2 * np.cos(yaw)
        dy =  size[0]/2 * np.sin(yaw)
        wx =  size[1]/2 * -np.sin(yaw)
        wy =  size[1]/2 *  np.cos(yaw)

        return np.array([
            [pos[:,0:1] + dx + wx, pos[:,1:2] + dy + wy],
            [pos[:,0:1] + dx - wx, pos[:,1:2] + dy - wy],
            [pos[:,0:1] - dx - wx, pos[:,1:2] - dy - wy],
            [pos[:,0:1] - dx + wx, pos[:,1:2] - dy + wy],
        ]).transpose(2,0,1,3).squeeze(-1)

def overlaps(
        target: int,
        hptr_data: dict,
        iou_thresh: float= 0.03,
        current_step: int=10
    ):

    target_pos = hptr_data["agent/pos"][:, target] # (91, 2) x,y
    target_yaw = hptr_data["agent/yaw_bbox"][:, target] # (91, 1) yaw
    target_size = hptr_data["agent/size"][target] # (3) h,w,d

    for i in range(hptr_data["agent/pos"].shape[1]):

        if i == target:
            continue

        other_pos = hptr_data["agent/pos"][:, i] # (91, 2) x,y
        other_yaw = hptr_data["agent/yaw_bbox"][:, i] # (91, 1) yaw
        other_size = hptr_data["agent/size"][i] # (91, 3) h,w,d

        target_box = get_corners(target_pos, target_size, target_yaw) # (91, 4, 2)
        other_box = get_corners(other_pos, other_size, other_yaw) # (91, 4, 2)

        
        poly1 = Polygon(target_box[current_step,...])
        poly2 = Polygon(other_box[current_step,...])

        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        if union != 0:
            iou = inter/union
        else:
            iou = 0

        if iou >= iou_thresh:
            return True
    return False


def classify_track(
    valid: np.ndarray,
    pos: np.ndarray,
    yaw: np.ndarray,
    spd: np.ndarray,
    kMaxSpeedForStationary: float = 2.0,  # (m/s)
    kMaxDisplacementForStationary: float = 5.0,  # (m)
    kMaxLateralDisplacementForStraight: float = 5.0,  # (m)
    kMinLongitudinalDisplacementForUTurn: float = -5.0,  # (m)
    kMaxAbsHeadingDiffForStraight: float = 0.5236,  # M_PI / 6.0
) -> int:
    """
    https://github.com/zhejz/HPTR/blob/d2c1cb31ff5138ebf4b2490e2689c2f9da962120/src/utils/pack_h5.py#L65
    github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/metrics/motion_metrics_utils.cc
    Args:
        valid: [n_step], bool
        pos: [n_step, 2], x,y
        yaw: [n_step], float32
        spd: [n_step], float32
    Returns:
        traj_type: int in range(N_AGENT_CMD)
            # STATIONARY = 0;
            # STRAIGHT = 1;
            # STRAIGHT_LEFT = 2;
            # STRAIGHT_RIGHT = 3;
            # LEFT_U_TURN = 4;
            # LEFT_TURN = 5;
            # RIGHT_U_TURN = 6;
            # RIGHT_TURN = 7;
    """
    i0 = valid.argmax()
    i1 = len(valid) - 1 - np.flip(valid).argmax()

    x, y = pos[i1] - pos[i0]
    final_displacement = np.sqrt(x**2 + y**2)

    _c = np.cos(-yaw[i0])
    _s = np.sin(-yaw[i0])
    dx = x * _c - y * _s
    dy = x * _s + y * _c

    heading_diff = yaw[i1] - yaw[i0]
    max_speed = max(spd[i0], spd[i1])

    if (
        max_speed < kMaxSpeedForStationary
        and final_displacement < kMaxDisplacementForStationary
    ):
        return 0  # TrajectoryType::STATIONARY;

    if np.abs(heading_diff) < kMaxAbsHeadingDiffForStraight:
        if np.abs(dy) < kMaxLateralDisplacementForStraight:
            return 1  # TrajectoryType::STRAIGHT;
        if dy > 0:
            return 2  # TrajectoryType::STRAIGHT_LEFT
        else:
            return 3  # TrajectoryType::STRAIGHT_RIGHT

    if heading_diff < -kMaxAbsHeadingDiffForStraight and dy < 0:
        if dx < kMinLongitudinalDisplacementForUTurn:
            return 6  # TrajectoryType::RIGHT_U_TURN
        else:
            return 7  # TrajectoryType::RIGHT_TURN

    if dx < kMinLongitudinalDisplacementForUTurn:
        return 4  # TrajectoryType::LEFT_U_TURN;

    return 5  # TrajectoryType::LEFT_TURN;

def get_num_predict(
    role: np.ndarray
    ) -> int:
    """
    Args:
        role: shape (64, 3), bool
    Returns:
        NUM_PREDICT_ROLES: int
    """

    return int(role[:, 2].sum())