import os
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from h5_utils import write_element_to_hptr_h5_file
from utils import classify_track, get_num_predict
from view import plot_scenario

# UniTraj to Waymo (hptr) map pl conversion
# UniTraj definition: https://github.com/vita-epfl/UniTraj/blob/735029a62889ebb6f1639fb7bf8f920d2498496c/unitraj/datasets/types.py#L40
# Waymo definition: https://github.com/zhejz/HPTR/blob/d2c1cb31ff5138ebf4b2490e2689c2f9da962120/src/pack_h5_womd.py#L17
one_hot_map = {
    1: 0,  # FREEWAY
    2: 1,  # SURFACE_STREET
    17: 2,  # STOP_SIGN
    3: 3,  # BIKE_LANE
    15: 4,  # TYPE_ROAD_EDGE_BOUNDARY
    16: 5,  # TYPE_ROAD_EDGE_MEDIAN
    6: 6,  # BROKEN
    9: 6,
    7: 7,  # SOLID_SINGLE
    11: 7,
    8: 8,  # DOUBLE
    10: 8,
    12: 8,
    13: 8,
    19: 9,  # SPEED_BUMP
    18: 10,  # CROSSWALK
}
# default dict if UniTraj type is 0 (=invalid) -> gets handled by using False valid values
one_hot_map = defaultdict(lambda: 0, one_hot_map)

# UniTraj to Waymo (hptr) agent conversion
one_hot_agent = {
    0: 0,  # VEHICLE
    1: 1,  # PEDESTRIAN
    2: 2,  # CYCLIST
}
# default dict if UniTraj type is 0 (=unset) or 4 (=other) -> gets handled by using False valid values

one_hot_agent = defaultdict(lambda: 0, one_hot_agent)
# "tl_lane/state", "tl_stop/state"
# LANE_STATE_UNKNOWN = 0;
# LANE_STATE_STOP = 1;
# LANE_STATE_CAUTION = 2;
# LANE_STATE_GO = 3;
# LANE_STATE_FLASHING = 4;


CURRENT_STEP = 10


def convert_unitraj_to_hptr_agent(data, hptr_data: dict, use_ped_cyc_keypoints= False):
    agent_pos = np.hstack(
        (data["obj_trajs"][..., 0:2], data["obj_trajs_future_state"][..., 0:2])
    )
    agent_vel = np.hstack(
        (data["obj_trajs"][..., 25:27], data["obj_trajs_future_state"][..., 2:4])
    )
    agent_valid = np.hstack((data["obj_trajs_mask"], data["obj_trajs_future_mask"]))

    if use_ped_cyc_keypoints:
        agent_kp = np.hstack(
            (data["obj_trajs"][..., 29:80], data["obj_trajs_future_state"][..., 4:56])
        )
        keypoint_valid = np.hstack((data["obj_kp_mask"], data["obj_future_kp_mask"]))

    n_agent = agent_pos.shape[0]
    assert n_agent == 64, "Number of agents must be 64"

    # agent/valid: shape (91, 64)
    hptr_data["agent/valid"] = agent_valid.transpose(1, 0).astype(bool)
    # agent/dest: shape (64)
    hptr_data["agent/dest"] = np.zeros(64, dtype=int)
    # agent/object_id: shape (64)
    hptr_data["agent/object_id"] = np.arange(64)
    # agent/pos: shape (91, 64, 2)
    hptr_data["agent/pos"] = agent_pos.transpose(1, 0, 2)
    # agent/size: shape (64, 3)
    hptr_data["agent/size"] = data["obj_trajs"][..., 0, 3:6]
    # agent/role: shape (64, 3)
    hptr_data["agent/role"] = np.zeros((64, 3), dtype=bool)
    # agent/kp: shape (91,64,51) agent/kp_valid: shape (91, 64, 17)
    if use_ped_cyc_keypoints:
        hptr_data["agent/kp"] = agent_kp.transpose(1, 0, 2)
        hptr_data["agent/kp_valid"] = keypoint_valid.transpose(1, 0, 2).astype(bool)
    # agent/type: shape (64, 3)
    # agent type unitraj (obj_trajs[..., 6:11]):
    # (veh, ped, cyc, predict, ego) -> no real one-hot encoding
    agent_type_one_hot = data["obj_trajs"][..., -1, 6:11]
    agent_type_int_waymo = []
    for i, type in enumerate(agent_type_one_hot):
        agent_type_int = np.argmax(type, axis=-1)
        agent_type_int_waymo.append(one_hot_agent[agent_type_int])

        if type[4] == 1: # Ego vehicle
            hptr_data["agent/role"][i][0] = True

        if get_num_predict(hptr_data["agent/role"]) < 8 and type[3] == 1: # Predict based on Unitraj
            hptr_data["agent/role"][i][2] = True

    assert get_num_predict(hptr_data["agent/role"]) <= 8, "Too many predict roles"

    hptr_data["agent/type"] = np.eye(3, dtype=bool)[agent_type_int_waymo]
    # agent/vel: shape (91, 64, 2)
    hptr_data["agent/vel"] = agent_vel.transpose(1, 0, 2)
    # agent/spd: shape (91, 64, 1)
    hptr_data["agent/spd"] = (
        np.linalg.norm(agent_vel, axis=2, keepdims=True)
        .transpose(1, 0, 2)
        .astype(np.float32)
    )
    # agent/acc: shape (91, 64, 1)
    # agent/yaw_bbox: shape (91, 64, 1)
    # agent/yaw_rate: shape (91, 64, 1)
    agent_acc = np.zeros((64, 91, 1))
    agent_yaw = np.zeros((64, 91, 1))
    agent_yaw_rate = np.zeros((64, 91, 1))
    for i in range(n_agent):
        if agent_valid[i].sum() > 1:
            valid_steps = np.where(agent_valid[i])[0]
            step_start = valid_steps[0]
            step_end = valid_steps[-1]
            speed = hptr_data["agent/spd"].transpose(1, 0, 2)[
                i, step_start : step_end + 1, :
            ]
            acc = np.diff(speed, axis=0) / 0.1
            agent_acc[i, step_start + 1 : step_end + 1, :] = acc

            vel_x = agent_vel[i, step_start : step_end + 1, 0]
            vel_y = agent_vel[i, step_start : step_end + 1, 1]
            yaw = np.arctan2(vel_y, vel_x).reshape(-1, 1)
            agent_yaw[i, step_start : step_end + 1, :] = yaw

            yaw_rate = np.diff(yaw, axis=0) / 0.1
            agent_yaw_rate[i, step_start + 1 : step_end + 1, :] = yaw_rate

    hptr_data["agent/acc"] = agent_acc.transpose(1, 0, 2).astype(np.float32)
    hptr_data["agent/yaw_bbox"] = agent_yaw.transpose(1, 0, 2).astype(np.float32)
    hptr_data["agent/yaw_rate"] = agent_yaw_rate.transpose(1, 0, 2).astype(np.float32)

    # agent/goal: shape (64, 4) TODO
    hptr_data["agent/goal"] = np.zeros((64, 4), dtype=np.float32)
    # agent/cmd: shape (64, 8)
    agent_cmd = np.zeros((64, 8), dtype=bool)
    for i in range(n_agent):
        track_type = classify_track(
            agent_valid[i],
            agent_pos[i, :],
            hptr_data["agent/yaw_bbox"].transpose(1, 0, 2)[i, :],
            hptr_data["agent/spd"].transpose(1, 0, 2)[i, :],
        )
        agent_cmd[i] = np.eye(8, dtype=bool)[track_type]
    hptr_data["agent/cmd"] = agent_cmd

    # check if all agents (with a role: ego, interaction, predict)
    # have at least one valid entry
    assert (
        hptr_data["agent/valid"]
        .transpose(1, 0)[hptr_data["agent/role"].any(-1)]
        .any(-1)
        .all()
    ), "All agents with a role must have at least one valid entry"

    """
    Note: agent_no_sim is used in validation dataset (and possibly test dataset).
    To get this work, the agent data is copied to these fields, since agent_no_sim is not available in UniTraj data.
    This is a workaround to get the HPTR model to work with the validation dataset.
    """
    # agent_no_sim/object_id: shape (256)
    hptr_data["agent_no_sim/object_id"] = np.arange(256) + 64
    # agent_no_sim/pos: shape (91, 256, 2)
    hptr_data["agent_no_sim/pos"] = np.zeros((91, 256, 2), dtype=np.float32)
    hptr_data["agent_no_sim/pos"][:, :64, :] = hptr_data["agent/pos"]
    # agent_no_sim/kp: shape (91, 256, 51) agent_no_sim/valid: shape (91, 256, 17)
    if use_ped_cyc_keypoints:
        hptr_data["agent_no_sim/kp"] = np.zeros((91, 256, 51), dtype=np.float32)
        hptr_data["agent_no_sim/kp"][:, :64, :] = hptr_data["agent/kp"]
        hptr_data["agent_no_sim/kp_valid"] = np.zeros((91, 256, 17), dtype=bool)
        hptr_data["agent_no_sim/kp_valid"][:, :64, :] = hptr_data["agent/kp_valid"]
    # agent_no_sim/size: shape (256, 3)
    hptr_data["agent_no_sim/size"] = np.zeros((256, 3), dtype=np.float32)
    hptr_data["agent_no_sim/size"][:64, :] = hptr_data["agent/size"]
    # agent_no_sim/spd: shape (91, 256, 1)
    hptr_data["agent_no_sim/spd"] = np.zeros((91, 256, 1), dtype=np.float32)
    hptr_data["agent_no_sim/spd"][:, :64, :] = hptr_data["agent/spd"]
    # agent_no_sim/type: shape (256, 3)
    hptr_data["agent_no_sim/type"] = np.zeros((256, 3), dtype=bool)
    hptr_data["agent_no_sim/type"][:64, :] = hptr_data["agent/type"]
    # agent_no_sim/valid: shape (91, 256)
    hptr_data["agent_no_sim/valid"] = np.zeros((91, 256), dtype=bool)
    hptr_data["agent_no_sim/valid"][:, :64] = hptr_data["agent/valid"]
    # agent_no_sim/vel: shape (91, 256, 2)
    hptr_data["agent_no_sim/vel"] = np.zeros((91, 256, 2), dtype=np.float32)
    hptr_data["agent_no_sim/vel"][:, :64, :] = hptr_data["agent/vel"]
    # agent_no_sim/yaw_bbox: shape (91, 256, 1)
    hptr_data["agent_no_sim/yaw_bbox"] = np.zeros((91, 256, 1), dtype=np.float32)
    hptr_data["agent_no_sim/yaw_bbox"][:, :64, :] = hptr_data["agent/yaw_bbox"]


def convert_unitraj_to_hptr_history_agent(data, hptr_data: dict, use_ped_cyc_keypoints= False):
    if not any(key.startswith("agent") for key in hptr_data.keys()):
        convert_unitraj_to_hptr_agent(data, hptr_data)

    hptr_data["history/agent/valid"] = hptr_data["agent/valid"][: CURRENT_STEP + 1]
    hptr_data["history/agent/object_id"] = hptr_data["agent/object_id"]
    hptr_data["history/agent/pos"] = hptr_data["agent/pos"][: CURRENT_STEP + 1]
    if use_ped_cyc_keypoints:
        hptr_data["history/agent/kp"] = hptr_data["agent/kp"][: CURRENT_STEP + 1]
        hptr_data["history/agent/kp_valid"] = hptr_data["agent/kp_valid"][: CURRENT_STEP + 1, :]
    hptr_data["history/agent/role"] = hptr_data["agent/role"]
    hptr_data["history/agent/size"] = hptr_data["agent/size"]
    hptr_data["history/agent/type"] = hptr_data["agent/type"]
    hptr_data["history/agent/vel"] = hptr_data["agent/vel"][: CURRENT_STEP + 1]
    hptr_data["history/agent/spd"] = hptr_data["agent/spd"][: CURRENT_STEP + 1]
    hptr_data["history/agent/acc"] = hptr_data["agent/acc"][: CURRENT_STEP + 1]
    hptr_data["history/agent/yaw_bbox"] = hptr_data["agent/yaw_bbox"][
        : CURRENT_STEP + 1
    ]
    hptr_data["history/agent/yaw_rate"] = hptr_data["agent/yaw_rate"][
        : CURRENT_STEP + 1
    ]
    hptr_data["history/agent_no_sim/object_id"] = hptr_data["agent_no_sim/object_id"]
    hptr_data["history/agent_no_sim/pos"] = hptr_data["agent_no_sim/pos"][
        : CURRENT_STEP + 1
    ]
    hptr_data["history/agent_no_sim/size"] = hptr_data["agent_no_sim/size"]
    hptr_data["history/agent_no_sim/spd"] = hptr_data["agent_no_sim/spd"][
        : CURRENT_STEP + 1
    ]
    hptr_data["history/agent_no_sim/type"] = hptr_data["agent_no_sim/type"]
    hptr_data["history/agent_no_sim/valid"] = hptr_data["agent_no_sim/valid"][
        : CURRENT_STEP + 1
    ]
    hptr_data["history/agent_no_sim/vel"] = hptr_data["agent_no_sim/vel"][
        : CURRENT_STEP + 1
    ]
    hptr_data["history/agent_no_sim/yaw_bbox"] = hptr_data["agent_no_sim/yaw_bbox"][
        : CURRENT_STEP + 1
    ]
    if use_ped_cyc_keypoints:
        hptr_data["history/agent_no_sim/kp"] = hptr_data["agent_no_sim/kp"][
        : CURRENT_STEP + 1
        ]
        hptr_data["history/agent_no_sim/kp_valid"] = hptr_data["agent_no_sim/kp_valid"][
        : CURRENT_STEP + 1
        ]



def convert_unitraj_to_hptr_map(data, hptr_data: dict):
    unitraj_map_pl = data["map_polylines"]
    # map/pos: shape (1024, 20, 2)
    hptr_data["map/pos"] = unitraj_map_pl[..., :2]
    # map/dir: shape (1024, 20, 2)
    hptr_data["map/dir"] = unitraj_map_pl[..., 3:5]
    # map/valid: shape (1024, 20)
    hptr_data["map/valid"] = data["map_polylines_mask"]
    # map/type: shape (1024, 11)
    map_type_int = np.argmax(unitraj_map_pl[..., 0, 9:29], axis=-1)
    map_type_int_waymo = [one_hot_map[int(t)] for t in map_type_int]
    hptr_data["map/type"] = np.eye(11, dtype=bool)[map_type_int_waymo]
    # map/boundary: shape (4)
    hptr_data["map/boundary"] = np.array(
        [
            hptr_data["map/pos"][..., 0].min(),  # min x
            hptr_data["map/pos"][..., 0].max(),  # max x
            hptr_data["map/pos"][..., 1].min(),  # min y
            hptr_data["map/pos"][..., 1].max(),  # max y
        ]
    )
    # map/id: shape (1024)
    # UniTraj has no map IDs
    hptr_data["map/id"] = np.arange(1024)


def convert_unitraj_to_hptr_tl(data, hptr_data: dict):
    # Currently, UniTraj data does not have traffic light information
    # tl_lane/idx: shape (91, 100)
    hptr_data["tl_lane/idx"] = np.zeros((91, 100), dtype=int)
    # tl_lane/state: shape (91, 100, 5)
    hptr_data["tl_lane/state"] = np.zeros((91, 100, 5), dtype=bool)
    # tl_lane/valid: shape (91, 100)
    hptr_data["tl_lane/valid"] = np.zeros((91, 100), dtype=bool)
    # tl_stop/dir: shape (91, 40, 2)
    hptr_data["tl_stop/dir"] = np.zeros((91, 40, 2), dtype=np.float32)
    # tl_stop/pos: shape (91, 40, 2)
    hptr_data["tl_stop/pos"] = np.zeros((91, 40, 2), dtype=np.float32)
    # tl_stop/state: shape (91, 40, 5)
    hptr_data["tl_stop/state"] = np.zeros((91, 40, 5), dtype=bool)
    # tl_stop/valid: shape (91, 40)
    hptr_data["tl_stop/valid"] = np.zeros((91, 40), dtype=bool)


def convert_unitraj_to_hptr_history_tl(data, hptr_data: dict):
    if not any(key.startswith("tl") for key in hptr_data.keys()):
        convert_unitraj_to_hptr_tl(data, hptr_data)

    hptr_data["history/tl_lane/idx"] = hptr_data["tl_lane/idx"][: CURRENT_STEP + 1]
    hptr_data["history/tl_lane/state"] = hptr_data["tl_lane/state"][: CURRENT_STEP + 1]
    hptr_data["history/tl_lane/valid"] = hptr_data["tl_lane/valid"][: CURRENT_STEP + 1]
    hptr_data["history/tl_stop/dir"] = hptr_data["tl_stop/dir"][: CURRENT_STEP + 1]
    hptr_data["history/tl_stop/pos"] = hptr_data["tl_stop/pos"][: CURRENT_STEP + 1]
    hptr_data["history/tl_stop/state"] = hptr_data["tl_stop/state"][: CURRENT_STEP + 1]
    hptr_data["history/tl_stop/valid"] = hptr_data["tl_stop/valid"][: CURRENT_STEP + 1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="train", choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--hptr_root_dir",
        type=str,
        default="/dir/to/nuscenes_hptr/",
        help="The root directory of the HPTR data. Within this directory the files training.h5, validation.h5, and testing.h5 are generated.",
    )
    parser.add_argument(
        "--unitraj_data_dir",
        type=str,
        default="/dir/to/nuscenes_unitraj/train/nuscenes_scenarionet",
        help="The directory of the UniTraj data to convert",
    )
    parser.add_argument(
        "--use_ped_cyc_keypoints",
        action="store_true",
        help="If --use_ped_cyc_keypoints is set, PEDESTRIAN and CYCLIST keypoints will be added to the HPTR format." \
        "VEHICLE keypoints will also be added, but they will appear padded at the center of the bbox",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="If save_plots is True, the plots are saved under --save_path",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="../res/plots/scenario",
        help="This path is used to store the scenario plots, the files are asumed to be named scenario_... .png",
    )
    args = parser.parse_args()
    dataset = args.dataset
    hptr_root_dir = args.hptr_root_dir
    unitraj_data_dir = args.unitraj_data_dir

    if dataset == "train":
        hptr_file = hptr_root_dir + "training.h5"
    elif dataset == "val":
        hptr_file = hptr_root_dir + "validation.h5"
    elif dataset == "test":
        hptr_file = hptr_root_dir + "testing.h5"

    data_files = os.listdir(unitraj_data_dir)
    data_files = [f for f in data_files if f.endswith(".h5")]

    num_samples = 0
    for filename in tqdm(data_files, desc="Converting data"):
        file_path = os.path.join(unitraj_data_dir, filename)
        with h5py.File(file_path, "r") as f:
            groups = list(f.keys())
            for i, group in enumerate(groups):
                data = {k: v[()] for k, v in f[group].items()}
                metadata = {}
                metadata["scenario_id"] = f[group]["scenario_id"]
                metadata["scenario_center"] = f[group]["map_center"][:2]
                metadata["scenario_yaw"] = 0
                metadata["with_map"] = True
                metadata["kalman_difficulty"] = f[group]["kalman_difficulty"]
                metadata["center_objects_world"] = f[group]["center_objects_world"]
                metadata["center_objects_id"] = f[group]["center_objects_id"]
                metadata["center_objects_type"] = f[group]["center_objects_type"]

                hptr_data = {}
                if dataset == "train":
                    convert_unitraj_to_hptr_agent(data, hptr_data, args.use_ped_cyc_keypoints)
                    convert_unitraj_to_hptr_map(data, hptr_data)
                    convert_unitraj_to_hptr_tl(data, hptr_data)
                    if args.save_plots:
                        plot_scenario(hptr_data, save_path= args.save_path + f"_g{i}_{filename}")
                elif dataset == "val":
                    convert_unitraj_to_hptr_agent(data, hptr_data, args.use_ped_cyc_keypoints)
                    convert_unitraj_to_hptr_history_agent(data, hptr_data, args.use_ped_cyc_keypoints)
                    convert_unitraj_to_hptr_map(data, hptr_data)
                    convert_unitraj_to_hptr_tl(data, hptr_data)
                    convert_unitraj_to_hptr_history_tl(data, hptr_data)
                    if args.save_plots:
                        plot_scenario(hptr_data,
                                      save_path= args.save_path + f"_g{i}_{filename}",
                                      use_ped_cyc_keypoints = args.use_ped_cyc_keypoints
                                      )
                elif dataset == "test":
                    convert_unitraj_to_hptr_history_agent(data, hptr_data, args.use_ped_cyc_keypoints)
                    convert_unitraj_to_hptr_map(data, hptr_data)
                    convert_unitraj_to_hptr_history_tl(data, hptr_data)

                write_element_to_hptr_h5_file(
                    hptr_file, str(num_samples), hptr_data, metadata
                )
                num_samples += 1

    with h5py.File(hptr_file, "a") as f:
        f.attrs["data_len"] = num_samples
