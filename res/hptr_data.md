# HPTR Data

## Training
agent/acc: shape (91, 64, 1)\
agent/cmd: shape (64, 8)\
agent/dest: shape (64)\
agent/goal: shape (64, 4)\
agent/object_id: shape (64)\
agent/pos: shape (91, 64, 2)\
agent/role: shape (64, 3)\
agent/size: shape (64, 3)\
agent/spd: shape (91, 64, 1)\
agent/type: shape (64, 3)\
agent/valid: shape (91, 64)\
agent/vel: shape (91, 64, 2)\
agent/yaw_bbox: shape (91, 64, 1)\
agent/yaw_rate: shape (91, 64, 1)\
map/boundary: shape (4)\
map/dir: shape (1024, 20, 2)\
map/id: shape (1024)\
map/pos: shape (1024, 20, 2)\
map/type: shape (1024, 11)\
map/valid: shape (1024, 20)\
tl_lane/idx: shape (91, 100)\
tl_lane/state: shape (91, 100, 5)\
tl_lane/valid: shape (91, 100)\
tl_stop/dir: shape (91, 40, 2)\
tl_stop/pos: shape (91, 40, 2)\
tl_stop/state: shape (91, 40, 5)\
tl_stop/valid: shape (91, 40)

## Validation
agent/acc: shape (91, 64, 1)\
agent/cmd: shape (64, 8)\
agent/dest: shape (64)\
agent/goal: shape (64, 4)\
agent/object_id: shape (64)\
agent/pos: shape (91, 64, 2)\
agent/role: shape (64, 3)\
agent/size: shape (64, 3)\
agent/spd: shape (91, 64, 1)\
agent/type: shape (64, 3)\
agent/valid: shape (91, 64)\
agent/vel: shape (91, 64, 2)\
agent/yaw_bbox: shape (91, 64, 1)\
agent/yaw_rate: shape (91, 64, 1)\
agent_no_sim/object_id: shape (256)\
agent_no_sim/pos: shape (91, 256, 2)\
agent_no_sim/size: shape (256, 3)\
agent_no_sim/spd: shape (91, 256, 1)\
agent_no_sim/type: shape (256, 3)\
agent_no_sim/valid: shape (91, 256)\
agent_no_sim/vel: shape (91, 256, 2)\
agent_no_sim/yaw_bbox: shape (91, 256, 1)\
history/agent/acc: shape (11, 64, 1)\
history/agent/object_id: shape (64)\
history/agent/pos: shape (11, 64, 2)\
history/agent/role: shape (64, 3)\
history/agent/size: shape (64, 3)\
history/agent/spd: shape (11, 64, 1)\
history/agent/type: shape (64, 3)\
history/agent/valid: shape (11, 64)\
history/agent/vel: shape (11, 64, 2)\
history/agent/yaw_bbox: shape (11, 64, 1)\
history/agent/yaw_rate: shape (11, 64, 1)\
history/agent_no_sim/object_id: shape (256)\
history/agent_no_sim/pos: shape (11, 256, 2)\
history/agent_no_sim/size: shape (256, 3)\
history/agent_no_sim/spd: shape (11, 256, 1)\
history/agent_no_sim/type: shape (256, 3)\
history/agent_no_sim/valid: shape (11, 256)\
history/agent_no_sim/vel: shape (11, 256, 2)\
history/agent_no_sim/yaw_bbox: shape (11, 256, 1)\
history/tl_lane/idx: shape (11, 100)\
history/tl_lane/state: shape (11, 100, 5)\
history/tl_lane/valid: shape (11, 100)\
history/tl_stop/dir: shape (11, 40, 2)\
history/tl_stop/pos: shape (11, 40, 2)\
history/tl_stop/state: shape (11, 40, 5)\
history/tl_stop/valid: shape (11, 40)\
map/boundary: shape (4)\
map/dir: shape (1024, 20, 2)\
map/id: shape (1024)\
map/pos: shape (1024, 20, 2)\
map/type: shape (1024, 11)\
map/valid: shape (1024, 20)\
tl_lane/idx: shape (91, 100)\
tl_lane/state: shape (91, 100, 5)\
tl_lane/valid: shape (91, 100)\
tl_stop/dir: shape (91, 40, 2)\
tl_stop/pos: shape (91, 40, 2)\
tl_stop/state: shape (91, 40, 5)\
tl_stop/valid: shape (91, 40)

## Testing
history/agent/acc: shape (11, 64, 1)\
history/agent/object_id: shape (64)\
history/agent/pos: shape (11, 64, 2)\
history/agent/role: shape (64, 3)\
history/agent/size: shape (64, 3)\
history/agent/spd: shape (11, 64, 1)\
history/agent/type: shape (64, 3)\
history/agent/valid: shape (11, 64)\
history/agent/vel: shape (11, 64, 2)\
history/agent/yaw_bbox: shape (11, 64, 1)\
history/agent/yaw_rate: shape (11, 64, 1)\
history/agent_no_sim/object_id: shape (256)\
history/agent_no_sim/pos: shape (11, 256, 2)\
history/agent_no_sim/size: shape (256, 3)\
history/agent_no_sim/spd: shape (11, 256, 1)\
history/agent_no_sim/type: shape (256, 3)\
history/agent_no_sim/valid: shape (11, 256)\
history/agent_no_sim/vel: shape (11, 256, 2)\
history/agent_no_sim/yaw_bbox: shape (11, 256, 1)\
history/tl_lane/idx: shape (11, 100)\
history/tl_lane/state: shape (11, 100, 5)\
history/tl_lane/valid: shape (11, 100)\
history/tl_stop/dir: shape (11, 40, 2)\
history/tl_stop/pos: shape (11, 40, 2)\
history/tl_stop/state: shape (11, 40, 5)\
history/tl_stop/valid: shape (11, 40)\
map/boundary: shape (4)\
map/dir: shape (1024, 20, 2)\
map/id: shape (1024)\
map/pos: shape (1024, 20, 2)\
map/type: shape (1024, 11)\
map/valid: shape (1024, 20)

# UniTraj Data
center_gt_final_valid_idx: 79.0\
center_gt_trajs: (80, 4)\
center_gt_trajs_mask: (80,)\
center_gt_trajs_src: (91, 10)\
center_objects_id: b'dc762bf1bc694d3e8141bf592f9b1456'\
center_objects_type: 1\
center_objects_world: (10,) (x,y,z,l,w,h,heading,vx,vy,valid)\
dataset_name_ b'nuscenes'\
kalman_difficulty: (3,)\
map_center: (3,)\
map_polylines: (1024, 20, 29)\
map_polylines_center: (1024, 3)\
map_polylines_mask: (1024, 20)\
obj_trajs: (64, 11, 29)\
obj_trajs_future_mask: (64, 80)\
obj_trajs_future_state: (64, 80, 4)\
obj_trajs_last_pos: (64, 3)\
obj_trajs_mask: (64, 11)\
obj_trajs_pos: (64, 11, 3)\
scenario_id: b'scene-0103_dc762bf1bc694d3e8141bf592f9b1456_c5f58c19249d4137ae063b0e9ecd8b8e'\
track_index_to_predict: 0\
trajectory_type: 7