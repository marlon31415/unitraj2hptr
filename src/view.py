import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict

# Visualization code adapted from https://github.com/KIT-MRT/future-motion

def plot_scenario(hptr_data: Dict,
                  save_path: str, 
                  num_joints: int = 17, 
                  ax_dist: int = 5):
    """
    Plot 3D trajectories and keypoints for multiple agents over time.

    Args:
        hptr_data: Dict, data in HPTR format
        save_path: str, location to save the plots
        num_joints: int, number of joints in the keypoints (default: 17)
        ax_dist: int
    """
    in_edge = [(4, 2),(3,1),(2,0),(1,0),(0,5),(0,6),(6,8),(5,7),(8,10),(7,9),
               (5,11),(6,12),(12,14),(11,13),(14,16),(13,15)] # kp graph

    pos = hptr_data["agent/pos"]
    kp = hptr_data["agent/kp"]
    type = hptr_data["agent/type"]
    yaw = hptr_data["agent/yaw_bbox"]

    timesteps, num_agents, _ = pos.shape

    fig = plt.figure(figsize=(15, 15), dpi=80)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.view_init(elev=50.0, azim=-75)

    for map_polyline, map_valid, map_type in zip(
        hptr_data["map/pos"],
        hptr_data["map/valid"],
        hptr_data["map/type"],
    ):
        map_polyline = map_polyline[map_valid]

        # lanes black, else white
        if (
            map_type[4]
            or map_type[5]
            or map_type[6]
            or map_type[7]
            or map_type[8]
            or map_type[9]
            or map_type[10]
        ):
            plt.plot(map_polyline[:, 0], map_polyline[:, 1], "-", c="white", zorder=-10)
            #pass
        else:
            #pass
            plt.plot(map_polyline[:, 0], map_polyline[:, 1], "-", c="black", zorder=-10)

    
    all_x, all_y, all_z = [], [], []

    colors = ["blue", "orange", "green"]

    # Plot trajectories (z = 0)
    for agent in range(num_agents):
            traj = pos[:, agent, :]
            mask = ~((traj[:, 0] == 0) & (traj[:, 1] == 0))
            filtered_traj = traj[mask]
            agent_yaw = yaw[:, agent, :]
            if len(filtered_traj) > 0:
                if traj[10, 0] != 0 and traj[10, 1] != 0:
                    color = colors[np.argmax(type[agent])]
                    ax.plot(filtered_traj[:, 0], filtered_traj[:, 1], zs=0, zdir='z', alpha=0.99, color = color)
                    all_x.extend([traj[10, 0]])
                    all_y.extend([traj[10, 1]])
                    all_z.extend([0])

                    if type[agent, 0]:
                        bbox = rotate_bbox_zaxis(car, float(agent_yaw[10,0]))
                    elif type[agent, 1]:
                        bbox = rotate_bbox_zaxis(pedestrian, float(agent_yaw[10,0]))
                    elif type[agent, 2]:
                        bbox = rotate_bbox_zaxis(cyclist, float(agent_yaw[10,0]))
                    
                    bbox = shift_cuboid(float(traj[10, 0]), float(traj[10, 1]), bbox)
                    add_cube(bbox, ax, color=color, alpha=0.1)

    # Plot keypoints
    for agent in range(num_agents):
        if type[agent, 1] == 1 or type[agent, 2] == 1:
            for t in range(timesteps):
                if t == 10:
                    keypoints_flat = kp[t, agent]
                    keypoints = keypoints_flat.reshape(num_joints, 3)
                    for edge in in_edge:
                        i, j = edge
                        p1, p2 = keypoints[i], keypoints[j]
                        if (p1 != p2).all():
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='yellow')
                            all_x.extend([p1[0], p2[0]])
                            all_y.extend([p1[1], p2[1]])
                            all_z.extend([p1[2], p2[2]])
    # Set axis limits
    margin = -0.35  # 10% margin
    if all_x and all_y and all_z:
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        
        dx = (x_max - x_min)
        dy = (y_max - y_min)
        
        max_range = max([dx,dy]) * margin

        ax.set_xlim(x_min - max_range, x_max + max_range)
        ax.set_ylim(y_min - max_range, y_max + max_range)

    ax.set_zlim(bottom=0, top=5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("tab:grey")
    
    plt.savefig(f"{save_path}.png")

def shift_cuboid(x_shift, y_shift, cuboid):
    cuboid = np.copy(cuboid)
    cuboid[:, 0] += x_shift
    cuboid[:, 1] += y_shift

    return cuboid


def rotate_point_zaxis(p, angle):
    rot_matrix = np.array(
        [
            [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
            [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0],
            [0, 0, 1],
        ]
    )
    return np.matmul(p, rot_matrix)


def rotate_bbox_zaxis(bbox, angle):
    bbox = np.copy(bbox)
    _bbox = []
    angle = np.rad2deg(-angle)
    for point in bbox:
        _bbox.append(rotate_point_zaxis(point, angle))

    return np.array(_bbox)


def add_cube(cube_definition, ax, color="b", edgecolor="k", alpha=0.2):
    cube_definition_array = [np.array(list(item)) for item in cube_definition]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0],
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]],
    ]

    faces = Poly3DCollection(
        edges, linewidths=1, edgecolors=edgecolor, facecolors=color, alpha=alpha
    )

    ax.add_collection3d(faces)
    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0)


car = np.array(
    [
        (-2.25, -1, 0),  # left bottom front
        (-2.25, 1, 0),  # left bottom back
        (2.25, -1, 0),  # right bottom front
        (-2.25, -1, 1.5),  # left top front -> height
    ]
)

pedestrian = np.array(
    [
        (-0.3, -0.3, 0),  # left bottom front
        (-0.3, 0.3, 0),  # left bottom back
        (0.3, -0.3, 0),  # right bottom front
        (-0.3, -0.3, 2),  # left top front -> height
    ]
)

cyclist = np.array(
    [
        (-1, -0.3, 0),  # left bottom front
        (-1, 0.3, 0),  # left bottom back
        (1, -0.3, 0),  # right bottom front
        (-1, -0.3, 2),  # left top front -> height
    ]
)
