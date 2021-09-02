import sys, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from animate_3d import rotanimate

struct_json = sys.argv[1] # structure result filename

with open(struct_json) as f:
    data = json.load(f)

def get_line_endpoints(lines, juncs_on_line):
    endpoints = []
    for l in lines:
        m, v = np.array(lines[l][:3]), np.array(lines[l][3:])
        q = np.cross(v, m)
        lambdas = []
        for j in juncs_on_line[l]:
            if j in junctions:
                p = np.array(junctions[j])
                lambdas.append(np.dot(v, p - q))
        lambdas = sorted(lambdas)
        if len(lambdas) >= 2:
            lambda1, lambda2 = lambdas[0], lambdas[-1]
            endpoints.append([q + lambda1 * v, q + lambda2 * v])
    return endpoints

junctions = dict(data["junctions"])
lines = dict(data["lines"])
juncs_on_line = dict(data["junctions_on_line"])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for start, end in get_line_endpoints(lines, juncs_on_line):
    if np.linalg.norm(start - end) >= 0.0:
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])
    pass

all_points = np.array([junctions[i] for i in junctions])

ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0.0, 0.5)

ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2])
world_limits = ax.get_w_lims()

# ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
ax.set_box_aspect((1, 1, 0.5))

ax.figure.set_size_inches(25, 10)

# angles = np.linspace(0,360,21)[:-1] # Take 20 angles between 0 and 360
ax.view_init(elev=30.0, azim=45.0)

# create a movie with 10 frames per seconds and 'quality' 2000
# rotanimate(ax, angles,'../results/' + struct_json[:-5] + '.mp4',fps=10,bitrate=2000)

plt.show()
