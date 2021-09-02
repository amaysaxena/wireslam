import sys, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gt = sys.argv[1] # groundtruth filename

gt_data = pd.read_csv(gt)

gt_timestamps = np.array([x for x in gt_data['#timestamp']], dtype=np.int64)

gt_p_x = np.array([x for x in gt_data[' p_RS_R_x [m]']])
gt_p_y = np.array([x for x in gt_data[' p_RS_R_y [m]']])
gt_p_z = np.array([x for x in gt_data[' p_RS_R_z [m]']])

gt_q_w = np.array([x for x in gt_data[' q_RS_w []']])
gt_q_x = np.array([x for x in gt_data[' q_RS_x []']])
gt_q_y = np.array([x for x in gt_data[' q_RS_y []']])
gt_q_z = np.array([x for x in gt_data[' q_RS_z []']])

xs = []
ys = []
zs = []

qw = []
qx = []
qy = []
qz = []
for f in sys.argv[2:]:
    rs_data = pd.read_csv(f)

    rs_timestamps = np.array([x for x in rs_data['#timestamp']], dtype=np.int64)
    rs_p_x = np.array([x for x in rs_data[' p_RS_R_x [m]']])
    rs_p_y = np.array([x for x in rs_data[' p_RS_R_y [m]']])
    rs_p_z = np.array([x for x in rs_data[' p_RS_R_z [m]']])

    xs.append(rs_p_x)
    ys.append(rs_p_y)
    zs.append(rs_p_z)

    rs_q_w = np.array([x for x in rs_data[' q_RS_w []']])
    rs_q_x = np.array([x for x in rs_data[' q_RS_x []']])
    rs_q_y = np.array([x for x in rs_data[' q_RS_y []']])
    rs_q_z = np.array([x for x in rs_data[' q_RS_z []']])

    qw.append(rs_q_w)
    qx.append(rs_q_x)
    qy.append(rs_q_y)
    qz.append(rs_q_z)

    l = f.split('/')[-1][:-4]

fig, ax = plt.subplots()
for i, f in enumerate(sys.argv[2:]):
    x_err = xs[i] - gt_p_x
    y_err = ys[i] - gt_p_y
    z_err = zs[i] - gt_p_z
    T_err = np.sqrt(np.square(x_err) + np.square(y_err) + np.square(z_err))

    l = f.split('/')[-1][:-4]
    ax.plot(T_err, label=l)
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Position Error (m)")
fig.canvas.draw()
plt.show()

fig, ax = plt.subplots()
for i, f in enumerate(sys.argv[2:]):
    qw_err = qw[i] - gt_q_w
    qx_err = qx[i] - gt_q_x
    qy_err = qy[i] - gt_q_y
    qz_err = qz[i] - gt_q_z
    R_err = np.sqrt(np.square(qw_err) + np.square(qx_err) + np.square(qy_err) + np.square(qz_err))

    l = f.split('/')[-1][:-4]
    ax.plot(R_err, label=l)
ax.set_xlabel("Time")
ax.set_ylabel("Orientation Error (distance in quat space)")
ax.legend()
fig.canvas.draw()
plt.show()

