import matplotlib.pyplot as plt
import numpy as np
import csv

x = np.array((1,), dtype=np.float32)
y = np.array((1,), dtype=np.float32)
z = np.array((1,), dtype=np.float32)
obx = np.array((1,), dtype=np.float32)
oby = np.array((1,), dtype=np.float32)
obz = np.array((1,), dtype=np.float32)
counter = np.array((1,), dtype=np.float32)
reward = np.array((1,), dtype=np.float32)

fig = plt.subplots()
ax = plt.axes(projection="3d")

def max(a,b):
    if a > b:
        return a
    else:
        return b

def min(a,b):
    if a > b:
        return b
    else:
        return a

with open('./check.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            counter = np.append(counter, row[0])
            x = np.append(x, row[1])
            maxx = max(x[line_count], x[line_count-1])
            minx = min(x[line_count], x[line_count-1])
            y = np.append(y, row[2])
            maxy = max(x[line_count], x[line_count - 1])
            miny = min(x[line_count], x[line_count - 1])
            z = np.append(z, row[3])
            maxz = max(x[line_count], x[line_count - 1])
            minz = min(x[line_count], x[line_count - 1])
            obx = np.append(obx, row[4])
            maxobx = max(x[line_count], x[line_count - 1])
            minobx = min(x[line_count], x[line_count - 1])
            oby = np.append(oby, row[5])
            maxoby = max(x[line_count], x[line_count - 1])
            minoby = min(x[line_count], x[line_count - 1])
            obz = np.append(obz, row[6])
            maxobz = max(x[line_count], x[line_count - 1])
            minobz = min(x[line_count], x[line_count - 1])
            reward = np.append(reward, row[7])

            line_count += 1

    print(f'Processed {line_count} lines.')

x = np.concatenate([np.array(x, dtype=np.float32), np.array(obx, dtype=np.float32)])
y = np.concatenate([np.array(y, dtype=np.float32), np.array(oby, dtype=np.float32)])
z = np.concatenate([np.array(z, dtype=np.float32), np.array(obz, dtype=np.float32)])

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

ax.scatter3D(x, y, z, c='green')
plt.show()
