import numpy as np
import heapq
import pyvista as pv
from scipy.ndimage import binary_dilation

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def astar_2d(occ2d, start, goal):
    nx, ny = occ2d.shape
    neigh = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)]
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = [(h(start,goal), 0, start, None)]
    came, gscore = {}, {start:0}
    while open_set:
        f, g, cur, parent = heapq.heappop(open_set)
        if cur in came: continue
        came[cur] = parent
        if cur == goal:
            path = []; n = cur
            while n:
                path.append(n)
                n = came[n]
            return path[::-1]
        for dx, dy in neigh:
            nb = (cur[0]+dx, cur[1]+dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny): continue
            if occ2d[nb]: continue
            ng = g + 1
            if ng < gscore.get(nb, np.inf):
                gscore[nb] = ng
                heapq.heappush(open_set, (ng + h(nb,goal), ng, nb, cur))
    return []

# ------------------ 메인 코드 ------------------

# 1) 3D OGM 불러오기
occ = np.load("OGM_2.0x2.0_LH.npy")   # shape = (nx, ny, nz)
voxel_size = 2.0                      # 셀 크기

# 2) PyVista ImageData 생성
nx, ny, nz = occ.shape
grid = pv.ImageData(
    dimensions=(nx+1, ny+1, nz+1),
    spacing=(voxel_size, voxel_size, voxel_size),
    origin=(0.0, 0.0, 0.0)
)
grid.cell_data["occ"] = occ.ravel(order="F")
voxels = grid.threshold(0.5, scalars="occ")

# ─── 맵 불러오자마자 한 번 시각화 ─────────────────────────
map_plotter = pv.Plotter()
map_plotter.add_mesh(
    voxels,
    color="lightgray",
    opacity=0.6,
    show_edges=False
)
map_plotter.show_grid()
map_plotter.show(title="3D Occupancy Grid (Loaded)")
map_plotter.close()
# ─────────────────────────────────────────────────────────

# 3) 2D slice A* 연산
x_s, y_s, z_s = 60, 10, 27.23
x_g, y_g, z_g = 90, 60, 80
start_idx = tuple(((np.array([x_s, y_s, z_s]) - 0) / voxel_size).astype(int))
goal_idx  = tuple(((np.array([x_g, y_g, z_g]) - 0) / voxel_size).astype(int))

k0 = 1
occ2d = occ[:, k0, :]
start2d = (start_idx[0], start_idx[2])
goal2d  = (goal_idx[0],  goal_idx[2])

tank_radius = 2.0
n_margin = int(np.ceil(tank_radius / voxel_size))
struct2d = np.ones((2*n_margin+1, 2*n_margin+1), dtype=bool)
occ2d_inflated = binary_dilation(occ2d, structure=struct2d)

path2d = astar_2d(occ2d_inflated, start2d, goal2d)
if not path2d:
    raise RuntimeError("2D에서 경로를 찾지 못했습니다.")

# 4) 3D grid 위에 2D 경로 표시
path_mask = np.zeros_like(occ, dtype=np.uint8)
for (i, j) in path2d:
    path_mask[i, k0, j] = 1

grid.cell_data["path"] = path_mask.ravel(order="F")
path_voxels = grid.threshold(0.5, scalars="path")

# 5) 시각화
plotter = pv.Plotter()
plotter.add_mesh(voxels,      color="lightgray", opacity=0.6, show_edges=False)
plotter.add_mesh(path_voxels, color="yellow",     opacity=1.0, show_edges=False)
plotter.show_grid()
plotter.show(title="3D Occupancy + 2D Path")
plotter.close()
