import numpy as np
import open3d as o3d
import heapq
import pyvista as pv
# point cloud ply파일 불러오기 파일 불러오기
pcd = o3d.io.read_point_cloud("./downsampling_outliner_point_cloud_(100_100)_1.0.ply")

# point cloud 시각화
o3d.visualization.draw_geometries([pcd])

# point to voxel로 변환 무지개
# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.5)

voxel_size = 0.5  # 원하는 voxel 크기 지정 (단위: meter)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

# voxel 시각화
o3d.visualization.draw_geometries([voxel_grid])
# voxel_grid.origin: 시작점 [x0,y0,z0], voxel_grid.voxel_size: 크기






###############################
x_s = 60 # 시작 점
y_s = 10
z_s = 27.23

x_g = 85 # 목표 점
y_g = 10
z_g = 80 










voxels = voxel_grid.get_voxels()  # Open3D 0.15+
# 1-1) 맵 크기 계산 (셀 수)
#    각 voxel.grid_index 는 (i,j,k) 정수 인덱스
indices = np.array([v.grid_index for v in voxels])
min_idx = indices.min(axis=0)
max_idx = indices.max(axis=0) + 1
dims = (max_idx - min_idx).astype(int)  # (nx, ny, nz)

# 1-2) Occupancy 배열 (0=free, 1=occupied)
occ = np.zeros(dims, dtype=np.uint8)
for v in voxels:
    i, j, k = np.array(v.grid_index) - min_idx
    occ[i, j, k] = 1



origin = (0, 0, 0)
nx, ny, nz = dims
# 1) UniformGrid 세팅
# 1) ImageData (UniformGrid 대체) 생성
grid = pv.ImageData(
    dimensions=(nx+1, ny+1, nz+1),           # 셀 단위로 사용할 땐 +1
    spacing=(voxel_size,)*3,                 # 각 축 voxel 크기
    origin=tuple(origin)                     # 맵의 원점
)

# 2) 셀 데이터로 Occupancy 정보 붙이기
#    Cell data를 쓰려면 dimensions = 셀수+1
grid.cell_data["occ"] = occ.ravel(order="F")  # Fortran order flatten

# 3) threshold로 장애물(occ>0.5)만 추출
voxels = grid.threshold(0.5, scalars="occ")

# 4) Plotter로 인터랙티브 시각화
plotter = pv.Plotter()
plotter.add_mesh(
    voxels,
    color="tomato",
    opacity=0.8,
    show_edges=False
)
plotter.show_grid()
plotter.show(title="3D Occupancy Grid")



# 맵 정보를 전역 좌표계로 변환할 때는:
# world_coord = origin + (idx + 0.5) * voxel_size

# --- 2. 3D A* 경로 계획 예제 --------------------------

# 2-1) 이웃 후보 (6-연결)
neighbor_offsets = np.array([
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],
], dtype=int)

def heuristic(a, b):
    # Euclidean 거리
    return np.linalg.norm(np.array(a) - np.array(b))

def astar_3d(occ, start, goal):
    """
    occ    : numpy (nx,ny,nz) occupancy grid
    start  : (i,j,k) 시작 인덱스
    goal   : (i,j,k) 도착 인덱스
    return : 인덱스 리스트 경로 혹은 빈 리스트
    """
    nx, ny, nz = occ.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, None))
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        f, g, current, parent = heapq.heappop(open_set)
        if current in came_from:
            continue
        came_from[current] = parent
        
        if current == goal:
            # 경로 복원
            path = []
            node = current
            while node:
                path.append(node)
                node = came_from[node]
            return path[::-1]
        
        for d in neighbor_offsets:
            nb = tuple(np.array(current) + d)
            # 경계 검사
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny and 0 <= nb[2] < nz):
                continue
            # 충돌 검사
            if occ[nb]:
                continue
            tentative_g = g + 1  # 등간격 가정
            if tentative_g < g_score.get(nb, np.inf):
                g_score[nb] = tentative_g
                f_score = tentative_g + heuristic(nb, goal)
                heapq.heappush(open_set, (f_score, tentative_g, nb, current))
    return []  # 경로 없음

# 2-2) 예시: 시작/목표 월드 좌표 → 그리드 인덱스로 변환
start_world = np.array([x_s, y_s, z_s])
goal_world  = np.array([x_g, y_g, z_g])
start_idx = tuple(((start_world - voxel_grid.origin) / voxel_size - min_idx).astype(int))
goal_idx  = tuple(((goal_world  - voxel_grid.origin) / voxel_size - min_idx).astype(int))

# 2-3) A* 호출
path_idx = astar_3d(occ, start_idx, goal_idx)
if not path_idx:
    print("경로를 찾지 못했습니다.")
else:
    # 인덱스 경로 → 월드 좌표 경로
    path_world = [
        voxel_grid.origin + (np.array(idx) + min_idx + 0.5) * voxel_size
        for idx in path_idx
    ]
    print("월드 좌표 경로:", path_world)
