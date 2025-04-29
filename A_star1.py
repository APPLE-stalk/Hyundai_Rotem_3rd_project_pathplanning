import numpy as np
import open3d as o3d
import heapq
import pyvista as pv
from scipy.ndimage import binary_dilation # 장애물 마진 용

def heuristic(a, b):
    # Euclidean 거리
    return np.linalg.norm(np.array(a) - np.array(b))

def astar_2d(occ2d, start, goal):
    nx, ny = occ2d.shape
    neigh = [(1,0),(-1,0),(0,1),(0,-1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = [(h(start,goal), 0, start, None)]
    came, gscore = {}, {start:0}
    while open_set:
        f, g, cur, parent = heapq.heappop(open_set)
        if cur in came: continue
        came[cur] = parent
        if cur == goal:
            path=[]; n=cur
            while n: path.append(n); n=came[n]
            return path[::-1]
        for dx,dy in neigh:
            nb = (cur[0]+dx, cur[1]+dy)
            if not (0<=nb[0]<nx and 0<=nb[1]<ny): continue
            if occ2d[nb]: continue
            ng = g+1
            if ng < gscore.get(nb, float('inf')):
                gscore[nb] = ng
                heapq.heappush(open_set, (ng+h(nb,goal), ng, nb, cur))
    return []



# point cloud 파일(.ply) 불러오기 파일 불러오기
# 100m*100m 사이즈, 1.0m*1.0m단위로 down sampling 한 맵으로 성공하였음 "./downsampling_outliner_point_cloud_(100_100)_1.0.ply"
pcd = o3d.io.read_point_cloud("./downsampling_outliner_point_cloud_(100_100)_1.0.ply")

# point cloud 시각화
# o3d.visualization.draw_geometries([pcd])

# point -> voxel로 변환
voxel_size = 2.0  # 원하는 voxel 크기 지정 (단위: meter)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

# voxel 시각화
# 검은색으로 나옴. open3d 특성상 어쩔 수 없는 듯
# o3d.visualization.draw_geometries([voxel_grid])




x_s = 60 # 시작 점
y_s = 10
z_s = 27.23

x_g = 90 # 목표 점
y_g = 60
z_g = 80


voxels = voxel_grid.get_voxels()
print('voxel 갯수: ', len(voxels)) # 2944개
print('voxel 1개 예시: ', voxels[0])
print('voxel 좌표', voxels[0].grid_index, 'voxel 색상', voxels[0].color) # voxel 데이터 접근 방법: 


# 맵 크기 계산 (셀 수)
# 각 voxel.grid_index 는 (i,j,k) 정수 인덱스
indices = np.array([v.grid_index for v in voxels])
min_idx = indices.min(axis=0) # axis=0의미: 같은 열끼리 비교하여 min값을 구하시오
max_idx = indices.max(axis=0) + 1 # 슬라이싱 할 때 마지막 인덱시 미포함 고려하여 +1씩 해주기
dims = (max_idx - min_idx).astype(int)  # (nx, ny, nz) 각 축별 길이값 구하기
nx, ny, nz = dims


# Occupancy 배열 (0=free, 1=occupied)
# 0==free => 갈 수 있음 // 1==occupied => 갈 수 없음
occ = np.zeros(dims, dtype=np.uint8)
for v in voxels:
    i, j, k = np.array(v.grid_index) - min_idx
    occ[i, j, k] = 1



# Occupied Grid Map 시각화
# ImageData 생성
origin = min_idx
grid = pv.ImageData(
    dimensions=(nx+1, ny+1, nz+1), # 셀 단위로 사용할 땐 +1
    spacing=(voxel_size,)*3,       # 각 축 voxel 크기
    origin=tuple(origin)           # 맵의 원점
)

# 1D로 펼쳐서 grid 객체에 추가하기 
grid.cell_data["occ"] = occ.ravel(order="F") # order="F": 평탄화

# grid객체 중 장애물(occ>0.5)
voxels = grid.threshold(0.5, scalars="occ") # occ값이 임계값 0.5 이상(1인 voxel만)인 것들만 추려서 voxels에 담음

# Plotter로 인터랙티브 시각화
plotter = pv.Plotter()
plotter.add_mesh(
    voxels,
    color="lightgray",
    opacity=0.8,
    show_edges=False
)
plotter.show_grid()
plotter.show(title="3D Occupancy Grid")
plotter.close() 

# 맵 정보를 전역 좌표계로 변환할 때는:
# world_coord = origin + (idx + 0.5) * voxel_size


# 시작/목표 월드 좌표 → 그리드 인덱스로 변환
start_world = np.array([x_s, y_s, z_s])
goal_world  = np.array([x_g, y_g, z_g])
start_idx = tuple(((start_world - voxel_grid.origin) / voxel_size - min_idx).astype(int))
goal_idx  = tuple(((goal_world  - voxel_grid.origin) / voxel_size - min_idx).astype(int))


# 사용할 z 레벨 선택
k0 = 1  # 0 또는 1
# 2D occupancy slice
occ2d   = occ[:, :, k0]        # shape = (51,51)
start2d = start_idx[:2]        # (i_s, j_s)
goal2d  = goal_idx[:2]         # (i_g, j_g)

# 2D 벽 마진 두기
tank_radius = 2.0       # 전차 반경 (m), 전장: 10.8, 차체: 7.5, 전폭: 3.6, 전고: 2.4
voxel_size   = 2.0       # 셀 크기 (m)
n_margin     = int(np.ceil(tank_radius / voxel_size))

# 2D용: 정사각 커널
struct2d = np.ones((2*n_margin+1, 2*n_margin+1), dtype=bool)

# -- 2D slice에만 마진 적용 --
occ2d_inflated = binary_dilation(occ2d, structure=struct2d)

# 실행
# path2d = astar_2d(occ2d, start2d, goal2d)
path2d = astar_2d(occ2d_inflated, start2d, goal2d) # 장애물 마진 넣기
if path2d:
    print('2D 경로 찾음')
    # print("2D 경로 인덱스:", path2d)
else:
    print("2D에서도 경로를 찾지 못했습니다.")


# 3d맵 위에 2d path 그리기
path_mask = np.zeros_like(occ, dtype=np.uint8)
for (i, j) in path2d:
    path_mask[i, j, k0] = 1
grid.cell_data["path"] = path_mask.ravel(order="F")
path_voxels = grid.threshold(0.5, scalars="path")

plotter = pv.Plotter() 
plotter.add_mesh(voxels,      color="lightgray", opacity=0.6) 
plotter.add_mesh(path_voxels, color="yellow",     opacity=1.0)
plotter.show_grid()
plotter.show(title="3D Occupancy + 2D Path")
plotter.close()



# # 2-3) A* 호출
# path_idx = astar_3d(occ, start_idx, goal_idx)
# if not path_idx:
#     print("경로를 찾지 못했습니다.")
# else:
#     # 인덱스 경로 → 월드 좌표 경로
#     path_world = [
#         voxel_grid.origin + (np.array(idx) + min_idx + 0.5) * voxel_size
#         for idx in path_idx
#     ]
#     print("월드 좌표 경로:", path_world)

# 3D용: 정육면체 커널
# struct3d = np.ones((2*n_margin+1, 2*n_margin+1, 2*n_margin+1), dtype=bool)
# -- 전체 3D grid에 적용하고 싶다면 --
# occ_inflated = binary_dilation(occ, structure=struct3d)
# occ = occ_inflated.astype(np.uint8)
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




def astar_3d(occ, start, goal):
    print('ooo', occ[0][0:10], occ[1][0:10], occ[2][0:10],)
    print('sss', start[0],start[1],start[2],)
    print('ggg', goal[0], goal[1], goal[2])
    """
    occ    : numpy (nx,ny,nz) occupancy grid
    start  : (i,j,k) 시작 인덱스
    goal   : (i,j,k) 도착 인덱스
    return : 인덱스 리스트 경로 혹은 빈 리스트
    """
    nx, ny, nz = occ.shape
    print('nnn', nx, ny, nz)
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
