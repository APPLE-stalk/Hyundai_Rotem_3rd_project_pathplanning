import numpy as np
import heapq
import pyvista as pv
from scipy.ndimage import binary_dilation # 장애물 마진 용
import time
import matplotlib.pyplot as plt

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

# 2D path 시각화
def plot_2d_data(data: np.ndarray):
    """
    data: shape = (N, 2) 배열
    1열 → x, 2열 → y로 간주하고 산점도로 시각화
    """
    x = data[:, 0]
    y = data[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y,
                s=20,          # 점 크기
                alpha=0.7,     # 투명도
                edgecolor='k', # 윤곽선
                label='samples')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of Shape (N, 2) Data')
    plt.grid(True)
    plt.axis('equal')    # x, y 스케일 동일비율
    plt.legend()
    plt.tight_layout()
    plt.show()
    plotter.close()  # 명시적으로 닫아 줍니다

# 3D path 시각화
def plot_3d_data(data: np.ndarray):
    x = data[:, 0]
    z = data[:, 1]
    y = data[:, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, z, y,
            s=20,          # 점 크기
            alpha=0.7,     # 투명도
            edgecolor='k', # 윤곽선
            label='samples')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('3D Scatter Plot of Shape (N, 3) Data')
    ax.legend()

    # 3D 축 동일 비율 설정
    # max_range = np.max([x.max() - x.min(), z.max() - z.min(), y.max() - y.min()])
    # mid_x = (x.max() + x.min()) * 0.5
    # mid_z = (z.max() + z.min()) * 0.5
    # mid_y = (y.max() + y.min()) * 0.5
    # ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    # ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    # ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    

    plt.tight_layout()
    plt.show()
    plotter.close()  # 명시적으로 닫아 줍니다


# 등간격(유클리디안) 선형 보간
import numpy as np

def resample_by_arc_length_2d(data: np.ndarray,
                        spacing: float,
                        include_endpoint: bool = True
                        ) -> np.ndarray:
    """
    2D 폴리라인을 아크 길이에 따라 일정 간격으로 재샘플링합니다.

    Parameters
    ----------
    data : np.ndarray, shape (N,2)
        원본 폴리라인 점들 (x, y).
    spacing : float
        재샘플링 간격 (아크 길이 기준). 0 < spacing <= 전체 길이.
    include_endpoint : bool, default True
        True면 마지막 점(총 길이)을 반드시 포함합니다.

    Returns
    -------
    np.ndarray, shape (M,2)
        아크 길이 방향으로 spacing 간격 보간된 (x, y) 점들.
    """
    if spacing <= 0:
        raise ValueError("spacing은 0보다 커야 합니다.")
    
    # 1) x,y 추출 및 세그먼트 길이 계산
    x_old, y_old = data[:,0], data[:,1]
    dx = np.diff(x_old)
    dy = np.diff(y_old)
    seg_len = np.hypot(dx, dy)              # 각 구간 길이
    s_old = np.concatenate(([0], np.cumsum(seg_len)))
    total_len = s_old[-1]

    if spacing > total_len:
        # 간격이 너무 크면 시작/끝 두 점만 반환
        return np.array([[x_old[0], y_old[0]], [x_old[-1], y_old[-1]]])

    # 2) 새로 뽑을 s 위치 생성
    s_new = np.arange(0, total_len, spacing)
    if include_endpoint and s_new[-1] < total_len:
        s_new = np.append(s_new, total_len)

    # 3) s → x, y 선형 보간
    x_new = np.interp(s_new, s_old, x_old)
    y_new = np.interp(s_new, s_old, y_old)

    return np.vstack((x_new, y_new)).T

def resample_by_arc_length_3d(data: np.ndarray,
                            spacing: float,
                            include_endpoint: bool = True) -> np.ndarray:

    if spacing <= 0:
        raise ValueError("spacing은 0보다 커야 합니다.")
    
    # 좌표 분리
    x_old, y_old, z_old = data[:,0], data[:,1], data[:,2]
    # 각 구간 길이 계산
    dx = np.diff(x_old)
    dy = np.diff(y_old)
    dz = np.diff(z_old)
    seg_len = np.sqrt(dx**2 + dy**2 + dz**2)
    s_old = np.concatenate(([0], np.cumsum(seg_len)))
    total_len = s_old[-1]

    # spacing이 전체 길이보다 클 경우 시작/끝만 반환
    if spacing > total_len:
        return np.array([[x_old[0], y_old[0], z_old[0]],
                        [x_old[-1], y_old[-1], z_old[-1]]])

    # 새로 뽑을 s 위치 생성
    s_new = np.arange(0, total_len, spacing)
    if include_endpoint and s_new[-1] < total_len:
        s_new = np.append(s_new, total_len)

    # s → x, y, z 선형 보간
    x_new = np.interp(s_new, s_old, x_old)
    y_new = np.interp(s_new, s_old, y_old)
    z_new = np.interp(s_new, s_old, z_old)

    return np.vstack((x_new, y_new, z_new)).T

# ====================== point cloud 파일(.ply) 불러오기 ======================
# pcd = o3d.io.read_point_cloud("./downsampling_outliner_point_cloud_(100_100)_1.0.ply")

# point cloud 시각화(무지개)
# o3d.visualization.draw_geometries([pcd])

# point -> voxel로 변환
# voxel_size = 2
# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

# voxel 시각화(검은색)
# o3d.visualization.draw_geometries([voxel_grid])



# ====================== voxel grid 파일(.ply) 불러오기 ======================
# voxel_size = 2
# voxel_grid = o3d.io.read_voxel_grid("flat(100x100)_vg(2.0x2.0)_visual.ply")

# voxels = voxel_grid.get_voxels()
# print('voxel 갯수: ', len(voxels)) # 2944개
# print('voxel 1개 예시: ', voxels[0])
# print('voxel 좌표', voxels[0].grid_index, 'voxel 색상', voxels[0].color) # voxel 데이터 접근 방법: 
# 맵 크기 계산 (셀 수)
# 각 voxel.grid_index 는 (i,j,k) 정수 인덱스
# indices = np.array([v.grid_index for v in voxels])
# min_idx = indices.min(axis=0) # axis=0의미: 같은 열끼리 비교하여 min값을 구하시오
# max_idx = indices.max(axis=0) + 1 # 슬라이싱 할 때 마지막 인덱시 미포함 고려하여 +1씩 해주기
# dims = (max_idx - min_idx).astype(int)  # (nx, ny, nz) 각 축별 길이값 구하기
# nx, ny, nz = dims


# Occupancy 배열 (0=free, 1=occupied)
# 0==free => 갈 수 있음 // 1==occupied => 갈 수 없음
# occ = np.zeros(dims, dtype=np.uint8)
# for v in voxels:
#     i, j, k = np.array(v.grid_index) - min_idx
#     occ[i, j, k] = 1


# ====================== OGM(.npy) 불러오기 ======================
st_time = time.perf_counter()
occ = np.load("OGM_2.0x2.0_LH.npy")
ed_time = time.perf_counter()
print(f'파일 읽기 소요시간: {ed_time - st_time:.6f}초')
voxel_size = 2

# PyVista ImageData 생성
nx, nz, ny = occ.shape # 우, 전방, 높이
grid = pv.ImageData(
    dimensions=(nx+1, nz+1, ny+1),
    spacing=(voxel_size, voxel_size, voxel_size),
    origin=(0.0, 0.0, 0.0)
)
grid.cell_data["occ"] = occ.ravel(order="F")
voxels = grid.threshold(0.5, scalars="occ")



# ====================== OGM 시각화 ======================
map_plotter = pv.Plotter()
map_plotter.add_mesh(
    voxels,
    color="lightgray",
    opacity=0.6,
    show_edges=False
)
map_plotter.show_grid(
    xtitle="X (m)",
    ytitle="Z (m)",
    ztitle="Y (m)",
    show_xaxis=True,
    show_yaxis=True,
    show_zaxis=True,
    show_xlabels=True,
    show_ylabels=True,
    show_zlabels=True
)
map_plotter.show(title="3D Occupancy Grid (Loaded)")
map_plotter.close()




# ====================== 전처리(2D slicing) ======================
st_time = time.perf_counter()
x_s = 60 # 시작 점 [우, 전방, 높이]
z_s = 28
y_s = 10


x_g = 90 # 목표 점 [우, 전방, 높이]
z_g = 80
y_g = 60


# 시작/목표 월드 좌표 → 그리드 인덱스로 변환
start_world = np.array([x_s, z_s, y_s])
goal_world  = np.array([x_g, z_g, y_g])
start_idx = tuple(((start_world - 0) / voxel_size - 0).astype(int))
goal_idx  = tuple(((goal_world  - 0) / voxel_size - 0).astype(int))


# 사용할 z 레벨 선택
k0 = 1  # 0 또는 1
print("before slicing shape: ", np.shape(occ))

# LH 좌표계 map 
occ2d = occ[:, :, k0]                   # 우, 전방, 높이
print("after slicing shape: ", np.shape(occ))
start2d = (start_idx[0], start_idx[1])  # 우, 전방
goal2d  = (goal_idx[0],  goal_idx[1])
ed_time = time.perf_counter()
print(f'slicing 소요시간: {ed_time - st_time:.6f}초')



# ====================== binary_dilation 이전 시각화 ======================
plt.figure(figsize=(6,6))
plt.imshow(occ2d.T,         # 전방(y) 축이 위로 오도록 전치
        origin='lower',  # 원점이 왼쪽 아래
        cmap='gray_r')   # 0→흰색(free), 1→검은색(occupied)
plt.colorbar(label='Occupancy')
plt.xlabel('X index')
plt.ylabel('Z index')
plt.title(f'before binary_dilation')
plt.grid(False)
plt.show()




# ====================== 2D 벽 마진 주기(binary_dilation) ======================
st_time = time.perf_counter()

tank_radius = 2.0       # 전차 반경 (m), 전장: 10.8, 차체: 7.5, 전폭: 3.6, 전고: 2.4
voxel_size   = 2.0       # 셀 크기 (m)
n_margin     = int(np.ceil(tank_radius / voxel_size))

# 2D용: 정사각 커널
struct2d = np.ones((2*n_margin+1, 2*n_margin+1), dtype=bool)

# 2D slice에만 마진 적용
occ2d_inflated = binary_dilation(occ2d, structure=struct2d)
ed_time = time.perf_counter()
print(f'margin 연산 소요시간: {ed_time - st_time:.6f}초')



# ====================== binary_dilation 이후 시각화 ======================
plt.figure(figsize=(6,6))
plt.imshow(occ2d_inflated.T,         # 전방(y) 축이 위로 오도록 전치
        origin='lower',  # 원점이 왼쪽 아래
        cmap='gray_r')   # 0→흰색(free), 1→검은색(occupied)
plt.colorbar(label='Occupancy')
plt.xlabel('X index')
plt.ylabel('Z index')
plt.title(f'after binary_dilation (3x3)')
plt.grid(False)
plt.show()



# ====================== 2D A* 알고리즘 ======================
st_time = time.perf_counter()
# path2d = astar_2d(occ2d, start2d, goal2d) # 장애물 마진 없이 A*
path2d = astar_2d(occ2d_inflated, start2d, goal2d) # 장애물 마진 넣어 A*
ed_time = time.perf_counter()
print(f'A* 소요시간: {ed_time - st_time:.6f}초')
if path2d:
    print('2D 경로 찾음')
    print("2D 경로 인덱스:", path2d)
    
    path_arr_2d = np.array(path2d, dtype=int)  # shape = (N, 2)
    
    # 3D로 확장 위해 y(높이)정보 추가해줌
    # k0 컬럼 생성
    k0_col = np.full((path_arr_2d.shape[0], 1), k0, dtype=int)
    # 수평으로 합쳐서 shape = (N,3) 배열 만들기
    path_arr_3d = np.hstack((path_arr_2d, k0_col))
else:
    print("2D에서 경로를 찾지 못했습니다.")


# 3d맵 위에 2d path 그리기
path_mask = np.zeros_like(occ, dtype=np.uint8)
for (i, j) in path_arr_2d:
    path_mask[i, j, k0] = 1

sp_mask = np.zeros_like(occ, dtype=np.uint8) # 시작점
gp_mask = np.zeros_like(occ, dtype=np.uint8) # 도착점
sp_mask[(start_idx[0], start_idx[1], k0)] = 1
gp_mask[(goal_idx[0], goal_idx[1], k0)]  = 1

grid.cell_data["path"] = path_mask.ravel(order="F")
grid.cell_data["sp"]   = sp_mask.ravel(order="F")
grid.cell_data["gp"]   = gp_mask.ravel(order="F")

path_voxels    = grid.threshold(0.5, scalars="path")
path_voxels_sp = grid.threshold(0.5, scalars="sp")
path_voxels_gp = grid.threshold(0.5, scalars="gp")

plotter = pv.Plotter()
plotter.add_mesh(voxels,        color="lightgray", opacity=0.6, show_edges=False)
plotter.add_mesh(path_voxels,   color="yellow",     opacity=1.0,   show_edges=False)
plotter.add_mesh(path_voxels_sp, color="blue",     opacity=1.0,   show_edges=False)
plotter.add_mesh(path_voxels_gp, color="red",      opacity=1.0,   show_edges=False)
plotter.show_grid(
    xtitle="X (m)",
    ytitle="Z (m)",
    ztitle="Y (m)",
    show_xaxis=True,
    show_yaxis=True,
    show_zaxis=True,
    show_xlabels=True,
    show_ylabels=True,
    show_zlabels=True
)
plotter.show(title="3D Occupancy + 2D Path (Start/Goal)")
plotter.close()



# ====================== Linear Interpolation ======================
## 2d
print(f"Before Linear Interpolation 2d way point 갯수: {len(path_arr_2d)} 개")
plot_2d_data(path_arr_2d) # before interpolation 시각화 
st_time = time.perf_counter()
interpolated_2d_path_arr = resample_by_arc_length_2d(path_arr_2d, spacing = 0.5)
ed_time = time.perf_counter()
print(f'2d Linear Interpolation 소요시간: {ed_time - st_time:.6f}초')
plot_2d_data(interpolated_2d_path_arr) # after interpolation 시각화 
print(f"After Linear Interpolation way point 갯수: {len(interpolated_2d_path_arr)} 개")


## 3d
print(f"Before Linear Interpolation 3d way point 갯수: {len(path_arr_3d)} 개")
plot_3d_data(path_arr_3d) # before interpolation 시각화
st_time = time.perf_counter()
interpolated_3d_path_arr = resample_by_arc_length_3d(path_arr_3d, spacing = 0.5)
ed_time = time.perf_counter()
print(f'3d Linear Interpolation 소요시간: {ed_time - st_time:.6f}초')
plot_3d_data(interpolated_3d_path_arr) # after interpolation 시각화
print(f"After Linear Interpolation way point 갯수: {len(interpolated_3d_path_arr)} 개")




# 패스 저장(.csv 방식)
np.savetxt('interpolated_path_2d.csv', interpolated_2d_path_arr, delimiter=',', header='x,z', comments='')  # comments='' 로 '#' 주석 제거  
np.savetxt('interpolated_path_3d.csv', interpolated_3d_path_arr, delimiter=',', header='x,z,y', comments='')  # comments='' 로 '#' 주석 제거  
    
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
# neighbor_offsets = np.array([
#     [ 1,  0,  0],
#     [-1,  0,  0],
#     [ 0,  1,  0],
#     [ 0, -1,  0],
#     [ 0,  0,  1],
#     [ 0,  0, -1],
# ], dtype=int)




# def astar_3d(occ, start, goal):
#     print('ooo', occ[0][0:10], occ[1][0:10], occ[2][0:10],)
#     print('sss', start[0],start[1],start[2],)
#     print('ggg', goal[0], goal[1], goal[2])
#     """
#     occ    : numpy (nx,ny,nz) occupancy grid
#     start  : (i,j,k) 시작 인덱스
#     goal   : (i,j,k) 도착 인덱스
#     return : 인덱스 리스트 경로 혹은 빈 리스트
#     """
#     nx, ny, nz = occ.shape
#     print('nnn', nx, ny, nz)
#     open_set = []
#     heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, None))
#     came_from = {}
#     g_score = {start: 0}
    
#     while open_set:
#         f, g, current, parent = heapq.heappop(open_set)
#         if current in came_from:
#             continue
#         came_from[current] = parent
        
#         if current == goal:
#             # 경로 복원
#             path = []
#             node = current
#             while node:
#                 path.append(node)
#                 node = came_from[node]
#             return path[::-1]
        
#         for d in neighbor_offsets:
#             nb = tuple(np.array(current) + d)
#             # 경계 검사
#             if not (0 <= nb[0] < nx and 0 <= nb[1] < ny and 0 <= nb[2] < nz):
#                 continue
#             # 충돌 검사
#             if occ[nb]:
#                 continue
#             tentative_g = g + 1  # 등간격 가정
#             if tentative_g < g_score.get(nb, np.inf):
#                 g_score[nb] = tentative_g
#                 f_score = tentative_g + heuristic(nb, goal)
#                 heapq.heappush(open_set, (f_score, tentative_g, nb, current))
#     return []  # 경로 없음
