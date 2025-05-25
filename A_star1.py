import numpy as np
import heapq
import pyvista as pv
from scipy.ndimage import binary_dilation # 장애물 마진 용
import time
import matplotlib.pyplot as plt
from scipy import ndimage # 소벨마스크

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
    data = np.asarray(data)
    x = data[:, 0]
    z = data[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, z,
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
    data = np.asarray(data)
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

def resample_by_arc_length_2d(data: np.ndarray,
                        spacing: float,
                        include_endpoint: bool = True
                        ) -> np.ndarray:
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




def wall_mask_from_ogm(ogm, voxel_size, grad_thresh):
    """
    ogm       : (nx, ny, nz) float or bool ndarray (occupancy probability or binary)
    voxel_size: 실공간 한 칸 크기 (m)
    grad_thresh: gradient magnitude 임계값 (단위: occupancy change per m)
    returns   : bool ndarray, 같은 shape, True인 곳이 “벽” 셀
    """
    # 1) 이진화
    occ = (ogm > 0.5).astype(float)

    # 2) Sobel 미분 (3D)
    gx = ndimage.sobel(occ, axis=0, mode='nearest') / (8 * voxel_size)
    gy = ndimage.sobel(occ, axis=1, mode='nearest') / (8 * voxel_size)
    gz = ndimage.sobel(occ, axis=2, mode='nearest') / (8 * voxel_size)

    # 3) gradient 크기
    grad = np.sqrt(gx**2 + gy**2 + gz**2)

    # 4) 임계값으로 벽 셀 마스크
    return grad > grad_thresh

def ogm_to_height_map(ogm, voxel_size, axis=2):
    """
    ogm        : (nx, ny, nz) bool ndarray
    voxel_size : 한 칸 크기 (m)
    axis       : 높이 축 인덱스 (예: z가 세 번째면 axis=2)
    returns    : (nx, ny) float ndarray, 높이값(m)
    """
    # argmax는 첫 True 인덱스를 반환하니, 없으면 0 높이로 치환
    occ = ogm.astype(bool)
    idx_max = np.argmax(occ, axis=axis)
    # voxel 인덱스를 실제 높이(m)로 변환
    return idx_max * voxel_size

def slope_mask(H, h, theta_deg):
    """
    H         : (nx, nz) float ndarray, 높이맵 (nan 허용)
    h         : 수평 해상도 (voxel_size, m)
    theta_deg : 임계 경사각 (°)

    returns   : bool ndarray, True인 곳이 "벽" 또는 경사도가 너무 큰 셀
    """
    # 1) nan을 최저 높이값으로 채워서 sobel 처리
    H_min = np.nanmin(H)
    H_filled = np.where(np.isnan(H), H_min, H)

    # 2) Sobel 필터로 기울기 계산 (axis=0: x방향, axis=1: z방향)
    #    sobel 커널의 분모가 8*h인 이유는 Sobel 연산 결과가 픽셀당 차분값을
    #    4*h 로 나눈 뒤 다시 2로 나눈 형태이기 때문입니다.
    gx = ndimage.sobel(H_filled, axis=0, mode='nearest') / (8 * h)
    gz = ndimage.sobel(H_filled, axis=1, mode='nearest') / (8 * h)

    # 3) 기울기 크기로부터 각도 계산
    slope_rad = np.arctan(np.sqrt(gx**2 + gz**2))

    # 4) 임계 각도 이상인 곳만 True
    return slope_rad > np.deg2rad(theta_deg)


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
# ogm= np.load("OGM_2.0x2.0_LH.npy")
# ogm= np.load("OGM_kd_concept.npy")



# meta = np.load('OGM_maze1_with_meta.npz')
meta = np.load('../mapping/3D_LiDAR_fake/open3d/flat_and_hills_whole_25_05_13/flat_and_hills_whole_OGM(0.5_0.5)_with_meta.npz')
ogm  = meta['data']
origin     = meta['origin']# + meta['resolution'] * (0, 0, 0.5)
voxel_size = meta['resolution']
print(origin)
print(voxel_size)


# origin = (10.5, 12.5, 0.0)

ed_time = time.perf_counter()
print(f'파일 읽기 소요시간: {ed_time - st_time:.6f}초')

# PyVista ImageData 생성
nx, nz, ny = ogm.shape                              # 우, 전방, 높이 #칸수, 칸수*voxel_size가 진짜 길이
grid = pv.ImageData(                                # PyVista에서 “격자(point) + 셀(cell)” 구조를 만드는 클래스
    dimensions=(nx+1, nz+1, ny+1),                  # 셀 갯수에 따른 포인트 갯수
    spacing=(voxel_size, voxel_size, voxel_size),   # 포인트 간격
    origin=origin                          # 격자의 (0, 0, 0)을 월드 좌표계의 (0, 0, 0)으로 지정
)
grid.cell_data["occ"] = ogm.ravel(order="F")
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


# # ====================== 2D Height-Map 만들기 ======================
# HM = build_top_height_map(voxels, voxel_grid.origin, voxel_size, nx, nz)
# HM_slope = slope_mask(HM, voxel_size, 31, 7)


# import matplotlib.pyplot as plt
# plt.imshow(HM, origin='lower')   # 배열의 (0,0)을 좌하단에 두려면 origin='lower'
# plt.colorbar(label='cell value')
# plt.title('OGM visualized with imshow')
# plt.xlabel('x index')
# plt.ylabel('z index')
# plt.show()

# plt.imshow(HM_slope, origin='lower')   # 배열의 (0,0)을 좌하단에 두려면 origin='lower'
# plt.colorbar(label='cell value')
# plt.title('slope')
# plt.xlabel('x index')
# plt.ylabel('z index')
# plt.show()

# 2D 높이맵 생성
H = ogm_to_height_map(ogm, voxel_size=0.5, axis=2)

# 기존 slope_mask 사용 예
wall_mask_2d = slope_mask(H, h=0.5, theta_deg=31)

# 3) 시각화
plt.figure(figsize=(12, 5))

# 높이 맵 시각화
plt.subplot(1, 2, 1)
plt.imshow(H.T, origin='lower', interpolation='nearest')
plt.title('Height Map')
plt.xlabel('X index')
plt.ylabel('Z index')
plt.colorbar(label='Height (m)')

# 벽 마스크 시각화
plt.subplot(1, 2, 2)
plt.imshow(wall_mask_2d.T, origin='lower', interpolation='nearest')
plt.title(f'Wall Mask (>{31}°)')
plt.xlabel('X index')
plt.ylabel('Z index')
plt.colorbar(label='Wall Mask (True=Wall)')

plt.tight_layout()
plt.show()


# ====================== 전처리(2D slicing) ======================
st_time = time.perf_counter()
# x_s = 45 # 시작 점 [우, 전방, 높이]
# z_s = 27
# y_s = 10

# x_g = 90 # 목표 점 [우, 전방, 높이]
# z_g = 80
# y_g = 10

# x_s = 11.5 # 시작 점 [우, 전방, 높이]
# z_s = 13.5
# y_s = 0

# x_g = 16.5 # 목표 점 [우, 전방, 높이]
# z_g = 19.5
# y_g = 0

x_s = 60 # 시작 점 [우, 전방, 높이]
z_s = 27
y_s = 8

# x_g = 145 # 목표 점 [우, 전방, 높이]
# z_g = 110
# y_g = 8
x_g = 280 # 목표 점 [우, 전방, 높이]
z_g = 280
y_g = 8


# 시작/목표 월드 좌표 → 그리드 인덱스로 변환
start_world = np.array([x_s, z_s, y_s])
goal_world  = np.array([x_g, z_g, y_g])
start_idx = tuple(((start_world - origin) / voxel_size).astype(int)) # 월드좌표(m) -> 셀 인덱스(칸)
goal_idx  = tuple(((goal_world  - origin) / voxel_size).astype(int))


# 사용할 z 레벨 선택
# k0 = 5  # 0 또는 1
k0 = 15
print("before slicing shape: ", np.shape(ogm))

# LH 좌표계 map 
# ogm2d = ogm[:, :, k0]                   # 우, 전방, 높이
ogm2d = wall_mask_2d
print("after slicing shape: ", np.shape(ogm2d)) # 우, 전방
start2d = (start_idx[0], start_idx[1])  # 우, 전방 (인덱스 단위)
goal2d  = (goal_idx[0],  goal_idx[1])
ed_time = time.perf_counter()
print(f'slicing 소요시간: {ed_time - st_time:.6f}초')



# ====================== binary_dilation 이전 시각화 ======================
Nx, Nz = ogm2d.shape # 우, 전방
# extent = [0, Nx * voxel_size,     # x축: 0m ~ Nx*voxel_size m
#           0, Nz * voxel_size]     # z축: 0m ~ Nz*voxel_size m

plt.figure(figsize=(6,6))
plt.imshow(ogm2d.T,                 # 전방(y) 축이 위로 오도록 전치
        origin='lower',             # 원점이 왼쪽 아래
        cmap='gray_r')              # 0→흰색(free), 1→검은색(occupied)
        # extent=extent)            # 축 단위 m 단위로 변경            
plt.colorbar(label='Occupancy')
plt.xlabel('X index')
plt.ylabel('Z index')
plt.title('before binary_dilation')
plt.grid(False)
plt.show()




# ====================== 2D 벽 마진 주기(binary_dilation) ======================
st_time = time.perf_counter()

tank_radius = 3.0       # 전차 반경 (m), 전장: 10.8, 차체: 7.5, 전폭: 3.6, 전고: 2.4
# voxel_size  = 2.0       # 셀 크기 (m)
n_margin    = int(np.ceil(tank_radius / voxel_size))

# 2D용: 정사각 커널
struct2d = np.ones((2*n_margin+1, 2*n_margin+1), dtype=bool)

# 2D slice에만 마진 적용
occ2d_inflated = binary_dilation(ogm2d, structure=struct2d)
ed_time = time.perf_counter()
print(f'margin 연산 소요시간: {ed_time - st_time:.6f}초')



# ====================== binary_dilation 이후 시각화 ======================
plt.figure(figsize=(6,6))
plt.imshow(occ2d_inflated.T,        # 전방(y) 축이 위로 오도록 전치
        origin='lower',             # 원점이 왼쪽 아래
        cmap='gray_r')              # 0→흰색(free), 1→검은색(occupied)
        # extent=extent)            # m단위로 변환
plt.colorbar(label='Occupancy')
plt.xlabel('X index')
plt.ylabel('Z index')
plt.title(f'after binary_dilation (3x3)')
plt.grid(False)
plt.show()



# ====================== 2D A* 알고리즘 ======================
st_time = time.perf_counter()
path2d = astar_2d(ogm2d, start2d, goal2d) # 장애물 마진 없이 A*
# path2d = astar_2d(occ2d_inflated, start2d, goal2d) # 장애물 마진 넣어 A*
ed_time = time.perf_counter()
print(f'A* 소요시간: {ed_time - st_time:.6f}초')
if path2d:
    print('2D 경로 찾음')
    path_2d_arr = np.array(path2d, dtype=int)  # shape = (N, 2)
    
    
    k0_col = np.full((path_2d_arr.shape[0], 1), k0, dtype=int) # 라이다 높이 때문에.... 8로 함
    path_arr_3d = np.hstack((path_2d_arr, k0_col))
    print("3차원공간의 2D 경로 인덱스:", path_arr_3d)
else:
    print("2D에서 경로를 찾지 못했습니다.")

# A*로 구한 path를 world좌표로 변환하기
# path_world_2d = []  # 결과를 담을 리스트
# for idx in path_2d_arr:
#     i, j = idx
#     # 셀의 중앙 좌표 = origin + (index + 0.5)*voxel_size
#     x = origin[0] + (i + 0.5) * voxel_size
#     z = origin[1] + (j + 0.5) * voxel_size
#     path_world_2d.append((x, z))
    
path_world_3d = []  # 결과를 담을 리스트
for idx in path_arr_3d:
    i, k, j = idx
    # 셀의 중앙 좌표 = origin + (index + 0.5)*voxel_size
    x = origin[0] + (i) * voxel_size
    z = origin[1] + (k) * voxel_size
    y = origin[2] + (k + 0.5) * voxel_size
    y = 10 # j,                                                                  라이다 높이 때문에 8로 함
    path_world_3d.append((x, z, y))
    

# 3d맵 위에 2d path 그리기
path_mask = np.zeros_like(ogm, dtype=np.uint8)
for (i, j) in path_2d_arr:
    path_mask[i, j, k0] = 1

sp_mask = np.zeros_like(ogm, dtype=np.uint8) # 시작점
gp_mask = np.zeros_like(ogm, dtype=np.uint8) # 도착점
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
plotter.add_mesh(path_voxels,   color="red",     opacity=1.0,   show_edges=False)
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
# print(f"Before Linear Interpolation 2d way point 갯수: {len(path_2d_arr)} 개")
# plot_2d_data(path_2d_arr) # before interpolation 시각화 
# st_time = time.perf_counter()
# interpolated_2d_path_arr = resample_by_arc_length_2d(np.asarray(path_2d_arr), spacing = 0.5)
# ed_time = time.perf_counter()
# print(f'2d Linear Interpolation 소요시간: {ed_time - st_time:.6f}초')
# plot_2d_data(interpolated_2d_path_arr) # after interpolation 시각화 
# print(f"After Linear Interpolation way point 갯수: {len(interpolated_2d_path_arr)} 개")


## 3d
print(f"Before Linear Interpolation 3d way point 갯수: {len(path_arr_3d)} 개")
plot_3d_data(path_world_3d) # before interpolation 시각화
st_time = time.perf_counter()
interpolated_3d_path_arr = resample_by_arc_length_3d(np.asarray(path_world_3d), spacing = 0.5)
ed_time = time.perf_counter()
print(f'3d Linear Interpolation 소요시간: {ed_time - st_time:.6f}초')
print(f"After Linear Interpolation way point 갯수: {len(interpolated_3d_path_arr)} 개")
plot_3d_data(interpolated_3d_path_arr) # after interpolation 시각화
print(f"After Linear Interpolation way point 갯수: {len(interpolated_3d_path_arr)} 개")




# 패스 저장(.csv 방식)
# np.savetxt('interpolated_path_2d.csv', interpolated_2d_path_arr, delimiter=',', header='x,z', comments='')  # comments='' 로 '#' 주석 제거  
# np.savetxt('interpolated_path_3d.csv', interpolated_3d_path_arr, delimiter=',', header='x,z,y', comments='', fmt='%.2f')  # comments='' 로 '#' 주석 제거  
# np.savetxt('interpolated_path_3d_maze1.csv', interpolated_3d_path_arr, delimiter=',', header='x,z,y', comments='', fmt='%.2f')  # comments='' 로 '#' 주석 제거  
np.savetxt('interpolated_path_3d_flat_and_hills_whole.csv', interpolated_3d_path_arr, delimiter=',', header='x,z,y', comments='', fmt='%.2f')  # comments='' 로 '#' 주석 제거 

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
