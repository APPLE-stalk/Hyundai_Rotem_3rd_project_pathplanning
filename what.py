import numpy as np
import heapq
import pyvista as pv
from scipy.ndimage import binary_dilation   # 장애물 마진
from scipy.ndimage import binary_erosion    # 노이즈 erosion
import time
import matplotlib.pyplot as plt
from scipy import ndimage # 소벨마스크

# 2D OGM에서 A*
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



# 3D path 시각화
def plot_3d_data(data: np.ndarray):
    data = np.asarray(data)
    x = data[:, 0]
    z = data[:, 1]
    y = data[:, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, z, y, s=20, alpha=0.7, edgecolor='k', label='')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    
    ax.set_title('[2D path + height]')
    ax.legend()
    
    # # 숫자 눈금 지정
    # xticks = np.linspace(-10, 310, 6)   # 우
    # yticks = np.linspace(-10, 310, 6)   # 전방
    # zticks = np.linspace(-10, 20, 6)   # 높이
    
    # ax.set_xticks(xticks)
    # ax.set_zticks(zticks)
    # ax.set_yticks(yticks)
    
    ax.set_xlim(-10, 310)
    ax.set_ylim(-10, 310)
    ax.set_zlim(-10, 50)
    
    plt.tight_layout()
    plt.show()



# 3차원 보간
def resample_by_arc_length_3d(data: np.ndarray, spacing: float, include_endpoint: bool = True) -> np.ndarray:
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



# 3D OGM으로 2D height map 생성
def ogm_to_height_map(ogm, voxel_size, axis=2):
    # argmax는 첫 True 인덱스를 반환하니, 없으면 0 높이로 치환
    occ = ogm.astype(bool)
    idx_max = np.argmax(occ, axis=axis)
    # voxel 인덱스를 실제 높이(m)로 변환
    return idx_max * voxel_size



# Voxel간 기울기 측정하여 기동 가능 여부 판단
def slope_mask(H, h, theta_deg):
    # H         : (nx, nz) float ndarray, 높이맵 (nan 허용)
    # h         : 수평 해상도 (voxel_size, m)
    # theta_deg : 임계 경사각 (°)
    # returns   : bool ndarray, True인 곳이 "벽" 또는 경사도가 너무 큰 셀

    # nan인 공간 최솟값 대입
    H_min = np.nanmin(H)
    H_filled = np.where(np.isnan(H), H_min, H)
    
    # Sobel 필터로 기울기 계산 (axis=0: x방향, axis=1: z방향)
    # sobel 커널의 분모가 8*h인 이유는 Sobel 연산 결과가 픽셀당 차분값을
    # 4*h 로 나눈 뒤 다시 2로 나눈 형태이기 때문입니다.
    gx = ndimage.sobel(H_filled, axis=0, mode='nearest') / (8 * h)
    gz = ndimage.sobel(H_filled, axis=1, mode='nearest') / (8 * h)
    
    # 기울기 크기로부터 각도 계산
    slope_rad = np.arctan(np.sqrt(gx**2 + gz**2))
    
    # 임계 각도 이상인 곳만 True
    return slope_rad > np.deg2rad(theta_deg)





# ====================== OGM(.npy) 불러오기 ======================
st_time = time.perf_counter()
meta = np.load('./flat_and_hills_whole_OGM(0.5)_with_meta.npz') # OGM, voxel_grid원점, OGM 큐브 사이즈 정보
ogm  = meta['data']
origin     = meta['origin']                                     # 필요시 조정할 것
voxel_size = meta['resolution']
print('voxel_grid 원점: ', origin)
print('voxel_size: ', voxel_size)
ed_time = time.perf_counter()
print(f'OGM_with_meta파일 읽기 소요시간: {ed_time - st_time:.6f}초')

# PyVista 시각화에 필요한 Data 생성
nx, nz, ny = ogm.shape                              # 우, 전방, 높이의 칸수, 칸수*voxel_size가 실제 길이
grid = pv.ImageData(                                # PyVista에서 [격자(point) + 셀(cell)] 구조를 만드는 클래스
    dimensions=(nx+1, nz+1, ny+1),                  # 셀 갯수에 따른 포인트 갯수
    spacing=(voxel_size, voxel_size, voxel_size),   # 포인트 간격
origin=origin                                       # 격자의 (0, 0, 0)을 월드 좌표계의 (0, 0, 0)으로 지정
)
grid.cell_data["occ"] = ogm.ravel(order="F")
voxels = grid.threshold(0.5, scalars="occ")



# ====================== load한 OGM 시각화 ======================
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



# ====================== 2D Height-Map 만들기 ======================
H = ogm_to_height_map(ogm, voxel_size=voxel_size, axis=2)       # 3D OGM -> 2D height map
# 파라미터: 3D OGM, voxel_size, 높이 축

wall_mask_2d = slope_mask(H, h=voxel_size, theta_deg=31)        # voxel간 기울기 변화 측정하여 기동 불가지역(벽, 장애물)검출 mask
# 파라미터: 2D height map, voxel_size, 기동 불가 판정 각도


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(H.T, origin='lower', interpolation='nearest')        # 높이 맵
plt.title('Height Map(0.5m)')
plt.xlabel('X index')
plt.ylabel('Z index')
plt.colorbar(label='Height (m)')

# 벽 마스크 시각화
plt.subplot(1, 2, 2)
plt.imshow(wall_mask_2d.T, origin='lower', interpolation='nearest')
plt.title(f'Drivable area(0.5m) ref {31}°')
plt.xlabel('X index')
plt.ylabel('Z index')
plt.colorbar(label='Obstacle Mask(True =Obstacle)')

plt.tight_layout()
plt.show()


# ====================== 전처리(2D slicing) ======================
st_time = time.perf_counter()

x_s = 60                                                                # 시작 점 [우, 전방, 높이]
z_s = 27
y_s = 3

x_g = 280                                                               # 목표 점 [우, 전방, 높이]
z_g = 280
y_g = 3

# 시작/목표 월드 좌표 → 그리드 인덱스로 변환
start_world = np.array([x_s, z_s, y_s])
goal_world  = np.array([x_g, z_g, y_g])
start_idx = tuple(((start_world - origin) / voxel_size).astype(int))    # 월드좌표(m) -> 셀 인덱스(칸)
goal_idx  = tuple(((goal_world  - origin) / voxel_size).astype(int))


# 맵 지면에서 전차 좌표계까지 높이 차이 
k0 = 0                                  # [m]
k0 = int(k0/voxel_size)
ogm2d = wall_mask_2d
start2d = (start_idx[0], start_idx[1])  # 우, 전방 (인덱스 단위)
goal2d  = (goal_idx[0],  goal_idx[1])



# ====================== 처리 이전 시각화 ======================
Nx, Nz = ogm2d.shape # 우, 전방
# extent = [0, Nx * voxel_size,     # x축: 0m ~ Nx*voxel_size m
#           0, Nz * voxel_size]     # z축: 0m ~ Nz*voxel_size m

plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
plt.imshow(ogm2d.T,                 # 전방(y) 축이 위로 오도록 전치
    origin='lower',                 # 원점이 왼쪽 아래
    cmap='gray_r')                  # 0→흰색(free), 1→검은색(occupied)
    # extent=extent)                # 축 단위 m 단위로 변경            
plt.colorbar(label='Occupancy')
plt.xlabel('X index')
plt.ylabel('Z index')
plt.title('before prcessing')
plt.grid(False)



# ====================== 노이즈 없애기 (binary_erosion) ======================
struct2d = np.ones((3, 3), dtype=bool)
occ2d_eroded = binary_erosion(ogm2d, structure=struct2d)    # 2D slice 에 침식 적용
# ====================== 시각화 ======================
plt.subplot(2, 2, 2)
plt.imshow(occ2d_eroded.T,                                  # 전방(y) 축이 위로 오도록 전치
    origin='lower',                                         # 원점이 왼쪽 아래
    cmap='gray_r')                                          # 0→흰색(free), 1→검은색(occupied)
    # extent=extent)                                        # m단위로 변환
plt.colorbar(label='Occupancy')
plt.xlabel('X index')
plt.ylabel('Z index')
plt.title(f'erosion (3x3)')
plt.grid(False)


# ====================== 노이즈 없애기 (binary_dilation) ======================
struct2d = np.ones((3, 3), dtype=bool)
occ2d_eroded_dilated = binary_dilation(occ2d_eroded, structure=struct2d)    # 2D slice에만 마진 적용
# ====================== 시각화 ======================
plt.subplot(2, 2, 3)
plt.imshow(occ2d_eroded_dilated.T,                                          # 전방(y) 축이 위로 오도록 전치
    origin='lower',                                                         # 원점이 왼쪽 아래
    cmap='gray_r')                                                          # 0→흰색(free), 1→검은색(occupied)
    # extent=extent)                                                        # m단위로 변환
plt.colorbar(label='Occupancy')
plt.xlabel('X index')
plt.ylabel('Z index')
plt.title(f'erosion_dilation (3x3)')
plt.grid(False)

# ====================== 전차 반경 고려 장애물 margin dilation 연산용 2D 커널 ======================
tank_radius = 3                                                                             # 전차 반경 (m), 전장: 10.8, 차체: 7.5, 전폭: 3.6, 전고: 2.4
n_margin    = int(np.ceil(tank_radius / voxel_size))
# struct2d = np.ones((2*n_margin+1, 2*n_margin+1), dtype=bool)                                # 2D용: 정사각 커널
struct2d = np.ones((8, 8), dtype=bool)
occ2d_eroded_dilated_inflated = binary_dilation(occ2d_eroded_dilated, structure=struct2d)   # 2D slice에만 마진 적용
# ====================== 시각화 ======================
plt.subplot(2, 2, 4)
plt.imshow(occ2d_eroded_dilated_inflated.T,         # 전방(y) 축이 위로 오도록 전치
    origin='lower',                                 # 원점이 왼쪽 아래
    cmap='gray_r')                                  # 0→흰색(free), 1→검은색(occupied)
    # extent=extent)                                # m단위로 변환
plt.colorbar(label='Occupancy')
plt.xlabel('X index')
plt.ylabel('Z index')
plt.title(f'erosion_dilation_dilation (8x8)')
plt.grid(False)
# plt.tight_layout()
plt.show()




# ====================== 2D A* 알고리즘 ======================
st_time = time.perf_counter()
# path2d = astar_2d(ogm2d, start2d, goal2d)                                 # 장애물 마진 없이 A*
path2d = astar_2d(occ2d_eroded_dilated_inflated, start2d, goal2d)           # 장애물 마진 넣어서 A*
ed_time = time.perf_counter()
print(f'A* 소요시간: {ed_time - st_time:.6f}초')
if path2d:
    print('2D 경로 찾음')
    path_2d_arr = np.array(path2d, dtype=int)  # shape = (N, 2)
    
    k0_col = np.full((path_2d_arr.shape[0], 1), 0, dtype=int)
    for idx in range(len(path_2d_arr)):
        x1, z1 = path_2d_arr[idx]
        k0_col[idx] = np.max(np.where(ogm[x1, z1, :])[0]) + k0
    path_arr_3d = np.hstack((path_2d_arr, k0_col))
    
    path_world_3d = []                                                      # path_arr_3d를 월드 좌표로 변환
    for idx in path_arr_3d:
        i, k, j = idx
        # 셀의 중앙 좌표 = origin + (index + 0.5)*voxel_size
        x = origin[0] + (i) * voxel_size
        z = origin[1] + (k) * voxel_size
        y = origin[2] + (j + 0.5) * voxel_size
        path_world_3d.append((x, z, y))
    print("3차원공간의 3D 경로 인덱스:", path_arr_3d)
else:
    print("2D에서 경로를 찾지 못했습니다.")



# ====================== 2D OGM 맵에서 A* 결과 시각화 ======================
plt.figure(figsize=(6, 6))
plt.imshow(occ2d_eroded_dilated_inflated.T,     # 전방(y) 축이 위로 오도록 전치
        origin='lower',                         # 원점이 왼쪽 아래
        cmap='gray_r')                          # 0→흰색(free), 1→검은색(occupied)
plt.colorbar(label='Occupancy')

# path2d 궤적 추가
xs, zs = zip(*path2d)  
plt.scatter(xs, zs, s=3, c='orange', marker='o',label='path2d') # path
plt.scatter(x_s/voxel_size, z_s/voxel_size, s=3, c='blue', marker='o', label='Start Point') # sp
plt.scatter(x_g/voxel_size, z_g/voxel_size, s=3, c='red', marker='o', label='Goal Point') # gp

plt.legend(loc='best')

plt.xlabel('X index')
plt.ylabel('Z index')
plt.title('2D A* path on 2D OGM')
plt.grid(False)
plt.tight_layout()
plt.show()


# 3d맵 위에 2d path 그리기
path_mask = np.zeros_like(ogm, dtype=np.uint8)
for (i, j) in path_2d_arr:
    path_mask[i, j, np.where(ogm[i, j, :])[0] + k0] = 1
sp_mask = np.zeros_like(ogm, dtype=np.uint8)                                                        # 시작점
gp_mask = np.zeros_like(ogm, dtype=np.uint8)                                                        # 도착점
sp_mask[(start_idx[0], start_idx[1], np.where(ogm[start_idx[0], start_idx[1], :])[0] + k0)] = 1
gp_mask[(goal_idx[0], goal_idx[1], np.where(ogm[goal_idx[0], goal_idx[1], :])[0] + k0)]  = 1

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




# ====================== 3d Linear Interpolation ======================
print(f"Before Linear Interpolation 3d way point 갯수: {len(path_arr_3d)} 개")
plot_3d_data(path_world_3d) # before interpolation 시각화
st_time = time.perf_counter()
interpolated_3d_path_arr = resample_by_arc_length_3d(np.asarray(path_world_3d), spacing = 0.5)
ed_time = time.perf_counter()
print(f'3d Linear Interpolation 소요시간: {ed_time - st_time:.6f}초')
print(f"After Linear Interpolation way point 갯수: {len(interpolated_3d_path_arr)} 개")
plot_3d_data(interpolated_3d_path_arr) # after interpolation 시각화
print(f"After Linear Interpolation way point 갯수: {len(interpolated_3d_path_arr)} 개")



# ====================== 패스 저장(.csv 방식) ====================== 
# np.savetxt('interpolated_path_3d_flat_and_hills_whole.csv', interpolated_3d_path_arr, delimiter=',', header='x,z,y', comments='', fmt='%.2f')  # comments='' 로 '#' 주석 제거 

