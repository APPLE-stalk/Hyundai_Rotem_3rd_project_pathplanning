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

# heading 기준 시야각 +- 90도 벡터
def heading_to_unit_vec(theta_rad: float) -> np.ndarray:
    """2D 단위 벡터 [dx, dz] 반환 (x-z 평면)."""
    return np.array([np.cos(theta_rad), np.sin(theta_rad)])

# world 좌표를 voxel단위 binary map 좌표로 변환
def world_to_index(world_xy: np.ndarray,
                   origin: np.ndarray,
                   voxel_size: float,
                   *, round_mode="floor") -> tuple[int, int]:
    """월드 (x,z) → (row, col).  round_mode = 'floor' | 'nearest'"""
    rel = (world_xy - origin[:2]) / voxel_size
    idx = np.floor(rel).astype(int) if round_mode == "floor" else np.rint(rel).astype(int)
    return int(idx[1]), int(idx[0])

def sample_ray_cells(start_pos: np.ndarray,          # ★ 월드 [x,z]
                     direction: np.ndarray,          # 단위 벡터
                     *,                             # 이후는 키워드 전용
                     origin: np.ndarray,
                     voxel_size: float,
                     map_shape: tuple[int,int],
                     max_range_m: float = 300.0,
                     return_mode: str = "index"):   # "world" | "index"
    """반직선을 따라 찍히는 셀 목록 반환."""
    n_steps = int(max_range_m / voxel_size)
    cur     = start_pos.astype(float).copy()
    out_w, out_i = [], []

    for _ in range(n_steps):
        row, col = world_to_index(cur, origin, voxel_size)
        if not (0 <= row < map_shape[0] and 0 <= col < map_shape[1]):
            break
        out_w.append(cur.copy())        # 월드
        out_i.append((row, col))        # 인덱스
        cur += direction * voxel_size

    return out_w if return_mode=="world" else out_i

# 경유점 생성 위해 내가 적 전차 시야각 기준 전/후방 구분 함수
def is_front(xs: float, zs: float,
            xg: float, zg: float,
            h_g: float) -> bool:
    """
    적 전차( xg, zg, heading h_g ) 기준으로
    내 전차( xs, zs )가 전방( True )인지 후방( False )인지 반환.
    ▷ h_g : math_rad (0 = +x축, CCW가 양의 방향)
    """
    # 1) 적 전차 헤딩 방향 단위벡터 u = [cos, sin]
    u = np.array([np.cos(h_g), np.sin(h_g)])   # (dx, dz)

    # 2) 상대 위치벡터 r = 내 위치 – 적 위치
    r = np.array([xs - xg, zs - zg])           # (Δx, Δz)

    # 3) 내적 부호로 전·후방 결정
    #    dot > 0  → 전방 (각도 < 90°)
    #    dot < 0  → 후방 (각도 > 90°)
    return np.dot(u, r) >= 0                   # 동일선상(90°)도 전방 처리




# 수선의 발 내리는 함수
def foot_to_ray(P: np.ndarray,           # 내 전차 [x,z]
                O: np.ndarray,           # 적 전차 [x,z]
                v: np.ndarray,           # 반직선 방향 벡터
                *,                       # 이후는 **키워드 전용**
                origin: tuple,
                voxel_size: float,
                map_shape: tuple[int, int]
) -> tuple[np.ndarray, float, float]:
    """
    ▸ 수선의 발 F를 구하고, 맵 경계 안으로 클램핑.
    ▸ 반환: (F, d_self, d_enemy)
        · F         : 월드 좌표 (클램핑 결과)
        · d_self    : 내 전차 → F 거리
        · d_enemy   : 적 전차 → F 거리
    """
    # 1) 수선의 발 (무한 반직선 기준)
    u = v / np.linalg.norm(v)                 # 단위벡터
    t = max(np.dot(P - O, u), 0.0)            # O 기준 사영 길이
    F = O + t * u

    # 2) 맵 경계 클램핑 (ray 방향으로만 잘라냄)
    x_min = origin[0]
    z_min = origin[1]
    x_max = origin[0] + (map_shape[1] - 1) * voxel_size
    z_max = origin[1] + (map_shape[0] - 1) * voxel_size

    # 각 축별 허용 최대 t 계산
    if u[0] > 0:
        t_x = (x_max - O[0]) / u[0]
    elif u[0] < 0:
        t_x = (x_min - O[0]) / u[0]
    else:
        t_x = np.inf

    if u[1] > 0:
        t_z = (z_max - O[1]) / u[1]
    elif u[1] < 0:
        t_z = (z_min - O[1]) / u[1]
    else:
        t_z = np.inf

    t_max_rect = min(t_x, t_z)               # 지도 안에서 가능한 최대 t
    t_clamped  = np.clip(t, 0.0, t_max_rect)
    F          = O + t_clamped * u           # 클램핑된 수선의 발

    # 3) 거리 계산
    d_self  = np.linalg.norm(P - F)          # 내 전차 → F
    d_enemy = np.linalg.norm(O - F)          # 적 전차 → F
    return F, d_self, d_enemy

# 생성된 경유점 map 범위 고려하여 ray상에서 clamp
def clamp_point_on_ray(F: np.ndarray,        # 수선의 발(월드)
                       O: np.ndarray,        # 적 전차 위치
                       v: np.ndarray,        # ray 방향벡터
                       *,
                       origin: tuple,
                       voxel_size: float,
                       map_shape: tuple[int, int]) -> np.ndarray:
    """
    F 가 맵 밖이면, O→v 방향 선분을 지도 경계와 만나는 지점으로 잘라
    (= t 를 [0, t_max_rect] 로 제한) 반환합니다.
    """

    # 0) 단위벡터 & F 까지의 t
    u      = v / np.linalg.norm(v)
    t_F    = np.dot(F - O, u)          # O → F 거리(스칼라, m)

    # 1) ray 가 지도 밖으로 나갈 수 있는 최대 t 계산
    #    ─ x 경계
    if u[0] > 0:
        t_x = (origin[0] + (map_shape[1]-1)*voxel_size - O[0]) / u[0]
    elif u[0] < 0:
        t_x = (origin[0]                         - O[0]) / u[0]
    else:
        t_x = np.inf
    #    ─ z 경계
    if u[1] > 0:
        t_z = (origin[1] + (map_shape[0]-1)*voxel_size - O[1]) / u[1]
    elif u[1] < 0:
        t_z = (origin[1]                         - O[1]) / u[1]
    else:
        t_z = np.inf

    t_max_rect = min(t_x, t_z)           # 지도 안에서 허용되는 최대 t (≥0)

    # 2) 클램핑
    t_clamped  = np.clip(t_F, 0.0, t_max_rect)
    return O + t_clamped * u             # 월드 좌표

# # ====================== OGM(.npy) 불러오기 ======================
# st_time = time.perf_counter()
# meta = np.load('./flat_and_hills_whole_OGM(0.5)_with_meta.npz') # OGM, voxel_grid원점, OGM 큐브 사이즈 정보
# ogm  = meta['data']
# origin     = meta['origin']                                     # 필요시 조정할 것
# voxel_size = meta['resolution']
# print('voxel_grid 원점: ', origin)
# print('voxel_size: ', voxel_size)
# ed_time = time.perf_counter()
# print(f'OGM_with_meta파일 읽기 소요시간: {ed_time - st_time:.6f}초')

# # PyVista 시각화에 필요한 Data 생성
# nx, nz, ny = ogm.shape                              # 우, 전방, 높이의 칸수, 칸수*voxel_size가 실제 길이
# grid = pv.ImageData(                                # PyVista에서 [격자(point) + 셀(cell)] 구조를 만드는 클래스
#     dimensions=(nx+1, nz+1, ny+1),                  # 셀 갯수에 따른 포인트 갯수
#     spacing=(voxel_size, voxel_size, voxel_size),   # 포인트 간격
# origin=origin                                       # 격자의 (0, 0, 0)을 월드 좌표계의 (0, 0, 0)으로 지정
# )
# grid.cell_data["occ"] = ogm.ravel(order="F")
# voxels = grid.threshold(0.5, scalars="occ")



# # ====================== load한 OGM 시각화 ======================
# map_plotter = pv.Plotter()
# map_plotter.add_mesh(
#     voxels,
#     color="lightgray",
#     opacity=0.6,
#     show_edges=False
# )
# map_plotter.show_grid(
#     xtitle="X (m)",
#     ytitle="Z (m)",
#     ztitle="Y (m)",
#     show_xaxis=True,
#     show_yaxis=True,
#     show_zaxis=True,
#     show_xlabels=True,
#     show_ylabels=True,
#     show_zlabels=True
# )
# map_plotter.show(title="3D Occupancy Grid (Loaded)")
# map_plotter.close()



# # ====================== 2D Height-Map 만들기 ======================
# H = ogm_to_height_map(ogm, voxel_size=voxel_size, axis=2)       # 3D OGM -> 2D height map
# # 파라미터: 3D OGM, voxel_size, 높이 축

# wall_mask_2d = slope_mask(H, h=voxel_size, theta_deg=31)        # voxel간 기울기 변화 측정하여 기동 불가지역(벽, 장애물)검출 mask
# # 파라미터: 2D height map, voxel_size, 기동 불가 판정 각도


# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(H.T, origin='lower', interpolation='nearest')        # 높이 맵
# plt.title('Height Map(0.5m)')
# plt.xlabel('X index')
# plt.ylabel('Z index')
# plt.colorbar(label='Height (m)')

# # 벽 마스크 시각화
# plt.subplot(1, 2, 2)
# plt.imshow(wall_mask_2d.T, origin='lower', interpolation='nearest')
# plt.title(f'Drivable area(0.5m) ref {31}°')
# plt.xlabel('X index')
# plt.ylabel('Z index')
# plt.colorbar(label='Obstacle Mask(True =Obstacle)')

# plt.tight_layout()
# plt.show()


# # ====================== 전처리(2D slicing) ======================
# st_time = time.perf_counter()

# x_s = 60                                                                # 시작 점 [우, 전방, 높이]
# z_s = 27
# y_s = 3

# x_g = 280                                                               # 목표 점 [우, 전방, 높이]
# z_g = 280
# y_g = 3

# # 시작/목표 월드 좌표 → 그리드 인덱스로 변환
# start_world = np.array([x_s, z_s, y_s])
# goal_world  = np.array([x_g, z_g, y_g])
# start_idx = tuple(((start_world - origin) / voxel_size).astype(int))    # 월드좌표(m) -> 셀 인덱스(칸)
# goal_idx  = tuple(((goal_world  - origin) / voxel_size).astype(int))


# # 맵 지면에서 전차 좌표계까지 높이 차이 
# k0 = 0                                  # [m]
# k0 = int(k0/voxel_size)
# ogm2d = wall_mask_2d
# start2d = (start_idx[0], start_idx[1])  # 우, 전방 (인덱스 단위)
# goal2d  = (goal_idx[0],  goal_idx[1])



# # ====================== 처리 이전 시각화 ======================
# Nx, Nz = ogm2d.shape # 우, 전방
# # extent = [0, Nx * voxel_size,     # x축: 0m ~ Nx*voxel_size m
# #           0, Nz * voxel_size]     # z축: 0m ~ Nz*voxel_size m

# plt.figure(figsize=(10,10))
# plt.subplot(2, 2, 1)
# plt.imshow(ogm2d.T,                 # 전방(y) 축이 위로 오도록 전치
#     origin='lower',                 # 원점이 왼쪽 아래
#     cmap='gray_r')                  # 0→흰색(free), 1→검은색(occupied)
#     # extent=extent)                # 축 단위 m 단위로 변경            
# plt.colorbar(label='Occupancy')
# plt.xlabel('X index')
# plt.ylabel('Z index')
# plt.title('before prcessing')
# plt.grid(False)



# # ====================== 노이즈 없애기 (binary_erosion) ======================
# struct2d = np.ones((3, 3), dtype=bool)
# occ2d_eroded = binary_erosion(ogm2d, structure=struct2d)    # 2D slice 에 침식 적용
# # ====================== 시각화 ======================
# plt.subplot(2, 2, 2)
# plt.imshow(occ2d_eroded.T,                                  # 전방(y) 축이 위로 오도록 전치
#     origin='lower',                                         # 원점이 왼쪽 아래
#     cmap='gray_r')                                          # 0→흰색(free), 1→검은색(occupied)
#     # extent=extent)                                        # m단위로 변환
# plt.colorbar(label='Occupancy')
# plt.xlabel('X index')
# plt.ylabel('Z index')
# plt.title(f'erosion (3x3)')
# plt.grid(False)


# # ====================== 노이즈 없애기 (binary_dilation) ======================
# struct2d = np.ones((3, 3), dtype=bool)
# occ2d_eroded_dilated = binary_dilation(occ2d_eroded, structure=struct2d)    # 2D slice에만 마진 적용
# # ====================== 시각화 ======================
# plt.subplot(2, 2, 3)
# plt.imshow(occ2d_eroded_dilated.T,                                          # 전방(y) 축이 위로 오도록 전치
#     origin='lower',                                                         # 원점이 왼쪽 아래
#     cmap='gray_r')                                                          # 0→흰색(free), 1→검은색(occupied)
#     # extent=extent)                                                        # m단위로 변환
# plt.colorbar(label='Occupancy')
# plt.xlabel('X index')
# plt.ylabel('Z index')
# plt.title(f'erosion_dilation (3x3)')
# plt.grid(False)

# # ====================== 전차 반경 고려 장애물 margin dilation 연산용 2D 커널 ======================
# tank_radius = 3                                                                             # 전차 반경 (m), 전장: 10.8, 차체: 7.5, 전폭: 3.6, 전고: 2.4
# n_margin    = int(np.ceil(tank_radius / voxel_size))
# # struct2d = np.ones((2*n_margin+1, 2*n_margin+1), dtype=bool)                                # 2D용: 정사각 커널
# struct2d = np.ones((8, 8), dtype=bool)
# occ2d_eroded_dilated_inflated = binary_dilation(occ2d_eroded_dilated, structure=struct2d)   # 2D slice에만 마진 적용
# # ====================== 시각화 ======================
# plt.subplot(2, 2, 4)
# plt.imshow(occ2d_eroded_dilated_inflated.T,         # 전방(y) 축이 위로 오도록 전치
#     origin='lower',                                 # 원점이 왼쪽 아래
#     cmap='gray_r')                                  # 0→흰색(free), 1→검은색(occupied)
#     # extent=extent)                                # m단위로 변환
# plt.colorbar(label='Occupancy')
# plt.xlabel('X index')
# plt.ylabel('Z index')
# plt.title(f'erosion_dilation_dilation (8x8)')
# plt.grid(False)
# # plt.tight_layout()
# plt.show()


# ====================== 맵 정보 ======================
origin = (0, 0, 0)
voxel_size = 0.5
binary_map = np.zeros((600, 600), dtype=np.uint8)
map_shape = (600, 600)

# ====================== 상황 정보 ======================
# x_s = 10                                                                    # 시작 점 [우, 전방, 높이, 헤딩]
# z_s = 20
# x_s = 260
# z_s = 260
x_s = 250
z_s = 235
y_s = 0
h_s = 60/180*np.pi

x_g = 250                                                                   # 목표 점 [우, 전방, 높이, 헤딩]
z_g = 280
y_g = 0
h_g = -160/180*np.pi
# h_g = 45/180*np.pi

approach_min = 100 # 적 전차 접근 기준 거리

# ====================== 상황 정보 -> 맵 index 정보로 변환 ======================
start_world = np.array([x_s, z_s, y_s])
goal_world  = np.array([x_g, z_g, y_g])
start_idx = tuple(((start_world - origin) / voxel_size).astype(int))    # 월드좌표(m) -> 셀 인덱스(칸)
goal_idx  = tuple(((goal_world  - origin) / voxel_size).astype(int))

start2d = (start_idx[0], start_idx[1])  # 우, 전방 (인덱스 단위)
goal2d  = (goal_idx[0],  goal_idx[1])


# ====================== 2D OGM 경유점 찾기 ======================
# 내전차 heading 벡터 구하여 맵 index로 변환, 튜플(x, z)를 list로 묶어서 반환
hs_world  = sample_ray_cells(start_world[:2], heading_to_unit_vec(h_s),
                            origin=origin, voxel_size=voxel_size,
                            map_shape=map_shape, max_range_m=10.0,
                            return_mode="world")

# 적 전차 heading
hg_world  = sample_ray_cells(goal_world[:2], heading_to_unit_vec(h_g),
                            origin=origin, voxel_size=voxel_size,
                            map_shape=map_shape, max_range_m=10.0,
                            return_mode="world")

# 적전차 heading +-90도 벡터
v_left  = heading_to_unit_vec(h_g + np.pi / 2)   # world heading 기준 좌측 90° 벡터
v_right = heading_to_unit_vec(h_g - np.pi / 2)   # world heading 우측 90° 벡터

# 적전차 시야각 맵 index에서 구하기
left_world  = sample_ray_cells(goal_world[:2], heading_to_unit_vec(h_g+np.pi/2),# 좌측 시야각 선, 튜플(x, z)를 list로 묶어서 반환
                            origin=origin, voxel_size=voxel_size,
                            map_shape=map_shape, max_range_m=300.0,
                            return_mode="world")   
right_world = sample_ray_cells(goal_world[:2], heading_to_unit_vec(h_g-np.pi/2),
                            origin=origin, voxel_size=voxel_size,
                            map_shape=map_shape, max_range_m=300.0,
                            return_mode="world")   # 우측 시야각 선, 튜플(x, z)를 list로 묶어서 반환

# ─── 2) A* · 시각화용 인덱스 변환 ───
cells_hs    = [world_to_index(p, origin, voxel_size) for p in hs_world]
cells_hg    = [world_to_index(p, origin, voxel_size) for p in hg_world]
left_cells  = [world_to_index(p, origin, voxel_size) for p in left_world]
right_cells = [world_to_index(p, origin, voxel_size) for p in right_world]

# 적전차 좌우 시야각 시각화 위에 변환
left_xs  = [col for row, col in left_cells]                                                         # 시각화 위해 "우" 좌표 따기 
left_zs  = [row for row, col in left_cells]                                                         # 시각화 위해 "전방" 좌표 따기

right_xs = [col for row, col in right_cells]                                                        # 시각화 위해  "우" 좌표 따기 
right_zs = [row for row, col in right_cells]                                                        # 시각화 위해  "전방" 좌표 따기

# 내 전차 전방 헤딩 시각화 위해 변환
hs_xs = [col for row, col in cells_hs]
hs_zs = [row for row, col in cells_hs]

# 적 전차 전방 헤딩 시각화 위해 변환
hg_xs = [col for row, col in cells_hg]
hg_zs = [row for row, col in cells_hg]

# ====================== 경유점 알고리즘 ======================
front_flag = is_front(x_s, z_s, x_g, z_g, h_g)

if front_flag:
    print("내 전차는 적 전차의 전방에 있습니다.")
    print("조중수, 경유점 생성해서 적 측면 공격할 것")
    
    # ① 수선의 발 계산
    F_left,  d_self_l, d_enemy_l  = foot_to_ray(start_world[:2], goal_world[:2], v_left, origin=origin, voxel_size=voxel_size, map_shape=map_shape)

    F_right, d_self_r, d_enemy_r  = foot_to_ray(start_world[:2], goal_world[:2], v_right, origin=origin, voxel_size=voxel_size, map_shape=map_shape)


    # ② ray 방향 클램프
    F_left_cl  = clamp_point_on_ray(
        F_left,  goal_world[:2], v_left,
        origin=origin, voxel_size=voxel_size, map_shape=map_shape)

    F_right_cl = clamp_point_on_ray(
        F_right, goal_world[:2], v_right,
        origin=origin, voxel_size=voxel_size, map_shape=map_shape)

    # ③ 내 전차와 더 가까운 쪽 선택
    if np.linalg.norm(start_world[:2] - F_left_cl) <= np.linalg.norm(start_world[:2] - F_right_cl):
        F_sel       = F_left_cl
        d_enemy_sel = np.linalg.norm(goal_world[:2] - F_left_cl)
        sel_side    = "Left"
    else:
        F_sel       = F_right_cl
        d_enemy_sel = np.linalg.norm(goal_world[:2] - F_right_cl)
        sel_side    = "Right"
    print(f"{sel_side} FOV 수선의 발(클램프) 좌표 : ({F_sel[0]:.3f}, {F_sel[1]:.3f}) m")
    print(f"적 전차  →  수선의 발 거리          : {d_enemy_sel:.3f} m")
    
    # 최초 경유점 시각화
    row_sel, col_sel = world_to_index(F_sel, origin, voxel_size)
    
    # 수선의 발이 최소 접근거리보다 작은 경우
    if(d_enemy_sel < approach_min): 
        u_sel = v_left / np.linalg.norm(v_left) if sel_side == "Left" else v_right / np.linalg.norm(v_right)
        if d_enemy_sel < approach_min:
            # (1) 적 전차 위치 O 기준, 같은 방향 단위벡터 u_sel 로 approach_min 만큼 이동
            F_sel = goal_world[:2] + approach_min * u_sel

            # (2) 지도 밖으로 나갈 수 있으므로 다시 Ray-클램핑
            # F_sel = clamp_point_on_ray(
            #     F_sel, goal_world[:2], u_sel,
            #     origin=origin, voxel_size=voxel_size, map_shape=map_shape
            # )

            # (3) 거리 재계산·로그
            d_enemy_sel = np.linalg.norm(goal_world[:2] - F_sel)
            print(f"[최소 접근거리 적용] 새 수선의 발 : ({F_sel[0]:.3f}, {F_sel[1]:.3f}) m")
            print(f"적 전차 → 수선의 발 거리      : {d_enemy_sel:.3f} m")
            row_sel2, col_sel2 = world_to_index(F_sel, origin, voxel_size)
    
    

else:
    print("내 전차는 적 전차의 후방에 있습니다.")
    print("조종수, 최단거리로 공격할 것!")


# ====================== 2D A* 알고리즘 ======================
st_time = time.perf_counter()
# path2d = astar_2d(ogm2d, start2d, goal2d)                                 # 장애물 마진 없이 A*

path2d = astar_2d(binary_map, start2d, goal2d)           # 장애물 마진 넣어서 A*
ed_time = time.perf_counter()
print(f'A* 소요시간: {ed_time - st_time:.6f}초')
if path2d:
    print('2D 경로 찾음')
    path_2d_arr = np.array(path2d, dtype=int)  # shape = (N, 2)
    
    # k0_col = np.full((path_2d_arr.shape[0], 1), 0, dtype=int)
    # for idx in range(len(path_2d_arr)):
    #     x1, z1 = path_2d_arr[idx]
    #     k0_col[idx] = np.max(np.where(ogm[x1, z1, :])[0]) + k0
    # path_arr_3d = np.hstack((path_2d_arr, k0_col))
    
#     path_world_3d = []                                                      # path_arr_3d를 월드 좌표로 변환
#     for idx in path_arr_3d:
#         i, k, j = idx
#         # 셀의 중앙 좌표 = origin + (index + 0.5)*voxel_size
#         x = origin[0] + (i) * voxel_size
#         z = origin[1] + (k) * voxel_size
#         y = origin[2] + (j + 0.5) * voxel_size
#         path_world_3d.append((x, z, y))
#     print("3차원공간의 3D 경로 인덱스:", path_arr_3d)
else:
    print("2D에서 경로를 찾지 못했습니다.")
    



# ====================== 2D OGM 맵에서 A* 결과 시각화 ======================
plt.figure(figsize=(6, 6))
plt.imshow(binary_map.T,     # 전방(y) 축이 위로 오도록 전치
        origin='lower',                         # 원점이 왼쪽 아래
        cmap='gray_r')                          # 0→흰색(free), 1→검은색(occupied)
plt.colorbar(label='Occupancy')

# 경유점 구하기 위해 적 시야각 표현

plt.plot(hs_xs, hs_zs, linewidth=2, color='cyan',   label='Heading h_s (10 m)')
plt.plot(hg_xs, hg_zs, linewidth=2, color='yellow', label='Heading h_g (10 m)')

plt.scatter(left_xs,  left_zs,  s=1, c='lime',    marker='.', label='Left FOV boundary')
plt.scatter(right_xs, right_zs, s=1, c='magenta', marker='.', label='Right FOV boundary')

# 맵 내부의 최초 경유점(수선의 발)
plt.scatter(col_sel, row_sel, s=40, c='black', marker='X',
            label=f'{sel_side} foot (clamped)')

# 최소 접근 거래 내부로 찍힌 경우 최소 접근거리로 경유점 재설정 시각화
if "col_sel2" in locals() and "row_sel2" in locals() and col_sel2 is not None and row_sel2 is not None:
    plt.scatter(col_sel2, row_sel2, s=40, c='yellow', marker='X',
                label=f'{sel_side} foot (clamped)')

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
# path_mask = np.zeros_like(ogm, dtype=np.uint8)
# for (i, j) in path_2d_arr:
#     path_mask[i, j, np.where(ogm[i, j, :])[0] + k0] = 1
# sp_mask = np.zeros_like(ogm, dtype=np.uint8)                                                        # 시작점
# gp_mask = np.zeros_like(ogm, dtype=np.uint8)                                                        # 도착점
# sp_mask[(start_idx[0], start_idx[1], np.where(ogm[start_idx[0], start_idx[1], :])[0] + k0)] = 1
# gp_mask[(goal_idx[0], goal_idx[1], np.where(ogm[goal_idx[0], goal_idx[1], :])[0] + k0)]  = 1

# grid.cell_data["path"] = path_mask.ravel(order="F")

# grid.cell_data["sp"]   = sp_mask.ravel(order="F")
# grid.cell_data["gp"]   = gp_mask.ravel(order="F")

# path_voxels    = grid.threshold(0.5, scalars="path")
# path_voxels_sp = grid.threshold(0.5, scalars="sp")
# path_voxels_gp = grid.threshold(0.5, scalars="gp")

# plotter = pv.Plotter()
# plotter.add_mesh(voxels,        color="lightgray", opacity=0.6, show_edges=False)
# plotter.add_mesh(path_voxels,   color="yellow",     opacity=1.0,   show_edges=False)
# plotter.add_mesh(path_voxels_sp, color="blue",     opacity=1.0,   show_edges=False)
# plotter.add_mesh(path_voxels_gp, color="red",      opacity=1.0,   show_edges=False)
# plotter.show_grid(
#     xtitle="X (m)",
#     ytitle="Z (m)",
#     ztitle="Y (m)",
#     show_xaxis=True,
#     show_yaxis=True,
#     show_zaxis=True,
#     show_xlabels=True,
#     show_ylabels=True,
#     show_zlabels=True
# )
# plotter.show(title="3D Occupancy + 2D Path (Start/Goal)")
# plotter.close()




# ====================== 3d Linear Interpolation ======================
# print(f"Before Linear Interpolation 3d way point 갯수: {len(path_arr_3d)} 개")
# plot_3d_data(path_world_3d) # before interpolation 시각화
# st_time = time.perf_counter()
# interpolated_3d_path_arr = resample_by_arc_length_3d(np.asarray(path_world_3d), spacing = 0.5)
# ed_time = time.perf_counter()
# print(f'3d Linear Interpolation 소요시간: {ed_time - st_time:.6f}초')
# print(f"After Linear Interpolation way point 갯수: {len(interpolated_3d_path_arr)} 개")
# plot_3d_data(interpolated_3d_path_arr) # after interpolation 시각화
# print(f"After Linear Interpolation way point 갯수: {len(interpolated_3d_path_arr)} 개")



# ====================== 패스 저장(.csv 방식) ====================== 
# np.savetxt('interpolated_path_3d_flat_and_hills_whole.csv', interpolated_3d_path_arr, delimiter=',', header='x,z,y', comments='', fmt='%.2f')  # comments='' 로 '#' 주석 제거 

