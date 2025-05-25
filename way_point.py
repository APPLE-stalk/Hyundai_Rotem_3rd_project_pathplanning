import numpy as np
import heapq
import pyvista as pv
from scipy.ndimage import binary_dilation   # 장애물 마진
from scipy.ndimage import binary_erosion    # 노이즈 erosion
import time
import matplotlib.pyplot as plt
from scipy import ndimage # 소벨마스크
import matplotlib.patches as mpatches # 적전차 최소 접근거리 시각화
import math

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
def world_to_index(world_xy: np.ndarray, origin: np.ndarray, voxel_size: float, *, round_mode="floor") -> tuple[int, int]:
    """월드 (x,z) → (row, col).  round_mode = 'floor' | 'nearest'"""
    world_xy = np.asarray(world_xy, dtype=float)       # ★ 추가
    origin   = np.asarray(origin[:2], dtype=float)     # ★ 추가
    rel = (world_xy - origin[:2]) / voxel_size
    idx = np.floor(rel).astype(int) if round_mode == "floor" else np.rint(rel).astype(int)
    return int(idx[0]), int(idx[1]) # 우 전방

def index_to_world(idx_rc, origin, voxel_size):
    """
    2-D 인덱스 (row, col) → 월드 좌표 (x, z) 변환
      · row : “우( x )” 방향 인덱스
      · col : “전방( z )”  방향 인덱스
    """
    row, col = idx_rc            # (wp_idx[0], wp_idx[1])
    x = origin[0] + row * voxel_size
    z = origin[1] + col * voxel_size
    return np.array([x, z], dtype=float)

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

# 갈 수 없는 사각형(world 좌표)를 1로 채우기
def fill_rect_obstacle(binary_map,
                       top_left_xy, bottom_right_xy,   # 월드 좌표 (x,z)
                       origin, voxel_size):
    # (1) 월드 → 셀 인덱스
    row_tl, col_tl = world_to_index(top_left_xy,     origin, voxel_size)
    row_br, col_br = world_to_index(bottom_right_xy, origin, voxel_size)

    # (2) 행/열 정렬 (좌상단·우하단이 바뀌어 들어와도 대응)
    r0, r1 = sorted((row_tl, row_br))
    c0, c1 = sorted((col_tl, col_br))

    # (3) 맵 경계 클램핑
    r0 = max(r0, 0);              c0 = max(c0, 0)
    r1 = min(r1, binary_map.shape[0]-1)
    c1 = min(c1, binary_map.shape[1]-1)

    # (4) 채우기
    binary_map[r0:r1+1, c0:c1+1] = 1
    
    # return binary_map
    
# def find_free_on_ray(O_xy, u, start_t, *, search_outward,
#                     binary_map, origin, voxel_size, map_shape,
#                     max_extra=300.0, step=0.5):
#     """
#     O + t·u 선에서 0.5 m 간격으로 이동하며
#     binary_map==0 인 첫 셀을 찾아 (월드좌표, 인덱스) 반환.
#     search_outward=True  → t 증가 방향,  False → t 감소 방향.
#     """

#     # ── 0. 단위벡터 보정 ───────────────────────────────
#     u = u / np.linalg.norm(u)

#     # ── 1. 지도 경계까지 허용되는 t_max 계산 ──────────
#     if u[0] > 0:
#         t_x = (origin[0] + (map_shape[1]-1)*voxel_size - O_xy[0]) / u[0]
#     elif u[0] < 0:
#         t_x = (origin[0] - O_xy[0]) / u[0]
#     else:                                    # u.x == 0
#         t_x = np.inf

#     if u[1] > 0:
#         t_z = (origin[1] + (map_shape[0]-1)*voxel_size - O_xy[1]) / u[1]
#     elif u[1] < 0:
#         t_z = (origin[1] - O_xy[1]) / u[1]
#     else:                                    # u.z == 0
#         t_z = np.inf

#     t_max_map = max(0.0, min(t_x, t_z))      # 음수 방지

#     # ── 2. t 증분 검색 ───────────────────────────────
#     dir_sign = 1 if search_outward else -1
#     t = start_t
#     n_steps = int(max_extra / step)

#     for _ in range(n_steps):
#         t += dir_sign * step
#         if t < 0 or t > t_max_map:           # 지도 밖
#             break

#         P = O_xy + t*u                       # 월드 좌표
#         row, col = world_to_index(P, origin, voxel_size)
#         if not (0 <= row < map_shape[0] and 0 <= col < map_shape[1]):
#             break                            # 안전장치

#         if binary_map[row, col] == 0:        # ★ 첫 자유 셀
#             return P, (row, col)

#     return None, None
def find_free_on_ray(
    O_xy, u, start_t, *,                  # O : 적 전차 (x,z)
    binary_map, origin, voxel_size, map_shape,
    max_extra=300.0, step=0.5,
):
    """
    1)  O + t·u  선에서 0․5 m 간격으로 이동하며
    2)  먼저 적에게 **가까운 쪽(-t)** 으로 탐색,
        못 찾으면 **바깥(+t)** 으로 탐색.
    3)  binary_map==0 인 첫 셀을 (월드좌표, 인덱스) 로 반환.
        찾지 못하면 (None, None).
    """

    # ── 0. 단위벡터 정규화 ─────────────────────────────
    u = u / np.linalg.norm(u)

    # ── 1. 지도 경계까지 허용되는 t_max 계산 ──────────
    if u[0] > 0:
        t_x = (origin[0] + (map_shape[1]-1)*voxel_size - O_xy[0]) / u[0]
    elif u[0] < 0:
        t_x = (origin[0] - O_xy[0]) / u[0]
    else:
        t_x = np.inf

    if u[1] > 0:
        t_z = (origin[1] + (map_shape[0]-1)*voxel_size - O_xy[1]) / u[1]
    elif u[1] < 0:
        t_z = (origin[1] - O_xy[1]) / u[1]
    else:
        t_z = np.inf

    t_max = max(0.0, min(t_x, t_z))        # 0 ≤ t ≤ t_max

    # ── 2-A. “적에게 가까운 방향” 우선 탐색 (-t) ──────
    t = start_t
    n_steps = int(max_extra / step)
    for _ in range(n_steps):
        # 현재 t 검사
        P = O_xy + t * u
        row, col = world_to_index(P, origin, voxel_size)

        if (0 <= row < map_shape[0]) and (0 <= col < map_shape[1]):
            if binary_map[row, col] == 0:
                return P, (row, col)
        else:
            break                          # 맵을 벗어남

        t -= step                          # 적에게 더 접근
        if t < 0:                          # O 지점 지나면 중단
            break

    # ── 2-B. 못 찾았으면 바깥 방향(+t) 탐색 ───────────
    t = start_t + step                     # 이미 검사한 지점은 건너뜀
    for _ in range(n_steps):
        if t > t_max:
            break

        P = O_xy + t * u
        row, col = world_to_index(P, origin, voxel_size)

        if (0 <= row < map_shape[0]) and (0 <= col < map_shape[1]):
            if binary_map[row, col] == 0:
                return P, (row, col)
        else:
            break

        t += step                          # 바깥쪽으로 이동

    # ── 3. 실패 ───────────────────────────────────────
    return None, None

def waypoint_on_approach_ring(O_xy,          # 적 전차 (x,z)
                              heading,       # 적 전차 헤딩(rad)
                              radius,        # approach_min
                              binary_map, origin, voxel_size, map_shape,
                              prefer_xy):    # 내 전차 (x,z)
    """
    ▸ 적 전방 ± 150° 방향으로 원(approach_min) 위 두 점을 계산
    ▸ 맵 밖 / 장애물 셀은 제외
    ▸ 남은 점 가운데 내 전차에 더 가까운 것을 반환
      (없으면 None, None)
    """
    cand_pts = []
    for sign in (+1, -1):                    # +150°, -150°
        theta = heading + sign * math.radians(150)
        pt    = O_xy + radius * np.array([math.cos(theta), math.sin(theta)])

        # 맵 범위‧장애물 체크
        row, col = world_to_index(pt, origin, voxel_size)   # (x,z) → (row,col)
        if (0 <= row < map_shape[0] and 0 <= col < map_shape[1]
                and binary_map[row, col] == 0):
            cand_pts.append((pt, (row, col)))
    print('링 점점점점', cand_pts)
    if not cand_pts:
        return None, None          # 둘 다 막혔거나 지도 밖
    if len(cand_pts) == 1:
        return cand_pts[0]         # 하나만 통과

    # 두 점 다 가능 → 내 전차와 거리 비교
    dists = [np.linalg.norm(prefer_xy - p[0]) for p,_ in cand_pts]
    print('경유점 호호호', prefer_xy)
    print('거ㄹ리리ㅣ릴', dists)
    best  = cand_pts[int(np.argmin(dists))]
    return best                    # (월드좌표, (row,col))



# # ====================== OGM(.npy) 불러오기 ======================
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



# # ====================== 2D Height-Map 만들기 ======================
H = ogm_to_height_map(ogm, voxel_size=voxel_size, axis=2)       # 3D OGM -> 2D height map
# # 파라미터: 3D OGM, voxel_size, 높이 축

wall_mask_2d = slope_mask(H, h=voxel_size, theta_deg=31)        # voxel간 기울기 변화 측정하여 기동 불가지역(벽, 장애물)검출 mask
# # 파라미터: 2D height map, voxel_size, 기동 불가 판정 각도


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

# x_s = 60                                                                # 시작 점 [우, 전방, 높이]
# z_s = 27
x_s = 125
z_s = 50
y_s = 3

x_g = 120                                                               # 목표 점 [우, 전방, 높이]
z_g = 250
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

########### 덕규 코드
from scipy.ndimage import binary_closing, binary_fill_holes
# 1. binary_closing으로 작은 구멍 및 부분적 구멍 채우기
structure = np.ones((7, 7), dtype=bool)  # 7x7 구조 요소, 큰 구멍을 위해 조정 가능
occ2d_closed = binary_closing(occ2d_eroded_dilated_inflated, structure=structure).astype(bool)

# 2. binary_fill_holes으로 완전히 둘러싸인 큰 구멍 채우기
occ2d_filled = binary_fill_holes(occ2d_closed).astype(bool)
# print(f"Filled shape: {occ2d_filled.shape}")  # (300, 300)

occ2d_eroded_dilated_inflated = occ2d_filled
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


# ====================== 맵 정보 ======================
# origin = (0, 0, 0)
# voxel_size = 0.5
# binary_map = np.zeros((600, 600), dtype=np.uint8)
# map_shape = (600, 600)

# 갈 수 없는 지역 지정
# fill_rect_obstacle(binary_map,
#                 top_left_xy     = (50.0, 135.0),    # 좌상단  (x,z)
#                 bottom_right_xy = (250.0, 50.0),  # 우하단  (x,z)
#                 origin=origin, voxel_size=voxel_size)

# ====================== 상황 정보 ======================
# x_s = 10                                                                    # 시작 점 [우, 전방, 높이, 헤딩]
# z_s = 20

# x_s = 260
# z_s = 260
# x_s = 250
# z_s = 235
# y_s = 0
h_s = 10/180*np.pi

# x_g = 200                                                                   # 목표 점 [우, 전방, 높이, 헤딩]
# z_g = 150
# y_g = 0
h_g = -160/180*np.pi
# h_g = 45/180*np.pi

approach_min = 50 # 적 전차 접근 기준 거리
map_shape = (nx, nz)
# ====================== 상황 정보 -> 맵 index 정보로 변환 ======================
# start_world = np.array([x_s, z_s, y_s])
# goal_world  = np.array([x_g, z_g, y_g])
# start_idx = tuple(((start_world - origin) / voxel_size).astype(int))    # 월드좌표(m) -> 셀 인덱스(칸)
# goal_idx  = tuple(((goal_world  - origin) / voxel_size).astype(int))

# start2d = (start_idx[0], start_idx[1])  # 우, 전방 (인덱스 단위)
# goal2d  = (goal_idx[0],  goal_idx[1])


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
    print('이거 우전방 맞냐', F_left)
    F_right, d_self_r, d_enemy_r  = foot_to_ray(start_world[:2], goal_world[:2], v_right, origin=origin, voxel_size=voxel_size, map_shape=map_shape)


    # ② ray 방향 클램프
    F_left_cl  = clamp_point_on_ray(
        F_left,  goal_world[:2], v_left,
        origin=origin, voxel_size=voxel_size, map_shape=map_shape)
    print('이것도 우전방 맞냐', F_left_cl)
    F_right_cl = clamp_point_on_ray(
        F_right, goal_world[:2], v_right,
        origin=origin, voxel_size=voxel_size, map_shape=map_shape)

    # ③ 내 전차와 더 가까운 쪽 선택
    if np.linalg.norm(start_world[:2] - F_left_cl) <= np.linalg.norm(start_world[:2] - F_right_cl):
        F_sel       = F_left_cl
        print('이이이이거 우전방 맞냐', F_sel)
        d_enemy_sel = np.linalg.norm(goal_world[:2] - F_left_cl)
        sel_side    = "Left"
    else:
        F_sel       = F_right_cl
        d_enemy_sel = np.linalg.norm(goal_world[:2] - F_right_cl)
        sel_side    = "Right"
    print(f"{sel_side} FOV 수선의 발(클램프) 좌표 : ({F_sel[0]:.3f}, {F_sel[1]:.3f}) m")
    print(f"적 전차  →  수선의 발 거리          : {d_enemy_sel:.3f} m")
    
    # 최초 경유점 시각화용
    r_sel, f_sel = world_to_index(F_sel, origin, voxel_size) # 우 전방
    print('uuuuuuuuuuuuuuu', r_sel, f_sel)
    # 수선의 발이 최소 접근거리보다 작은 경우
    if(d_enemy_sel < approach_min): 
        u_sel = v_left / np.linalg.norm(v_left) if sel_side == "Left" else v_right / np.linalg.norm(v_right)
        if d_enemy_sel < approach_min:
            # (1) 적 전차 위치 O 기준, 같은 방향 단위벡터 u_sel 로 approach_min 만큼 이동
            F_sel1 = goal_world[:2] + approach_min * u_sel

            # (2) 지도 밖으로 나갈 수 있으므로 다시 Ray-클램핑
            F_sel1 = clamp_point_on_ray(
                F_sel1, goal_world[:2], u_sel,
                origin=origin, voxel_size=voxel_size, map_shape=map_shape
            )
            row_sel1, col_sel1 = world_to_index(F_sel1, origin, voxel_size)
            if occ2d_eroded_dilated_inflated[row_sel1, col_sel1] == 1:            # ← 막힌 셀
                print("경유점이 장애물!  FOV 선을 따라 재검색")

                # 1) 먼저 선택된 시야각(u_sel) 방향으로 맵 안쪽 끝까지 OUTWARD 탐색
                u_sel   = v_left/np.linalg.norm(v_left) if sel_side=="Left" else v_right/np.linalg.norm(v_right)
                t_start = np.dot(F_sel - goal_world[:2], u_sel)
                P_new, idx_new = find_free_on_ray(goal_world[:2], u_sel, t_start,
                                                # search_outward=True,
                                                binary_map=occ2d_eroded_dilated_inflated,
                                                origin=origin, voxel_size=voxel_size,
                                                map_shape=map_shape)
                print('우우우우 전방', P_new)
                # 2) 실패하면 반대 시야각(u_opp)으로, 맵 끝→안쪽(INWARD) 탐색
                if P_new is None:
                    u_opp = v_right/np.linalg.norm(v_right) if sel_side=="Left" else v_left/np.linalg.norm(v_left)
                    _, _, t_max = foot_to_ray(goal_world[:2], goal_world[:2], u_opp,
                                            origin=origin, voxel_size=voxel_size,
                                            map_shape=map_shape)
                    P_new, idx_new = find_free_on_ray(goal_world[:2], u_opp, t_max,
                                                    search_outward=False,
                                                    binary_map=occ2d_eroded_dilated_inflated,
                                                    origin=origin, voxel_size=voxel_size,
                                                    map_shape=map_shape)
                    if P_new is not None:
                        sel_side = "Right" if sel_side=="Left" else "Left"   # 방향 전환

                # 3) 성공 시 경유점 갱신
                if P_new is not None:
                    F_sel2         = P_new
                    row_sel2, col_sel2 = idx_new
                    d_enemy_sel2   = np.linalg.norm(goal_world[:2] - F_sel2)
                    print(f"[재검색 성공] 새 경유점 ({sel_side}) : "
                        f"{F_sel2[0]:.2f}, {F_sel2[1]:.2f} m  (적 거리 {d_enemy_sel2:.2f} m)")
                else:
                    print("두 방향 모두에서 자유 셀을 찾지 못했습니다 → 경유점 사용 포기")
                    
            if "row_sel2" in locals() and row_sel2 is not None and "col_sel2" in locals() and col_sel2 is not None:
                pass
            else:
                # (3) 거리 재계산·로그 # 근접문제 해결 후 True문제 없는 경우
                d_enemy_sel1 = np.linalg.norm(goal_world[:2] - F_sel1)
                print(f"[최소 접근거리 적용] 새 수선의 발 : ({F_sel1[0]:.3f}, {F_sel1[1]:.3f}) m")
                print(f"적 전차 → 수선의 발 거리      : {d_enemy_sel1:.3f} m")
                row_sel1, col_sel1 = world_to_index(F_sel1, origin, voxel_size)
            
            # if binary_map[row_sel1, col_sel1] == 1:            # ← 막힌 셀
            #     print("경유점이 장애물!  FOV 선을 따라 재검색")

            #     # 1) 먼저 선택된 시야각(u_sel) 방향으로 맵 안쪽 끝까지 OUTWARD 탐색
            #     u_sel   = v_left/np.linalg.norm(v_left) if sel_side=="Left" else v_right/np.linalg.norm(v_right)
            #     t_start = np.dot(F_sel - goal_world[:2], u_sel)
            #     P_new, idx_new = find_free_on_ray(goal_world[:2], u_sel, t_start,
            #                                     search_outward=True,
            #                                     binary_map=binary_map,
            #                                     origin=origin, voxel_size=voxel_size,
            #                                     map_shape=map_shape)

            #     # 2) 실패하면 반대 시야각(u_opp)으로, 맵 끝→안쪽(INWARD) 탐색
            #     if P_new is None:
            #         u_opp = v_right/np.linalg.norm(v_right) if sel_side=="Left" else v_left/np.linalg.norm(v_left)
            #         _, _, t_max = foot_to_ray(goal_world[:2], goal_world[:2], u_opp,
            #                                 origin=origin, voxel_size=voxel_size,
            #                                 map_shape=map_shape)
            #         P_new, idx_new = find_free_on_ray(goal_world[:2], u_opp, t_max,
            #                                         search_outward=False,
            #                                         binary_map=binary_map,
            #                                         origin=origin, voxel_size=voxel_size,
            #                                         map_shape=map_shape)
            #         if P_new is not None:
            #             sel_side = "Right" if sel_side=="Left" else "Left"   # 방향 전환
            #     # 3) 성공 시 경유점 갱신
            #     if P_new is not None:
            #         F_sel         = P_new
            #         row_sel3, col_sel3 = idx_new
            #         d_enemy_sel   = np.linalg.norm(goal_world[:2] - F_sel)
            #         print(f"[재검색 성공] 새 경유점 ({sel_side}) : "
            #             f"{F_sel[0]:.2f}, {F_sel[1]:.2f} m  (적 거리 {d_enemy_sel:.2f} m)")
            #     else:
            #         print("두 방향 모두에서 자유 셀을 찾지 못했습니다 → 경유점 사용 포기")
    
    # 수선의 발이 최소 접근거리 밖에 있는 경우
    else:
        
        if occ2d_eroded_dilated_inflated[r_sel, f_sel] == 1:            # ← 막힌 셀 우 전방
            print('hhhhhhhhhhhh', r_sel, f_sel) # 우 전방
            print("경유점이 장애물!  FOV 선을 따라 재검색")

            # 1) 먼저 선택된 시야각(u_sel) 방향으로 맵 안쪽 끝까지 OUTWARD 탐색
            u_sel   = v_left/np.linalg.norm(v_left) if sel_side=="Left" else v_right/np.linalg.norm(v_right)
            t_start = np.dot(F_sel - goal_world[:2], u_sel)
            P_new, idx_new = find_free_on_ray(goal_world[:2], u_sel, t_start,
                                            # search_outward=True,
                                            binary_map=occ2d_eroded_dilated_inflated,
                                            origin=origin, voxel_size=voxel_size,
                                            map_shape=map_shape)
            print('fnfnfnfn', P_new)
            # 2) 실패하면 반대 시야각(u_opp)으로, 맵 끝→안쪽(INWARD) 탐색
            if P_new is None:
                print('실패함???')
                u_opp = v_right/np.linalg.norm(v_right) if sel_side=="Left" else v_left/np.linalg.norm(v_left)
                _, _, t_max = foot_to_ray(goal_world[:2], goal_world[:2], u_opp,
                                        origin=origin, voxel_size=voxel_size,
                                        map_shape=map_shape)
                P_new, idx_new = find_free_on_ray(goal_world[:2], u_opp, t_max,
                                                search_outward=False,
                                                binary_map=occ2d_eroded_dilated_inflated,
                                                origin=origin, voxel_size=voxel_size,
                                                map_shape=map_shape)
                if P_new is not None:
                    sel_side = "Right" if sel_side=="Left" else "Left"   # 방향 전환
            # 3) 성공 시 경유점 갱신
            if P_new is not None:
                F_sel3         = P_new
                r_sel3, f_sel3   = idx_new  # idx_new 우, 전방
                print('fffffff', r_sel3, f_sel3) # 우 전방
                d_enemy_sel3   = np.linalg.norm(goal_world[:2] - F_sel3)
                print(f"[재검색 성공] 새 경유점 ({sel_side}) : "
                    f"{F_sel3[0]:.2f}, {F_sel3[1]:.2f} m  (적 거리 {d_enemy_sel3:.2f} m)") # 우 전방
            else:
                print("두 방향 모두에서 자유 셀을 찾지 못했습니다 → 경유점 사용 포기")

else:
    print("내 전차는 적 전차의 후방에 있습니다.")
    print("조종수, 최단거리로 공격할 것!")
    
# ====================== 만들어진 경유점 정리 ======================
# ── 경유점 인덱스(wp_idx) 확정 ────────────────────────────
wp_idx = None
if   "r_sel3" in locals() and r_sel3 is not None:   # 재검색 성공
    wp_idx = (r_sel3, f_sel3) # 우 전방
elif "row_sel2" in locals() and row_sel2 is not None:   # 최소거리 적용
    wp_idx = (row_sel2, col_sel2)
elif "row_sel1" in locals() and row_sel1 is not None:   # 접근만 적용
    wp_idx = (row_sel1, col_sel1)
elif "r_sel"  in locals() and r_sel  is not None:   # 최초 수선의 발
    wp_idx = (r_sel,  f_sel)

# ====================== 2D A* 알고리즘 2회 실시 ======================
if wp_idx is not None:        # 경유점이 있을 때
    path_s2w = astar_2d(occ2d_eroded_dilated_inflated, start2d, (wp_idx[0], wp_idx[1])) # 우 전방
    print('ssssssss', start2d)
    print('RRRRR FFFFF', wp_idx[0], wp_idx[1])
    print('11111111', len(path_s2w))
    
    wp_world = index_to_world((wp_idx[0], wp_idx[1]), origin, voxel_size)
    print('경유점 좌표', (wp_world[0], wp_world[1]))
    wp_ring_w, wp_ring_idx = waypoint_on_approach_ring(
        O_xy       = goal_world[:2],
        heading    = h_g,
        radius     = approach_min,          # 50 m
        binary_map = occ2d_eroded_dilated_inflated,
        origin     = origin,
        voxel_size = voxel_size,
        map_shape  = map_shape,
        prefer_xy  = (wp_idx[0], wp_idx[1]))       # 경유점 우, 전방
    
    print('왜 이러냐', wp_ring_idx[0], wp_ring_idx[1])
    path_w2r = astar_2d(occ2d_eroded_dilated_inflated, (wp_idx[0], wp_idx[1]),  (wp_ring_idx[0], wp_ring_idx[1]))
    
    path_r2g = astar_2d(occ2d_eroded_dilated_inflated, (wp_ring_idx[0], wp_ring_idx[1]),  goal2d)
    print('22222222', len(path_r2g))
    # ♦ 두 경로를 합칠 때 중복되는 웨이포인트 하나 제거
    # path_full = path_s2w[:-1] + path_w2g
    # print('333333', len(path_full))
    # print(path_full)
else:                          # 경유점이 없으면 한 번만
    print('경유점 없냐ㅑㅑ')
    path_s2w = []
    path_w2g = []
    path_full = astar_2d(occ2d_eroded_dilated_inflated, start2d, goal2d)
    

# ====================== 2D A* 알고리즘 ======================
# st_time = time.perf_counter()
# # path2d = astar_2d(ogm2d, start2d, goal2d)                                 # 장애물 마진 없이 A*

# path2d = astar_2d(binary_map, start2d, goal2d)           # 장애물 마진 넣어서 A*
# ed_time = time.perf_counter()
# print(f'A* 소요시간: {ed_time - st_time:.6f}초')
# if path2d:
#     print('2D 경로 찾음')
#     path_2d_arr = np.array(path2d, dtype=int)  # shape = (N, 2)
    
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
# else:
#     print("2D에서 경로를 찾지 못했습니다.")
    



# ====================== 2D OGM 맵에서 A* 결과 시각화 ======================
plt.figure(figsize=(6, 6))
plt.imshow(occ2d_eroded_dilated_inflated.T,     # 전방(y) 축이 위로 오도록 전치
        origin='lower',                         # 원점이 왼쪽 아래
        cmap='gray_r')                          # 0→흰색(free), 1→검은색(occupied)
plt.colorbar(label='Occupancy')

# 경유점 구하기 위해 적 시야각 표현

plt.plot(hs_zs, hs_xs, linewidth=2, color='cyan',   label='Heading h_s (10 m)')
plt.plot(hg_zs, hg_xs,  linewidth=2, color='yellow', label='Heading h_g (10 m)')

plt.scatter(left_zs, left_xs,   s=1, c='lime',    marker='.', label='Left FOV boundary')
plt.scatter(right_zs, right_xs, s=1, c='magenta', marker='.', label='Right FOV boundary')

# 적전차 최소 접근거리 시각화
row_c, col_c = world_to_index(goal_world[:2], origin, voxel_size)
circle = plt.Circle((row_c, col_c ),               # 뒤집어서 시각화
                    radius=approach_min/voxel_size,
                    fill=False,                   # 내부 비우기
                    edgecolor='red',
                    linewidth=1.5,
                    linestyle='-',
                    label=f'{approach_min:.0f} m radius')
plt.gca().add_patch(circle)      # 또는 ax.add_patch(circle)

# 맵 내부의 최초 경유점(수선의 발)
if "col_sel" in locals() and "row_sel" in locals() and f_sel is not None and r_sel is not None:
    plt.scatter(f_sel, r_sel, s=40, c='gray', marker='X',
                label=f'{sel_side} foot (clamped)')

# 최소 접근 거래 내부로 찍힌 경우 최소 접근거리로 경유점 재설정 시각화
if "col_sel2" in locals() and "row_sel2" in locals() and col_sel2 is not None and row_sel2 is not None:
    plt.scatter(col_sel2, row_sel2, s=40, c='yellow', marker='X',
                label=f'{sel_side} foot (clamped)')
    
# 찾은 경유점이 갈 수 없을 때 시야각선 따라서 재탐색 시각화
if "col_sel3" in locals() and "row_sel3" in locals() and r_sel3 is not None and f_sel3 is not None:
    plt.scatter(f_sel3, r_sel3, s=40, c='pink', marker='X', # 뒤집어서 출력
                label=f'{sel_side} foot (clamped)')

# path2d 궤적 추가
# xs, zs = zip(*path2d)  
# plt.scatter(xs, zs, s=3, c='orange', marker='o',label='path2d') # path
# plt.scatter(x_s/voxel_size, z_s/voxel_size, s=3, c='blue', marker='o', label='Start Point') # sp
# plt.scatter(x_g/voxel_size, z_g/voxel_size, s=3, c='red', marker='o', label='Goal Point') # gp

# 경유점 포함한 a* path 궤적 추가
if path_s2w:                                  # 시작 → 웨이포인트
    xs1, zs1 = zip(*path_s2w) # 우 전방
    plt.plot(xs1, zs1, color='orange', lw=2, label='Start → WP') 
    
if path_s2w:                                  # 웨이포인트 → 링포인트
    xs1, zs1 = zip(*path_w2r) # 우 전방
    plt.plot(xs1, zs1, color='green', lw=2, label='Start → WP') 
    
if path_r2g:                                  # 링포인트 → 목표
    xs2, zs2 = zip(*path_r2g) # 우 전방
    plt.plot(xs2, zs2, color='blue',   lw=2, label='WP → Goal') 

# if not path_s2w and not path_w2g:             # 경유점 없이 한 번에
#     xs, zs = zip(*path_full)  # 우 전방
#     plt.plot(xs, zs, color='orange', lw=2, label='Start → Goal') 
    
# 웨이포인트 마커
if wp_idx is not None:
    plt.scatter(wp_idx[0], wp_idx[1], s=60, c='red',
                marker='X', zorder=5, label='Waypoint')

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

