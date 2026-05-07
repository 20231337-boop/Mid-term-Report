# 자율이동체시스템 중간고사 대체 과제

성신여자대학교 자율이동체시스템 
학번: 20231337  
이름: 김유나

## 개요
KITTI Odometry **Sequence 09** 데이터를 이용해 다음 5문제를 해결한다.

1. Projection Matrix 해석
2. Projection Matrix를 이용한 3D → 2D 투영
3. Pose 데이터를 이용한 차량 궤적 시각화
4. Projection Matrix를 활용한 차선 해석 (Bayesian 분류 결과 결합)
5. 실패 구간 분석

## 파일 구조
- `중간과제.ipynb` — 메인 노트북 (모든 문제 풀이 + 결과)
- `outputs/` — 시각화 결과 이미지

## 데이터셋
KITTI Odometry Dataset, Sequence 09  
- `image_0/` (좌측 흑백, 1591 frames)
- `calib.txt`, `poses/09.txt`


import os
import numpy as np

DATA_ROOT = r"C:\Users\user\Downloads\DDD\KITTI_for_colab"
SEQ = "09"

calib_path = os.path.join(DATA_ROOT, "sequences", SEQ, "calib.txt")

def load_calib(path):
    """KITTI calib.txt 파싱 → P0~P3을 (3,4) 행렬로 반환"""
    calib = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, vals = line.split(":", 1)
            arr = np.array([float(v) for v in vals.split()])
            if key in ("P0", "P1", "P2", "P3", "Tr"):
                calib[key] = arr.reshape(3, 4)
    return calib

def decompose_projection(P):
    """KITTI rectified P → K, R, t 분해"""
    K = P[:, :3].copy()
    t = np.linalg.inv(K) @ P[:, 3]
    R = np.eye(3)
    return K, R, t

# 1) 4개 Projection Matrix 로드 및 출력
calib = load_calib(calib_path)
for key in ("P0", "P1", "P2", "P3"):
    print(f"=== {key} ===")
    print(calib[key])
    print()

# 2) P0 분해 (기준 카메라)
P0 = calib["P0"]
K, R, t = decompose_projection(P0)
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

print("=== P0 Intrinsic K ===")
print(K)
print(f"  f_x = {fx:.4f} px,  f_y = {fy:.4f} px")
print(f"  c_x = {cx:.4f} px,  c_y = {cy:.4f} px")
print(f"\n=== P0 Extrinsic ===")
print(f"R =\n{R}")
print(f"t = {t}")

# 3) 4개 카메라 파라미터 비교 + Stereo baseline
print(f"\n=== 4개 카메라 파라미터 비교 ===")
print(f"{'Cam':<4} {'fx':>10} {'fy':>10} {'cx':>10} {'cy':>10} {'tx (m)':>10}")
print("-" * 56)
for key in ("P0", "P1", "P2", "P3"):
    K_i, _, t_i = decompose_projection(calib[key])
    print(f"{key:<4} {K_i[0,0]:>10.4f} {K_i[1,1]:>10.4f} "
          f"{K_i[0,2]:>10.4f} {K_i[1,2]:>10.4f} {t_i[0]:>10.4f}")

_, _, t0 = decompose_projection(calib["P0"])
_, _, t1 = decompose_projection(calib["P1"])
_, _, t2 = decompose_projection(calib["P2"])
_, _, t3 = decompose_projection(calib["P3"])
print(f"\n흑백 stereo baseline (P0 → P1) = {abs(t1[0] - t0[0]):.4f} m")
print(f"컬러 stereo baseline (P2 → P3) = {abs(t3[0] - t2[0]):.4f} m") 

=== P0 ===
[[707.0912   0.     601.8873   0.    ]
 [  0.     707.0912 183.1104   0.    ]
 [  0.       0.       1.       0.    ]]

=== P1 ===
[[ 707.0912    0.      601.8873 -379.8145]
 [   0.      707.0912  183.1104    0.    ]
 [   0.        0.        1.        0.    ]]

=== P2 ===
[[7.070912e+02 0.000000e+00 6.018873e+02 4.688783e+01]
 [0.000000e+00 7.070912e+02 1.831104e+02 1.178601e-01]
 [0.000000e+00 0.000000e+00 1.000000e+00 6.203223e-03]]

=== P3 ===
[[ 7.070912e+02  0.000000e+00  6.018873e+02 -3.334597e+02]
 [ 0.000000e+00  7.070912e+02  1.831104e+02  1.930130e+00]
 [ 0.000000e+00  0.000000e+00  1.000000e+00  3.318498e-03]]

=== P0 Intrinsic K ===
[[707.0912   0.     601.8873]
 [  0.     707.0912 183.1104]
 [  0.       0.       1.    ]]
  f_x = 707.0912 px,  f_y = 707.0912 px
  c_x = 601.8873 px,  c_y = 183.1104 px

=== P0 Extrinsic ===
R =
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
t = [0. 0. 0.]

=== 4개 카메라 파라미터 비교 ===
Cam          fx         fy         cx         cy     tx (m)
--------------------------------------------------------
P0     707.0912   707.0912   601.8873   183.1104     0.0000
P1     707.0912   707.0912   601.8873   183.1104    -0.5372
P2     707.0912   707.0912   601.8873   183.1104     0.0610
P3     707.0912   707.0912   601.8873   183.1104    -0.4744

흑백 stereo baseline (P0 → P1) = 0.5372 m
컬러 stereo baseline (P2 → P3) = 0.5354 m



