# Wire Saw APC TF : virual para 생성 
import os
import os.path as pt
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression

# ============================================================
# 설정 하세요!
# ============================================================
CONFIG = {
    'paths': {
        'input_csv': r'./data/totalk_data.csv',
        'output_dir': r'./data',
    },
    'columns': {
        'c1_base': 'set_frame_temp_90pct',
        'c2_base': 'fdc_warm_up_time',
        'm_col': 'ctrl_out_wireguide_right_temp_0pct',
        # 'vp_c1', 'vp_c2' 는 Step 3 결과 확인 후 아래 SELECTED_FEATURES 에서 수동 설정
        'bow': {
            'TOTAL': 'avg_bow_bf_total',
            'SEED': 'avg_bow_bf_seed',
            'MID': 'avg_bow_bf_mid',
            'TAIL': 'avg_bow_bf_tail',
        },
        'base_features': [
            'set_frame_temp_0pct', 'set_frame_temp_10pct', 'set_frame_temp_20pct',
            'set_frame_temp_30pct', 'set_frame_temp_40pct', 'set_frame_temp_50pct',
            'set_frame_temp_60pct', 'set_frame_temp_70pct', 'set_frame_temp_80pct',
            'set_frame_temp_90pct', 'set_frame_temp_99pct', 'set_frame_temp_100pct'
        ]
    },
    'params': {
        'k_neighbors': 3,
        'close_pct': 15,
        'min_samples': 30,
        'correlation_threshold': 0.2,
    },
    'encoding': 'utf-8'
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def optimize_linear_combo(c1, c2, target):
    """
    VP = a * c1_norm + b * c2_norm 가 target 과의 절대 상관관계를 최대화하도록 최적화 (a^2 + b^2 = 1).
    """
    c1n = (c1 - c1.mean()) / (c1.std() + 1e-9)
    c2n = (c2 - c2.mean()) / (c2.std() + 1e-9)
    tn = (target - target.mean()) / (target.std() + 1e-9)

    def neg_abs_corr(theta):
        a, b = np.cos(theta[0]), np.sin(theta[0])
        vp = a * c1n + b * c2n
        if vp.std() < 1e-9:
            return 0.0
        return -abs(np.corrcoef(vp, tn)[0, 1])

    best = None
    for t0 in np.linspace(0, np.pi, 12):
        res = minimize(neg_abs_corr, [t0], method="Nelder-Mead")
        if best is None or res.fun < best.fun:
            best = res

    theta = best.x[0]
    a, b = np.cos(theta), np.sin(theta)
    vp = a * c1n + b * c2n
    r = np.corrcoef(vp, tn)[0, 1]

    if r < 0:
        a, b, vp, r = -a, -b, -vp, -r

    return a, b, r, vp


def create_virtual_parameters(df, c1_col, c2_col, m_col, y_col):
    """VP_A(M 재현), VP_B(Y 직접) 생성 후 df 에 컬럼 추가."""
    d = df.copy().reset_index(drop=True)
    valid = d[[c1_col, c2_col, m_col, y_col]].notna().all(axis=1)

    c1 = d.loc[valid, c1_col].values
    c2 = d.loc[valid, c2_col].values
    m = d.loc[valid, m_col].values
    y = d.loc[valid, y_col].values

    print(f"[PART 1] Virtual Parameter 생성 시작")
    print(f"유효 샘플 수: {valid.sum()}")

    base_corr_c1_y = np.corrcoef(c1, y)[0, 1]
    base_corr_c2_y = np.corrcoef(c2, y)[0, 1]
    base_corr_m_y = np.corrcoef(m, y)[0, 1]
    print(f"기존 상관: C1-Y={base_corr_c1_y:+.3f}, C2-Y={base_corr_c2_y:+.3f}, M-Y={base_corr_m_y:+.3f}")

    aA, bA, rA_m, vpA = optimize_linear_combo(c1, c2, m)
    rA_y = np.corrcoef(vpA, y)[0, 1]

    aB, bB, rB_y, vpB = optimize_linear_combo(c1, c2, y)
    rB_m = np.corrcoef(vpB, m)[0, 1]

    print(f"[VP_A: M 재현] a={aA:+.3f}, b={bA:+.3f} | M 상관={rA_m:+.3f}, Y 상관={rA_y:+.3f}")
    print(f"[VP_B: Y 직접] a={aB:+.3f}, b={bB:+.3f} | Y 상관={rB_y:+.3f}, M 상관={rB_m:+.3f}")

    d['VP_A'] = np.nan
    d['VP_B'] = np.nan
    d.loc[valid, 'VP_A'] = vpA
    d.loc[valid, 'VP_B'] = vpB

    vp_info = {
        'VP_A': {'a': aA, 'b': bA, 'corr_M': rA_m, 'corr_Y': rA_y},
        'VP_B': {'a': aB, 'b': bB, 'corr_M': rB_m, 'corr_Y': rB_y},
    }
    return d, vp_info


def _reproducibility_metrics(X_df, y, k_neighbors=3, close_pct=15):
    """재현성 비율 및 R^2 상한 계산"""
    X_std = ((X_df - X_df.mean()) / (X_df.std() + 1e-9)).fillna(0).values
    y_arr = np.asarray(y)
    n = len(y_arr)
    y_std_total = y_arr.std()

    dist = squareform(pdist(X_std, metric='euclidean'))
    np.fill_diagonal(dist, np.inf)

    knn_idx = np.argsort(dist, axis=1)[:, :k_neighbors]
    mean_nn_dist = np.sort(dist, axis=1)[:, :k_neighbors].mean(axis=1)

    y_diff_knn = np.array([
        np.mean(np.abs(y_arr[i] - y_arr[knn_idx[i]])) for i in range(n)
    ])

    close_thr = np.percentile(mean_nn_dist, close_pct)
    close_mask = mean_nn_dist <= close_thr

    if close_mask.sum() > 0:
        median_close_ydiff = np.median(y_diff_knn[close_mask])
    else:
        median_close_ydiff = np.nan

    ratio = median_close_ydiff / (y_std_total + 1e-9)

    close_ydiffs = np.array([
        np.abs(y_arr[i] - y_arr[knn_idx[i][0]]) for i in range(n) if close_mask[i]
    ])

    if len(close_ydiffs) > 1:
        noise_var = np.mean(close_ydiffs ** 2) / 2
        y_var = y_arr.var()
        r2_ceiling = max(0.0, 1 - noise_var / y_var) if y_var > 0 else 0.0
    else:
        noise_var, r2_ceiling = np.nan, np.nan

    return {
        'ratio': ratio,
        'r2_ceiling': r2_ceiling,
        'noise_var': noise_var,
        'y_std_total': y_std_total,
        'mean_nn_dist': mean_nn_dist,
        'y_diff_knn': y_diff_knn,
        'close_thr': close_thr,
        'median_close_ydiff': median_close_ydiff,
    }


def run_pipeline(df, c1_col, c2_col, m_col, y_col, base_feature_cols):
    """VP 생성 및 재현성/R^2 상한 진단 파이프라인 실행"""
    d, vp_info = create_virtual_parameters(df, c1_col, c2_col, m_col, y_col)

    need_cols = base_feature_cols + ['VP_A', 'VP_B', y_col]
    need_cols = [c for c in need_cols if c in d.columns]
    dv = d[need_cols].dropna().reset_index(drop=True)
    y = dv[y_col].values

    print(f"\n[PART 2] 재현성 및 R^2 상한 진단 (VP 전/후 비교)")

    configs = {
        'Existing_X_Only': base_feature_cols,
        'Existing_X_Plus_VP_A': base_feature_cols + ['VP_A'],
        'Existing_X_Plus_VP_B': base_feature_cols + ['VP_B'],
    }

    results = {}
    for name, cols in configs.items():
        cols = [c for c in cols if c in dv.columns]
        if not cols:
            continue
        met = _reproducibility_metrics(dv[cols], y,
                                       k_neighbors=CONFIG['params']['k_neighbors'],
                                       close_pct=CONFIG['params']['close_pct'])
        results[name] = met
        print(f"{name}: 재현성 비율={met['ratio']:.3f}, 추정 R^2 상한={met['r2_ceiling']:.3f}")

    base_ceil = results.get('Existing_X_Only', {}).get('r2_ceiling', np.nan)
    best_vp_ceil = max(
        results.get('Existing_X_Plus_VP_A', {}).get('r2_ceiling', 0),
        results.get('Existing_X_Plus_VP_B', {}).get('r2_ceiling', 0),
    )
    gain = best_vp_ceil - base_ceil if not np.isnan(base_ceil) else np.nan

    print(f"\n종합 결론: 기존 R^2 상한={base_ceil:.3f}, 최대 상한={best_vp_ceil:.3f}, 변화량={gain:+.3f}")

    if len(results) > 0:
        best_ratio = min(m['ratio'] for m in results.values())
        if best_ratio > 0.7:
            print("판단: 모든 구성에서 재현성 낮음. 핵심 변수 누락 가능성.")
        elif best_vp_ceil < 0.1:
            print("판단: R^2 상한 자체가 낮음. 현재 변수로는 예측 구조적 어려움.")
        else:
            print("판단: VP 추가로 개선 여지 확인됨. 모델 재설계 권장.")

    return d, vp_info, results


# ============================================================
# STEP 1 ~ 3: 잔차 분석 및 후보 인자 스크리닝
# ============================================================
def run_step_1_to_3(df, out_dir):
    """
    Step 1: M 잔차 생성
    Step 2: 잔차 vs BOW 검증
    Step 3: 잔차 설명 후보 스크리닝 (여기서 중단되고 결과 파일 저장)
    """
    d = df.copy().reset_index(drop=True)

    c1_base = CONFIG['columns']['c1_base']
    c2_base = CONFIG['columns']['c2_base']
    m_col = CONFIG['columns']['m_col']
    bow_cols = CONFIG['columns']['bow']

    print(f"=== STEP 1. M 잔차 생성 (M \~ {c1_base} + {c2_base}) ===")
    req1 = [c1_base, c2_base, m_col]
    mask1 = d[req1].notna().all(axis=1)
    idx1 = d[mask1].index

    if idx1.empty:
        raise ValueError("Step 1 수행을 위한 유효 데이터가 없습니다. 결측치를 확인하세요.")

    lr = LinearRegression().fit(d.loc[idx1, [c1_base, c2_base]], d.loc[idx1, m_col])

    d['M_explained'] = np.nan
    d['M_residual'] = np.nan
    d.loc[idx1, 'M_explained'] = lr.predict(d.loc[idx1, [c1_base, c2_base]])
    d.loc[idx1, 'M_residual'] = d.loc[idx1, m_col] - d.loc[idx1, 'M_explained']

    r2_m = lr.score(d.loc[idx1, [c1_base, c2_base]], d.loc[idx1, m_col])
    print(f"유효 샘플: {mask1.sum()}, R^2: {r2_m:.3f}")
    print(f"회귀 계수: {c1_base}={lr.coef_[0]:.4f}, {c2_base}={lr.coef_[1]:.4f}, intercept={lr.intercept_:.4f}")

    print(f"\n=== STEP 2. 잔차/설명분 vs BOW 상관 검증 ===")
    for pos, bow_col in bow_cols.items():
        if bow_col not in d.columns:
            continue
        sub = d[['M_residual', 'M_explained', bow_col]].dropna()
        if len(sub) < 2: continue

        r_res, p_res = stats.pearsonr(sub['M_residual'], sub[bow_col])
        r_exp, p_exp = stats.pearsonr(sub['M_explained'], sub[bow_col])
        marker = ' [Significant]' if abs(r_res) >= CONFIG['params']['correlation_threshold'] else ''
        print(f"{pos}: Residual r={r_res:+.3f} (p={p_res:.4f}){marker}, Explained r={r_exp:+.3f}")

    print(f"\n=== STEP 3. 잔차 설명 후보 스크리닝 ===")
    candidate_cols = [c for c in d.columns
                      if (c.startswith('fdc_') or c.startswith('set_'))
                      and c not in [c1_base, c2_base, m_col]
                      and pd.api.types.is_numeric_dtype(d[c])]

    screen = []
    for col in candidate_cols:
        pair = d[['M_residual', col]].dropna()
        if len(pair) < CONFIG['params']['min_samples'] or pair[col].nunique() < 2:
            continue
        try:
            r, p = stats.pearsonr(pair[col], pair['M_residual'])
            screen.append({'Feature': col, 'r': round(r, 3), 'p': round(p, 4), 'N': len(pair)})
        except Exception:
            continue

    screen_df = pd.DataFrame(screen).sort_values('r', key=abs, ascending=False)
    print("\n상위 15 개 후보 인자")
    print(screen_df.head(15).to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    screen_path = pt.join(out_dir, "residual_screening.csv")
    screen_df.to_csv(screen_path, index=False, encoding=CONFIG['encoding'])
    print(f"\n스크리닝 결과 저장 완료: {screen_path}")
    print(f"--> 위 결과를 확인한 후, 코드 상단 'SELECTED_FEATURES' 에 vp_c1, vp_c2 를 입력하고 Step 4~6 을 실행하세요.")

    return d, screen_df, lr


# ============================================================
# STEP 4 ~ 6: VP_C 생성 및 최종 파라미터 저장
# ============================================================
def run_step_4_to_6(df, out_dir, selected_c1, selected_c2, lr_model=None):
    """
    Step 4: 선택된 인자로 VP_C 생성
    Step 5: VP_C vs BOW 검증
    Step 6: 파라미터 저장
    """
    d = df.copy().reset_index(drop=True)

    c1_base = CONFIG['columns']['c1_base']
    c2_base = CONFIG['columns']['c2_base']
    m_col = CONFIG['columns']['m_col']
    bow_cols = CONFIG['columns']['bow']

    # Step 1 의 회귀 모델이 전달되지 않았다면 재계산 (일관성 위해 권장되지 않으나 안전장치)
    if lr_model is None:
        print("경고: 이전 단계의 회귀 모델 정보가 없습니다. 잔차 계산을 위해 재학습을 수행합니다.")
        req1 = [c1_base, c2_base, m_col]
        mask1 = d[req1].notna().all(axis=1)
        if mask1.sum() == 0:
            raise ValueError("잔차 계산을 위한 유효 데이터가 없습니다.")
        lr_model = LinearRegression().fit(d.loc[mask1, [c1_base, c2_base]], d.loc[mask1, m_col])

        d['M_explained'] = np.nan
        d['M_residual'] = np.nan
        d.loc[mask1, 'M_explained'] = lr_model.predict(d.loc[mask1, [c1_base, c2_base]])
        d.loc[mask1, 'M_residual'] = d.loc[mask1, m_col] - d.loc[mask1, 'M_explained']
    else:
        # 기존 DataFrame 에 잔차가 없으면 계산
        if 'M_residual' not in d.columns or d['M_residual'].isna().all():
            req1 = [c1_base, c2_base, m_col]
            mask1 = d[req1].notna().all(axis=1)
            d['M_explained'] = np.nan
            d['M_residual'] = np.nan
            d.loc[mask1, 'M_explained'] = lr_model.predict(d.loc[mask1, [c1_base, c2_base]])
            d.loc[mask1, 'M_residual'] = d.loc[mask1, m_col] - d.loc[mask1, 'M_explained']

    print(f"=== STEP 4. VP_C 생성 ({selected_c1} + {selected_c2} -> M_residual) ===")

    # VP_C 생성에 필요한 데이터 추출
    req4 = [selected_c1, selected_c2, 'M_residual']
    mask4 = d[req4].notna().all(axis=1)

    if mask4.sum() < CONFIG['params']['min_samples']:
        raise ValueError(f"VP_C 생성을 위한 유효 샘플이 부족합니다 ({mask4.sum()}개). 결측치를 확인하거나 다른 인자를 선택하세요.")

    idx4 = d[mask4].index

    c1_vals = d.loc[idx4, selected_c1].values
    c2_vals = d.loc[idx4, selected_c2].values
    m_res_vals = d.loc[idx4, 'M_residual'].values

    a, b, r_vpc_res, vpc = optimize_linear_combo(c1_vals, c2_vals, m_res_vals)

    d['VP_C'] = np.nan
    d.loc[idx4, 'VP_C'] = vpc

    print(f"최적 계수: a={a:+.3f} ({selected_c1}), b={b:+.3f} ({selected_c2})")
    print(f"VP_C - M 잔차 상관: r={r_vpc_res:+.3f}")

    print(f"\n=== STEP 5. VP_C vs 위치별 BOW 검증 ===")
    vpc_bow_results = []
    for pos, bow_col in bow_cols.items():
        if bow_col not in d.columns:
            continue
        sub = d[['VP_C', bow_col]].dropna()
        if len(sub) < 2: continue

        r, p = stats.pearsonr(sub['VP_C'], sub[bow_col])
        marker = ' [Significant]' if abs(r) >= CONFIG['params']['correlation_threshold'] else ''
        print(f"VP_C vs BOW_{pos}: r={r:+.3f} (p={p:.4f}){marker}")
        vpc_bow_results.append({'Position': pos, 'r': r, 'p': p, 'N': len(sub)})

    print(f"\n=== STEP 6. 파라미터 저장 ===")
    vp_params = {
        'residual_regression': {
            'features': [c1_base, c2_base],
            'coef': lr_model.coef_.tolist(),
            'intercept': float(lr_model.intercept_),
            'target': m_col,
        },
        'vp_c': {
            'c1': selected_c1, 'c2': selected_c2,
            'a': float(a), 'b': float(b),
            'c1_mean': float(np.mean(c1_vals)), 'c1_std': float(np.std(c1_vals)),
            'c2_mean': float(np.mean(c2_vals)), 'c2_std': float(np.std(c2_vals)),
            'corr_residual': float(r_vpc_res),
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    param_path = pt.join(out_dir, "vp_c_params.json")
    with open(param_path, 'w', encoding=CONFIG['encoding']) as f:
        json.dump(vp_params, f, indent=2, ensure_ascii=False)
    print(f"파라미터 저장 완료: {param_path}")

    r_tail_vpc = [r['r'] for r in vpc_bow_results if r['Position'] == 'TAIL']
    if r_tail_vpc:
        r_final = r_tail_vpc[0]
        print(f"\n최종 요약: VP_C - TAIL BOW 상관 = {r_final:.3f}")
        if abs(r_final) >= CONFIG['params']['correlation_threshold']:
            print("결론: 제어 가능 인자 경로 성립 (실무 활용 가능)")
        else:
            print("결론: 경로 약함. 추가 변수 또는 비선형 조합 필요")

    return d, vp_params


def apply_vp_c(df_new, params_path):
    """저장된 파라미터를 사용하여 신규 데이터에 VP_C 적용"""
    with open(params_path, encoding=CONFIG['encoding']) as f:
        p = json.load(f)

    vc = p['vp_c']
    d = df_new.copy()

    # 존재하지 않는 컬럼 처리
    if vc['c1'] not in d.columns or vc['c2'] not in d.columns:
        raise KeyError(f"신규 데이터에 필요한 컬럼 {vc['c1']}, {vc['c2']} 이 존재하지 않습니다.")

    c1_norm = (d[vc['c1']] - vc['c1_mean']) / (vc['c1_std'] + 1e-9)
    c2_norm = (d[vc['c2']] - vc['c2_mean']) / (vc['c2_std'] + 1e-9)

    d['VP_C'] = vc['a'] * c1_norm + vc['b'] * c2_norm
    return d


input_path = CONFIG['paths']['input_csv']
output_dir = CONFIG['paths']['output_dir']

if not os.path.exists(input_path):
    raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")


import pandas as pd

df = pd.read_csv(input_path, encoding='utf-8')

# recipe
df_r = df[df['process_time']=='13.3Hr']
# df_r = df[df['process_time']=='18.5Hr']

print("\n>>> [Flow 1] Step 1~3: 잔차 분석 및 후보 인자 스크리닝 시작")
d_interim, screen_df, lr_model = run_step_1_to_3(df_r, output_dir)


##### 설정 하세요!
# 위 Step 3 결과를 보고 M_residual 과 상관관계가 높은 상위 2 개 컬럼명을 아래에 직접 입력하세요.
SELECTED_FEATURES = {
    'vp_c1': 'set_slurry_temp_99pct',        # 예: 'fdc_set_tension'
    'vp_c2': 'fdc_set_tension' # 예: 'set_frame_temp_30pct'
}
#####


sel_c1 = SELECTED_FEATURES.get('vp_c1')
sel_c2 = SELECTED_FEATURES.get('vp_c2')

if not sel_c1 or not sel_c2:
    print("\n[중단] SELECTED_FEATURES 에 vp_c1, vp_c2 가 설정되지 않았습니다.")
    print("1. 출력된 'residual_screening.csv' 를 확인하세요.")
    print("2. 코드 상단 SELECTED_FEATURES 변수에 적절한 컬럼명을 입력하세요.")
    print("3. 코드를 다시 실행하세요.")
else:
    print(f"\n>>> [Flow 2] Step 4~6: VP_C 생성 시작 (선택 인자: {sel_c1}, {sel_c2})")
    d_final, vp_params = run_step_4_to_6(df, output_dir, sel_c1, sel_c2, lr_model)

    # 적용 테스트
    df_test = apply_vp_c(d_final, pt.join(output_dir, "vp_c_params.json"))
    print("\n전체 프로세스 완료.")


# ============================================================
# 실행 결과
# ============================================================

# <>:20: SyntaxWarning: invalid escape sequence '\~'
# <>:20: SyntaxWarning: invalid escape sequence '\~'
# C:\Users\Sksiltron\AppData\Local\Temp\ipykernel_41520\792990836.py:20: SyntaxWarning: invalid escape sequence '\~'
#   print(f"\n>>> [Flow 2] Step 4~6: VP_C 생성 시작 (선택 인자: {sel_c1}, {sel_c2})")
#
# >>> [Flow 2] Step 4~6: VP_C 생성 시작 (선택 인자: set_slurry_temp_99pct, fdc_set_tension)
# === STEP 4. VP_C 생성 (set_slurry_temp_99pct + fdc_set_tension -> M_residual) ===
# 최적 계수: a=+0.984 (set_slurry_temp_99pct), b=+0.177 (fdc_set_tension)
# VP_C - M 잔차 상관: r=+0.112
#
# === STEP 5. VP_C vs 위치별 BOW 검증 ===
# VP_C vs BOW_TOTAL: r=-0.089 (p=0.0000)
# VP_C vs BOW_SEED: r=-0.094 (p=0.0000)
# VP_C vs BOW_MID: r=-0.089 (p=0.0000)
# VP_C vs BOW_TAIL: r=-0.048 (p=0.0000)
#
# === STEP 6. 파라미터 저장 ===
# 파라미터 저장 완료: ./data\vp_c_params.json
