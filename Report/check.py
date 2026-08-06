from datetime import timedelta   # 파일 상단에 이미 있으면 생략

# 검사 날짜 (기존)
meas_dates_dt = sorted(pd.to_datetime(
    df_3200['HIS_REGIST_DTTM'].astype(str).str[:8], errors='coerce'
).dropna().unique().tolist())

# ★ FDC는 가공 시점 기준이라 검사보다 며칠 앞섬 → 룩백 확장
FDC_LOOKBACK_DAYS = 4   # 18Hr wire + 버퍼 (필요시 조정)
fdc_date_set = set()
for d in meas_dates_dt:
    for i in range(FDC_LOOKBACK_DAYS + 1):
        fdc_date_set.add((d - timedelta(days=i)).strftime('%Y%m%d'))
meas_dates = sorted(fdc_date_set)
print(f"  검사 날짜: {[d.strftime('%Y%m%d') for d in meas_dates_dt]}")
print(f"  FDC 조회 날짜(룩백 {FDC_LOOKBACK_DAYS}일): {meas_dates}")
