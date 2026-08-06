# FDC에 어떤 식별 컬럼이 있고, Y/3200과 뭐가 겹치는지
print('[진단] FDC 컬럼:', [c for c in df_fdc.columns if any(k in c.upper() for k in ['LOT','WIRE','BLK','ID','SUBLOT','CAR','CST'])])
print('[진단] Y/3200(df_merged) 식별 컬럼:', [c for c in df_merged.columns if any(k in c.upper() for k in ['LOT','WIRE','BLK','ID','SUBLOT','CAR','CST'])])

# FDC의 NEW_WIRE_ID vs Y쪽 wire 관련 컬럼 값 비교
if 'NEW_WIRE_ID' in df_fdc.columns:
    print('[진단] FDC NEW_WIRE_ID 샘플:', df_fdc['NEW_WIRE_ID'].dropna().unique()[:5].tolist())




print('[진단] 3200 컬럼 전체:', df_3200.columns.tolist())
print('[진단] 3200 각 식별 컬럼 샘플:')
for c in df_3200.columns:
    if any(k in c.upper() for k in ['LOT','WIRE','BLK','ID','CAR','CST','SUBLOT']):
        print(f'   {c}:', df_3200[c].dropna().unique()[:3].tolist())
