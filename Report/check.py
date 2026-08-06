print('[진단] Y쪽 BLK_NO 샘플:', df_merged['BLK_NO'].dropna().unique()[:5].tolist())
print('[진단] FDC쪽 BLK_NO(원래 LOT_ID) 샘플:', fdc_agg['BLK_NO'].dropna().unique()[:5].tolist())
print('[진단] Y BLK_NO 개수:', df_merged['BLK_NO'].nunique(),
      '/ FDC 개수:', fdc_agg['BLK_NO'].nunique())
