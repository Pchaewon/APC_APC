import pandas as pd
d = pd.read_csv('./data/WireSaw_Field_Test_preprocessed.csv', encoding='cp949')
# 날짜 컬럼 찾기
datecol = [c for c in d.columns if 'DTTM' in c.upper() or 'DATE' in c.upper()][:3]
print('날짜 컬럼:', datecol)
for c in datecol:
    print(f'{c} 날짜분포:', d[c].astype(str).str[:8].value_counts().to_dict())
print('총', len(d), '행')
