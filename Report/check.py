import pandas as pd
d = pd.read_csv('./data/field_store.csv')
print('frame 실제:', [c for c in d.columns if 'set_frame_temp' in c][:3], '...')
print('slurry 실제:', [c for c in d.columns if 'set_slurry_temp' in c][:3], '...')
print('bow/warp:', [c for c in d.columns if 'bow' in c.lower() or 'warp' in c.lower()])
print('조건:', [c for c in d.columns if any(k in c for k in ['ingot','wait','warm'])])
print('식별:', [c for c in d.columns if any(k in c for k in ['eqp','wire','date'])])
