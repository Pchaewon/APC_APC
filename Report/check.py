import pandas as pd
d = pd.read_csv('./recommend_future.csv')
pcts = [0,10,20,30,40,50,60,70,80,90,100]  # 11개
for _, r in d.iterrows():
    print(f"\n=== {r['eqp']} (WAF {int(r['n_waf_used'])}개) ===")
    fr = [round(r.get(f'rec_set_frame_temp_{p}pct'),2) for p in pcts]
    sl = [round(r.get(f'rec_set_slurry_temp_{p}pct'),2) for p in pcts]
    print('frame :', fr)
    print('slurry:', sl)
    print('pred_bow:', round(r['frame_pred_bow'],3), '/', round(r['slurry_pred_bow'],3))
