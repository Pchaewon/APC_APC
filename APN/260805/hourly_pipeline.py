# -*- coding: utf-8 -*-
"""
시간별 리포트 메일링 파이프라인 (배포 오케스트레이션)
─────────────────────────────────────────
두 단계:
  [1] 학습 (오프라인, 주기적): 저장된 전체 데이터 → dual 모델 학습·저장
  [2] 추천 (1시간마다): DB에서 test 데이터 → 10개 장비 리포트 → 메일 발송

test 데이터 로드(DB)는 fetch_test_data()에 인터페이스만 두었음.
추후 실제 DB 쿼리로 채우면 됨.

실행:
  python hourly_pipeline.py train    # 모델 학습 (가끔)
  python hourly_pipeline.py run       # 1시간마다 실행할 추천+메일
  python hourly_pipeline.py once      # 메일 없이 리포트만 생성 (테스트)
"""
import os
import os.path as pt
import sys
import glob
import copy
from datetime import datetime

# 같은 폴더 모듈
import train_inverse_dual as dual
import generate_report as report


# ═══════════════════════════════════════
# 전역 설정
# ═══════════════════════════════════════
PIPELINE_CONFIG = {
    # ── 경로 ──
    'train_csv':  r'D:\chaewon\APC\02.TF\260726\data\data.csv',   # 학습 전체
    'model_dir':  r'./apc_model_dual',
    'report_dir': r'./reports',
    'log_dir':    r'./pipeline_logs',

    # ── 대상 장비 (추천 리포트 만들 장비) ──
    'target_eqps': {
        '15-19': ['BSWS30','BSWS31','BSWS35','BSWS42','BSWS48'],
        '21':    ['BSWS51','BSWS53','BSWS55','BSWS57','BSWS58'],
    },

    # ── 메일 설정 (추후 채움) ──
    'mail': {
        'enabled': False,                 # True로 바꾸면 실제 발송
        'smtp_host': '',                  # 예: 'smtp.company.com'
        'smtp_port': 587,
        'sender': '',                     # 발신 주소
        'recipients': [],                 # 수신자 리스트
        'subject_tpl': '[Wire Saw APC] {eqp} Recipe 추천 ({time})',
    },

    'target_bow': 1.75,
    'encoding': 'utf-8',
}


# ═══════════════════════════════════════
# [1] 학습 단계
# ═══════════════════════════════════════
def train_models(pcfg):
    """저장된 전체 데이터로 dual 모델(frame/slurry) 학습·저장."""
    print(f"\n{'='*56}\n[학습] 전체 데이터 기반 dual 모델\n{'='*56}")
    # dual 모듈의 CONFIG를 파이프라인 설정으로 덮어씀
    dcfg = copy.deepcopy(dual.CONFIG)
    dcfg['input_csv'] = pcfg['train_csv']
    dcfg['model_dir'] = pcfg['model_dir']
    dcfg['target_bow'] = pcfg['target_bow'] if 'target_bow' in dual.CONFIG else dcfg.get('inverse_target_bow')
    dual.train(dcfg)
    print(f"✅ 모델 저장: {pcfg['model_dir']}/")


# ═══════════════════════════════════════
# [2-a] test 데이터 로드 (DB) — 추후 구현
# ═══════════════════════════════════════
def fetch_test_data(pcfg, eqp):
    """
    ★ 추후 구현: DB에서 해당 장비의 최근 데이터를 가져옴.

    반환해야 할 것: pandas DataFrame
      - 리포트에 필요한 컬럼 모두 포함
        (eqp_nm_3200, new_fdc_wire_id, date_3200, process_time,
         set_frame_temp_*pct, set_slurry_temp_*pct,
         shift_of_wireguide_l/r_*pct,
         avg_bow_bf_*, avg_warp_bf_*,
         fdc_ingot_len, fdc_wait_time, warm_up_time,
         range_slurry_temp_10_0, range_wire_guide_10_99 등)
      - 최근 N lot (리포트 trend_n + rolling window 충분히)

    현재는 placeholder. 실제 DB 연동 시 이 함수만 채우면 됨.
    예:
        import pyodbc / sqlalchemy
        query = f"SELECT ... WHERE eqp='{eqp}' AND date >= ... ORDER BY date"
        return pd.read_sql(query, conn)
    """
    raise NotImplementedError(
        f"fetch_test_data 미구현 — DB 연동 코드를 여기에 작성하세요. "
        f"(장비 {eqp}의 최근 데이터를 DataFrame으로 반환)")


def fetch_test_data_from_csv(pcfg, eqp, csv_path):
    """임시: CSV에서 해당 장비 데이터 로드 (DB 대체, 테스트용)."""
    import pandas as pd
    df = pd.read_csv(csv_path, encoding=pcfg['encoding'], encoding_errors='replace')
    return df[df['eqp_nm_3200'] == eqp].copy()


# ═══════════════════════════════════════
# [2-b] 장비별 리포트 생성
# ═══════════════════════════════════════
def generate_reports(pcfg, test_csv=None):
    """
    대상 장비별로 test 데이터 로드 → 리포트 HTML 생성.
    test_csv 주어지면 CSV에서(테스트), 아니면 fetch_test_data(DB)에서.
    반환: [(eqp, html_path), ...]
    """
    os.makedirs(pcfg['report_dir'], exist_ok=True)
    all_eqps = []
    for grp, eqps in pcfg['target_eqps'].items():
        all_eqps.extend(eqps)
    print(f"\n{'='*56}\n[리포트] {len(all_eqps)}개 장비\n{'='*56}")

    results = []
    for eqp in all_eqps:
        try:
            # test 데이터 확보
            if test_csv:
                tdf = fetch_test_data_from_csv(pcfg, eqp, test_csv)
                src = 'CSV'
            else:
                tdf = fetch_test_data(pcfg, eqp)
                src = 'DB'

            if tdf is None or len(tdf) == 0:
                print(f"  ⚠ {eqp}: 데이터 없음 — 스킵")
                continue

            # report 모듈 CONFIG 구성 (임시 CSV로 저장해 넘김)
            tmp_csv = pt.join(pcfg['report_dir'], f'_tmp_{eqp}.csv')
            tdf.to_csv(tmp_csv, index=False, encoding='utf-8-sig')

            rcfg = copy.deepcopy(report.CONFIG)
            rcfg['model_dir'] = pcfg['model_dir']
            rcfg['recent_csv'] = tmp_csv
            rcfg['eqp_name'] = eqp
            rcfg['target_bow'] = pcfg['target_bow']

            html_path = report.build_report(rcfg)
            results.append((eqp, html_path))
            os.remove(tmp_csv)
            print(f"  ✅ {eqp}: {pt.basename(html_path)} ({src})")
        except NotImplementedError as e:
            print(f"  ⏸ {eqp}: {e}")
        except Exception as e:
            print(f"  ⚠ {eqp} 실패: {e}")
    return results


# ═══════════════════════════════════════
# [2-c] 메일 발송
# ═══════════════════════════════════════
def send_mail(pcfg, eqp, html_path):
    """
    HTML 리포트를 메일 본문으로 발송.
    mail.enabled=False면 발송 안 하고 로그만.
    """
    mcfg = pcfg['mail']
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    subject = mcfg['subject_tpl'].format(eqp=eqp, time=time_str)

    if not mcfg['enabled']:
        print(f"  [메일 OFF] {eqp} → {subject}")
        return False

    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.application import MIMEApplication

        html = open(html_path, encoding='utf-8').read()
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = mcfg['sender']
        msg['To'] = ', '.join(mcfg['recipients'])
        msg.attach(MIMEText(html, 'html', 'utf-8'))
        # HTML 파일도 첨부 (선택)
        with open(html_path, 'rb') as f:
            att = MIMEApplication(f.read(), _subtype='html')
            att.add_header('Content-Disposition', 'attachment',
                           filename=pt.basename(html_path))
            msg.attach(att)

        with smtplib.SMTP(mcfg['smtp_host'], mcfg['smtp_port']) as s:
            s.starttls()
            # s.login(user, pw)  # 필요 시
            s.sendmail(mcfg['sender'], mcfg['recipients'], msg.as_string())
        print(f"  ✉ {eqp} 발송 완료 → {len(mcfg['recipients'])}명")
        return True
    except Exception as e:
        print(f"  ⚠ {eqp} 메일 실패: {e}")
        return False


# ═══════════════════════════════════════
# 오케스트레이션
# ═══════════════════════════════════════
def run_hourly(pcfg, test_csv=None, send=True):
    """1시간마다 실행: 리포트 생성 → 메일 발송."""
    os.makedirs(pcfg['log_dir'], exist_ok=True)
    t0 = datetime.now()
    print(f"\n{'#'*56}\n# 시간별 파이프라인 실행: {t0:%Y-%m-%d %H:%M:%S}\n{'#'*56}")

    # 모델 존재 확인
    for name in ['frame', 'slurry']:
        if not os.path.exists(pt.join(pcfg['model_dir'], name, 'model.pkl')):
            print(f"❌ 모델 없음: {name} — 먼저 'train' 실행 필요")
            return

    reports = generate_reports(pcfg, test_csv=test_csv)

    if send:
        print(f"\n[메일 발송]")
        sent = 0
        for eqp, path in reports:
            if send_mail(pcfg, eqp, path):
                sent += 1
        print(f"\n발송: {sent}/{len(reports)}")

    # 로그
    dt = (datetime.now() - t0).total_seconds()
    log = pt.join(pcfg['log_dir'], f'run_{t0:%Y%m%d_%H%M%S}.log')
    with open(log, 'w', encoding='utf-8') as f:
        f.write(f"실행: {t0}\n리포트: {len(reports)}개\n소요: {dt:.1f}초\n")
        for eqp, path in reports:
            f.write(f"  {eqp}: {path}\n")
    print(f"\n✅ 완료 ({dt:.1f}초) · 로그: {log}")
    return reports


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    # 테스트용 CSV 경로 (있으면 DB 대신 사용)
    test_csv = sys.argv[2] if len(sys.argv) > 2 else None

    if mode == 'train':
        train_models(PIPELINE_CONFIG)
    elif mode == 'run':
        # DB에서 가져와 리포트+메일 (fetch_test_data 구현 필요)
        run_hourly(PIPELINE_CONFIG, test_csv=test_csv, send=True)
    elif mode == 'once':
        # 메일 없이 리포트만 (테스트) — test_csv 필요
        run_hourly(PIPELINE_CONFIG, test_csv=test_csv, send=False)
    else:
        print("사용법: python hourly_pipeline.py [train|run|once] [test_csv]")
