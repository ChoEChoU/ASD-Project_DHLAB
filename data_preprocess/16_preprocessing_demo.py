# data_preprocess/16_preprocessing_demo.py
import os
import argparse
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--excel",
        type=str,
        default="./data/raws/demo/Demography_all.xlsx",
        help="원본 설문 엑셀 경로"
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="all responses",
        help="엑셀 시트명"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./data/preprocessed/tabular",
        help="전처리 CSV 저장 디렉토리"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---------------------------
    # 열 정의
    # ---------------------------
    p1_cols = ['patient_id', 'P1B4', 'P1B5', 'P1B6', 'P1B8', 'P1B9', 'P1B10', 'P1B11', 'P1B12', 'P1B13', 'P1B14', 'P1B15']
    p2_cols = ['patient_id', 'P2B2', 'P2B22', 'P2B24', 'P2B27', 'P2B32', 'P2B38', 'P2B45'] + [f'P2B{i}' for i in range(3, 22)]
    p3_cols = ['patient_id', 'P3B2', 'P3B5', 'P3B9', 'P3B13', 'P3B17', 'P3B21', 'P3B25']
    p4_cols = ['patient_id', 'P4B2', 'P4B11', 'P4B15', 'P4B17', 'P4B19', 'P4B21', 'P4B23', 'P4B25', 'P4B27', 'P4B29', 'P4B31', 'P4B33', 'P4B35']
    p5_cols = ['patient_id', 'P5B2', 'P5B5', 'P5B6', 'P5B9', 'P5B11', 'P5B13', 'P5B15', 'P5B17', 'P5B19', 'P5B21', 'P5B24', 'P5B25', 'P5B26', 'P5B28']

    # ---------------------------
    # 데이터 로드/정리
    # ---------------------------
    data = pd.read_excel(args.excel, sheet_name=args.sheet)
    # patient_id 열 통일
    data = data.rename(columns={'P1B2': 'patient_id'})

    # patient_id 필터: 'B-'로 시작하는 행만
    data = data.drop(data[~data['patient_id'].astype(str).str.startswith('B-', na=False)].index)

    # 중복 제거(유지: 마지막 항목)
    data = data.drop_duplicates(subset='patient_id', keep='last')

    # 상단 설명행 등 제거(기존 코드 유지)
    if len(data) >= 2:
        data = data.drop(data.index[:2])

    # ---------------------------
    # P1
    # ---------------------------
    df = data[[c for c in p1_cols if c in data.columns]].copy()
    df.replace({'예': 0, '아니오': 1, '아니요': 1, '모르겠다': 2}, inplace=True)
    if 'P1B6' in df:  df['P1B6'] = df['P1B6'].map({'부': 0, '모': 1, '기타': 2})
    if 'P1B8' in df:  df['P1B8'] = df['P1B8'].map({'2세대 가구(예시:부모+자녀)': 0, '3세대 가구(예시:조부모+부모+자녀)': 1, '4세대 이상 가구': 2})

    if {'P1B10', 'P1B11'}.issubset(df.columns):
        df.loc[df['P1B10'] == '기타', 'P1B10'] = df.loc[df['P1B10'] == '기타', 'P1B11']
        df.drop('P1B11', axis=1, inplace=True)
        df['P1B10'] = df['P1B10'].map({
            '첫째아': 0, '둘째아': 1,
            '셋째이상': 2, '셋째아': 2, '셋째': 2, '넷째': 2, '4째아': 2, '셋째딸': 2,
            '쌍둥이': 3, '첫째아, 쌍둥이': 3, '둘째아, 쌍둥이': 3, '쌍둥이, 기타': 3,
            '외동아': 4, '첫째아, 외동아': 4
        })

    if {'P1B12', 'P1B13'}.issubset(df.columns):
        df.loc[df['P1B12'] == '기타', 'P1B12'] = df.loc[df['P1B12'] == '기타', 'P1B13']
        df.drop('P1B13', axis=1, inplace=True)
        df['P1B12'] = df['P1B12'].map({'부': 0, '모': 1, '조부모': 2, '공동양육': 3, '부,모 공동 주양육': 3, '엄마, 시터': 1}).fillna(4)

    if 'P1B14' in df: df['P1B14'] = df['P1B14'].map({'고졸': 0, '대졸': 1, '대학원 석사 이상': 2}).fillna(3)
    if 'P1B15' in df: df['P1B15'] = df['P1B15'].map({'고졸': 0, '대졸': 1, '대학원 석사 이상': 2}).fillna(3)

    p1_csv = os.path.join(args.outdir, "P1_processed.csv")
    df.to_csv(p1_csv, index=False)

    # ---------------------------
    # P2
    # ---------------------------
    df = data[[c for c in p2_cols if c in data.columns]].copy()
    df.replace({'예': 0, '아니오': 1, '아니요': 1, '모르겠다': 2}, inplace=True)
    if 'P2B27' in df: df['P2B27'] = df['P2B27'].map({'피운 적 없다': 0, '지금은 끊었다': 1, '요즘도 피운다': 2})
    if 'P2B32' in df: df['P2B32'] = df['P2B32'].map({'피운 적 없다': 0, '지금은 끊었다': 1, '요즘도 피운다': 2})
    if 'P2B38' in df: df['P2B38'] = df['P2B38'].map({'마신 적 없다': 0, '지금은 끊었다': 1, '요즘도 마신다': 2})
    if 'P2B45' in df: df['P2B45'] = df['P2B45'].map({'마신 적 없다': 0, '지금은 끊었다': 1, '요즘도 마신다': 2})

    cols = [c for c in [f'P2B{i}' for i in range(3, 22)] if c in df.columns]

    def norm(s):
        if pd.isna(s):
            return np.nan
        s = str(s).replace('\u00A0', ' ').strip()
        return ' '.join(s.split())

    for c in cols + (['P2B2'] if 'P2B2' in df.columns else []):
        df[c] = df[c].map(norm)

    def choose_value(row):
        vals = [row[c] for c in cols if pd.notna(row[c])]
        if not vals:
            return np.nan
        if '부모' in vals:
            return '부모'
        if '형제' in vals:
            return '형제'
        return vals[0]

    if 'P2B2' in df.columns:
        if is_categorical_dtype(df['P2B2']):
            df['P2B2'] = df['P2B2'].astype(object)

        mask = df['P2B2'] != '해당사항 없음'
        df.loc[mask, 'P2B2'] = df.loc[mask].apply(choose_value, axis=1)
        df['P2B2'] = df['P2B2'].map({'부모': 0, '형제': 1, '조부모': 2, '사촌': 3, '삼촌': 3, '이모': 3, '고모': 3, '해당사항 없음': 4}).fillna(4)

    if cols:
        df.drop(cols, axis=1, inplace=True, errors="ignore")

    p2_csv = os.path.join(args.outdir, "P2_processed.csv")
    df.to_csv(p2_csv, index=False)

    # ---------------------------
    # P3
    # ---------------------------
    df = data[[c for c in p3_cols if c in data.columns]].copy()
    df.replace({'예': 0, '아니오': 1, '아니요': 1, '모르겠다': 2}, inplace=True)
    if 'P3B2' in df:  df['P3B2'] = df['P3B2'].map({'의학적인 도움 없이 임신하였다': 0, '의학적인 도움을 받았다': 1})
    if 'P3B25' in df: df['P3B25'] = df['P3B25'].map({'해당사항 없음': 0, '있음': 1}).fillna(1)
    p3_csv = os.path.join(args.outdir, "P3_processed.csv")
    df.to_csv(p3_csv, index=False)

    # ---------------------------
    # P4
    # ---------------------------
    df = data[[c for c in p4_cols if c in data.columns]].copy()
    df.replace({'예': 0, '아니오': 1, '아니요': 1, '모르겠다': 2}, inplace=True)
    if 'P4B2' in df: df['P4B2'] = df['P4B2'].map({'해당사항 없음': 0}).fillna(1)
    p4_csv = os.path.join(args.outdir, "P4_processed.csv")
    df.to_csv(p4_csv, index=False)

    # ---------------------------
    # P5
    # ---------------------------
    df = data[[c for c in p5_cols if c in data.columns]].copy()
    df.replace({'예': 0, '아니오': 1, '아니요': 1, '모르겠다': 2, '모름': 2}, inplace=True)
    if 'P5B2' in df:
        df['P5B2'] = df['P5B2'].replace({
            '정상(임신 주수를 다 채우고 출생)': 0,
            '미숙아(~36주 6일 출생)': 1
        })
    if 'P5B5' in df:
        df['P5B5'] = df['P5B5'].replace({
            '제왕절개': 0,
            '질식분만(자연분만)': 1,
            '질식분만(유도분만)': 1,
            '질식분만(기계분만, 예-겸자분만/흡입분만)': 1
        })
    p5_csv = os.path.join(args.outdir, "P5_processed.csv")
    df.to_csv(p5_csv, index=False)

    # ---------------------------
    # 최종 병합
    # ---------------------------
    p1_final = pd.read_csv(p1_csv)
    p2_final = pd.read_csv(p2_csv)
    p3_final = pd.read_csv(p3_csv)
    p4_final = pd.read_csv(p4_csv)
    p5_final = pd.read_csv(p5_csv)

    final = p1_final.merge(p2_final, on='patient_id', how='inner') \
                    .merge(p3_final, on='patient_id', how='inner') \
                    .merge(p4_final, on='patient_id', how='inner') \
                    .merge(p5_final, on='patient_id', how='inner')

    final = final.drop_duplicates(subset='patient_id', keep='first')

    out_csv = os.path.join(args.outdir, "Demo_processed.csv")
    final.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv}")

if __name__ == "__main__":
    main()