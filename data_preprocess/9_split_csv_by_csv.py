import os
import json
import pandas as pd
from collections import defaultdict

# 경로 설정
task = "I"
month = "18"
all_csv_path = f'./data/1017_splits/task_{task}/cell_month_{month}/input_label.csv'  # 전체 VideoPath, Label 포함 CSV
split_root = './data/1017_splits/no_matching_fold'      # fold_0, fold_1 ... 폴더가 있는 디렉토리
output_dir = f'./data/1017_splits/task_{task}/cell_month_{month}/input_data_no_matching'
label_map = {0: 'Normal', 1: 'Others'}

os.makedirs(output_dir, exist_ok=True)

# 전체 데이터 로딩
df_all = pd.read_csv(all_csv_path)
df_all['VideoID'] = df_all['VideoPath'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
df_all['patient_id'] = df_all['VideoPath'].apply(lambda x: x.split('/')[0])
df_all['Label'] = df_all['Label'].astype(int)

stats_records = []
all_test_patients = []

# 각 fold 순회
for fold_name in sorted(os.listdir(split_root)):
    fold_path = os.path.join(split_root, fold_name)
    if not os.path.isdir(fold_path):
        continue
    print(fold_name.split('_'))
    fold_idx = int(fold_name.split('_')[-1])
    print(f"🔁 Fold {fold_idx}")

    df_split = {}
    for split in ['train', 'valid', 'test']:
        csv_path = os.path.join(fold_path, f'{split}.csv')
        df_split[split] = pd.read_csv(csv_path)
        df_split[split]['Split'] = split

    test_subjects = set(df_split['test']['patient_id'])
    all_test_patients.append(test_subjects)

    df_combined = []
    for split in ['train', 'valid', 'test']:
        subjects = df_split[split]['patient_id']
        df_sub = df_all[df_all['patient_id'].isin(subjects)].copy()
        df_sub['Split'] = split
        df_combined.append(df_sub)

    df_all_split = pd.concat(df_combined, ignore_index=True)

    # JSON 생성
    database = {}
    for _, row in df_all_split.iterrows():
        database[row['VideoID']] = {
            "subset": row['Split'],
            "annotations": [{"segment": ["0", "0"], "label": label_map[row['Label']]}]
        }

    with open(os.path.join(output_dir, f'gt_fold{fold_idx}.json'), 'w') as f:
        json.dump({'database': database}, f, indent=4)

    # split txt 저장
    for split in ['train', 'valid', 'test']:
        txt_path = os.path.join(output_dir, f'split_{split}_fold_{fold_idx}.txt')
        video_ids = df_all_split[df_all_split['Split'] == split]['VideoID'].tolist()
        with open(txt_path, 'w') as f:
            f.write('\n'.join(video_ids))

    # 통계 저장
    for split in ['train', 'valid', 'test']:
        df_part = df_all_split[df_all_split['Split'] == split]
        sample_total = len(df_part)
        sample_normal = (df_part['Label'] == 0).sum()
        sample_others = (df_part['Label'] == 1).sum()

        patient_group = df_part.groupby('patient_id')['Label'].first()
        patient_total = len(patient_group)
        patient_normal = (patient_group == 0).sum()
        patient_others = (patient_group == 1).sum()

        stats_records.append({
            "fold": fold_idx,
            "split": split,
            "sample_total": sample_total,
            "sample_normal": sample_normal,
            "sample_others": sample_others,
            "patient_total": patient_total,
            "patient_normal": patient_normal,
            "patient_others": patient_others
        })

        print(f"\n📊 [Fold {fold_idx} - {split.upper()}]")
        print(f"   ▶ Samples : {sample_total} | Normal: {sample_normal} ({sample_normal/sample_total:.1%}) | Others: {sample_others} ({sample_others/sample_total:.1%})")
        print(f"   ▶ Patients: {patient_total} | Normal: {patient_normal} ({patient_normal/patient_total:.1%}) | Others: {patient_others} ({patient_others/patient_total:.1%})")

# 통계 CSV 저장
stats_df = pd.DataFrame(stats_records)
stats_df.to_csv(os.path.join(output_dir, 'fold_split_statistics.csv'), index=False)

# test fold 간 중복 확인
test_union = set()
for i, test_set in enumerate(all_test_patients):
    overlap = test_union & test_set
    if overlap:
        raise ValueError(f"❌ Fold {i}의 test set과 이전 fold와 중복됨: {overlap}")
    test_union.update(test_set)

print("✅ 모든 fold 간 test 환자 ID 중복 없음!")