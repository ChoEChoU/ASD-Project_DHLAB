import os
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from utils import misc_utils
from run_eval import evaluate
from torchmetrics.classification import MulticlassROC, MulticlassConfusionMatrix, MulticlassAUROC

def load_weight(model_file, net):
    if model_file is not None:
        print("loading from file: ", model_file)
        net.load_state_dict(torch.load(model_file), strict=False)

def inference(net, config, test_loader, cls_gt, model_file=None):
    np.set_printoptions(formatter={'float_kind': "{:.4f}".format})

    with torch.no_grad():
        net.eval()
        load_weight(model_file, net)

        final_res = {
            'version': 'VERSION 1.3',
            'results': {},
            'external_data': {'used': True, 'details': 'Features from I3D Network'}
        }

        label_df = pd.read_csv(config.patient_csv)
        video_to_patient = {
            path.split('/')[1]: path.split('/')[0] for path in label_df['VideoPath']
        }

        patient_score_dict = defaultdict(list)
        patient_label_dict = {}

        for _data, _label, _, vid_name, vid_num_seg in test_loader:
            _data = _data.cuda()
            _label = _label.cuda()

            video_id = vid_name[0]
            patient_id = video_to_patient.get(video_id, "UNKNOWN")
            # print(video_id)
            # print(patient_id)
            if patient_id == "UNKNOWN":
                print(f"⚠️ patient_id not found for {video_id}, skipping.")
                continue

            cas_rgb, att_rgb, cls_rgb = net(_data)

            combined_cas = misc_utils.instance_selection_function(
                cas_rgb.softmax(-1),
                att_rgb.permute(0, 2, 1),
                (cls_rgb.softmax(-1) @ cls_gt)[..., :1]
            )

            _, topk_indices = torch.topk(combined_cas, config.num_segments // 8, dim=1)
            cas_top_rgb = torch.mean(torch.gather(cas_rgb, 1, topk_indices), dim=1)
            score_np = cas_top_rgb.softmax(1)[0].cpu().numpy()

            patient_score_dict[patient_id].append(score_np)
            patient_label_dict[patient_id] = _label.argmax().item()

            pred = np.array([np.argmax(score_np)])
            score_supp = cas_top_rgb.softmax(1)

            if len(pred) != 0:
                cas_pred = combined_cas[0].cpu().numpy()[:, pred]
                cas_pred = np.reshape(cas_pred, (config.num_segments, -1, 1))
                cas_pred = misc_utils.upgrade_resolution(cas_pred, config.scale)

                proposal_dict = {}
                for t in range(len(config.act_thresh)):
                    cas_temp = cas_pred.copy()
                    zero_location = np.where(cas_temp[:, :, 0] < config.act_thresh[t])
                    cas_temp[zero_location] = 0

                    seg_list = []
                    for c in range(len(pred)):
                        pos = np.where(cas_temp[:, c, 0] > 0)
                        seg_list.append(pos)

                    proposals = misc_utils.get_proposal_oic(
                        seg_list, cas_pred.copy(),
                        score_supp[0, :].cpu().data.numpy(),
                        pred, config.scale, vid_num_seg[0].cpu().item(),
                        config.feature_fps, config.num_segments, config.gamma
                    )

                    for j in range(len(proposals)):
                        if not proposals[j]:
                            continue
                        class_id = proposals[j][0][0]
                        if class_id not in proposal_dict:
                            proposal_dict[class_id] = []
                        proposal_dict[class_id] += proposals[j]

                final_proposals = []
                for class_id in proposal_dict:
                    final_proposals.append(misc_utils.basnet_nms(
                        proposal_dict[class_id],
                        config.nms_thresh,
                        config.soft_nms,
                        config.nms_alpha
                    ))

                final_res['results'][video_id] = misc_utils.result2json(final_proposals, config.stage)

        # ⬇️ Patient-level aggregation
        final_vid_list = list(patient_score_dict.keys())
        final_score_list = []
        final_pred_list = []
        final_label_list = []

        for pid in final_vid_list:
            avg_score = np.mean(np.stack(patient_score_dict[pid]), axis=0)
            pred = np.argmax(avg_score)

            final_score_list.append(avg_score)
            final_pred_list.append(pred)
            final_label_list.append(patient_label_dict[pid])
            final_res['results'][pid] = []

        correct = sum(p == l for p, l in zip(final_pred_list, final_label_list))
        test_acc = correct / len(final_label_list)

        # ⬇️ 확률 포함한 결과 저장
        score_cols = [f'prob_class_{i}' for i in range(len(final_score_list[0]))]
        score_df = pd.DataFrame(final_score_list, columns=score_cols)
        score_df.insert(0, 'patient_id', final_vid_list)
        score_df['prediction'] = final_pred_list
        score_df['label'] = final_label_list
        score_df.to_csv(f'{config.model_path}/{test_loader.dataset.mode}_result.csv', index=False)

        # ⬇️ JSON 저장
        json_path = os.path.join(config.model_path, 'temp_result.json')
        with open(json_path, 'w') as f:
            json.dump(final_res, f)

        # ⬇️ mAP 계산
        mean_ap, _, ap = evaluate(
            config.gt_path,
            json_path,
            None,
            tiou_thresholds=np.linspace(0.1, 0.7, 7),
            plot=False,
            subset='valid',
            verbose=config.verbose
        )

        # ⬇️ Metrics (only in test mode)
        if test_loader.dataset.mode == 'test':
            num_classes = len(set(final_label_list))
            scores_tensor = torch.tensor(final_score_list)
            labels_tensor = torch.tensor(final_label_list)

            auroc_metric = MulticlassAUROC(num_classes=num_classes, average='macro')
            cf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
            roc_curve_metric = MulticlassROC(num_classes=num_classes)

            auroc = auroc_metric(scores_tensor, labels_tensor)

            cf_matrix.update(scores_tensor, labels_tensor)
            cfm_fig, _ = cf_matrix.plot(cmap='Blues')
            cfm_fig.savefig(f'{config.model_path}/confusion_matrix.png')

            roc_curve_metric.update(scores_tensor, labels_tensor)
            roc_fig, _ = roc_curve_metric.plot(score=True)
            roc_fig.savefig(f'{config.model_path}/roc_curve.png')

            with open(os.path.join(config.model_path, 'best_result.json'), 'w') as f:
                json.dump(final_res, f, indent=4)

            return mean_ap, test_acc, final_res, ap, auroc
        else:
            return mean_ap, test_acc, final_res, ap