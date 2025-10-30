import os
import json
import numpy as np
import torch
import random
from torch.utils.data import Dataset

class_dict = {0: 'Normal', 1: 'Others'}
# class_dict_stage = {1: {0: 'Normal', 1: 'Abnormal'},
#               2: {0: 'HR', 1: 'ASD'}}
# class_dict_stage = {1: {0: 'ASD', 1: 'Non-ASD'},
#               2: {0: 'Normal', 1: 'HR'}}


class AIAI_Feature(Dataset):
    def __init__(self, data_path, mode, modal, feature_fps, num_segments, len_feature, sampling, fold, stage, seed=-1,
                 supervision='weak'):
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            # noinspection PyUnresolvedReferences
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            # noinspection PyUnresolvedReferences
            torch.backends.cudnn.deterministic = True
            # noinspection PyUnresolvedReferences
            torch.backends.cudnn.benchmark = False
        self.fold = fold
        self.mode = mode
        self.modal = modal
        self.feature_fps = feature_fps
        self.num_segments = num_segments
        self.len_feature = len_feature

        if self.modal == 'all':
            self.feature_path = []
            for _modal in ['rgb', 'flow']:
                self.feature_path.append(os.path.join(data_path, 'features', self.mode, _modal))
        else:
            self.feature_path = os.path.join(data_path, 'features', self.modal)

        split_path = os.path.join(data_path, 'split_{}_fold_{}.txt'.format(self.mode, self.fold))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.strip())
        split_file.close()

        anno_path = os.path.join(data_path, f'gt_fold{self.fold}.json')
        anno_file = open(anno_path, 'r')
        self.anno = json.load(anno_file)
        anno_file.close()
        if stage:
            self.class_name_to_idx = dict((v, k) for k, v in class_dict_stage[stage].items())
        else:
            self.class_name_to_idx = dict((v, k) for k, v in class_dict.items())
        self.num_classes = len(self.class_name_to_idx.keys())

        self.supervision = supervision
        self.sampling = sampling

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, vid_num_seg, sample_idx = self.get_data(index)
        label, temp_anno = self.get_label(index, vid_num_seg, sample_idx)

        return data, label, temp_anno, self.vid_list[index], vid_num_seg

    def get_data(self, index):
        vid_name = self.vid_list[index]

        vid_num_seg = 0

        if self.modal == 'all':
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                               vid_name + '.npy')).astype(np.float32)
            flow_feature = np.load(os.path.join(self.feature_path[1],
                                                vid_name + '.npy')).astype(np.float32)

            vid_num_seg = rgb_feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(rgb_feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(rgb_feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            rgb_feature = rgb_feature[sample_idx]
            flow_feature = flow_feature[sample_idx]

            feature = np.concatenate((rgb_feature, flow_feature), axis=1)
        else:
            feature = np.load(os.path.join(self.feature_path,
                                           vid_name + '.npy')).astype(np.float32)

            vid_num_seg = feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            feature = feature[sample_idx]

        return torch.from_numpy(feature), vid_num_seg, sample_idx

    def get_label(self, index, vid_num_seg, sample_idx):
        vid_name = self.vid_list[index]
        anno_list = self.anno['database'][vid_name]['annotations']
        label = np.zeros([self.num_classes], dtype=np.float32)

        classwise_anno = [[]] * self.num_classes

        for _anno in anno_list:
            label[self.class_name_to_idx[_anno['label']]] = 1
            classwise_anno[self.class_name_to_idx[_anno['label']]].append(_anno)

        if self.supervision == 'weak':
            return label, torch.Tensor(0)
        else:
            raise ValueError('Supervision argument must be "weak" in this experiment.')

    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)
