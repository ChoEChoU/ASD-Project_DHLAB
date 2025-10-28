import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusterHead(nn.Module):
    def __init__(self, output_dim, num_prototypes, temperature=10):
        super().__init__()
        self.t = temperature
        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        self.normalize_prototypes()
        return self.prototypes(x) * self.t


class WTALModel(nn.Module):
    def __init__(self, config):
        super(WTALModel, self).__init__()
        self.len_feature = config.len_feature
        self.num_classes = config.num_classes
        self.num_clusters = config.num_clusters

        # video classification branch = feature encoder + snippet classifier
        self.cas_module_rgb = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=self.num_classes, kernel_size=1, padding=0),
        )
        # self.cas_module_flow = nn.Sequential(
        #     nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=512, out_channels=self.num_classes, kernel_size=1, padding=0),
        # )

        # attention branch
        # feature encoder
        self.base_module_rgb = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=1, padding=0),
            nn.ReLU(),
        )
        # self.base_module_flow = nn.Sequential(
        #     nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=1, padding=0),
        #     nn.ReLU(),
        # )
        # attention layer
        self.att_module_rgb = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0)
        # self.att_module_flow = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0)

        # cluster head
        # self.clu_head_flow = ClusterHead(512, self.num_clusters)
        self.clu_head_rgb = ClusterHead(512, self.num_clusters)

    def forward(self, inp):
        # print(inp.shape)
        # inp = inp.squeeze(-1).squeeze(-1) 
        inp = inp.permute(0, 2, 1)

        
        cas_rgb = self.cas_module_rgb(inp).permute(0, 2, 1)

        
        base_rgb = self.base_module_rgb(inp)

        
        att_rgb = torch.sigmoid(self.att_module_rgb(base_rgb))

        
        emb_rgb = F.normalize(base_rgb.permute(0, 2, 1), dim=-1)

        
        clu_rgb = self.clu_head_rgb(emb_rgb)

        if self.training:
            return  cas_rgb, att_rgb, clu_rgb, base_rgb
        else:
            return cas_rgb, att_rgb, clu_rgb
