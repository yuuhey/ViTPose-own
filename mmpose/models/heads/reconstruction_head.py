import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmpose.models.builder import HEADS, build_loss

@HEADS.register_module()
class ReconstructionHead(BaseModule):
    def __init__(self, in_channels, hidden_dim, out_channels, num_layers, loss_cfg, init_cfg=None):
            super(ReconstructionHead, self).__init__(init_cfg)
            layers = [nn.Flatten()]
            layers.append(nn.Linear(in_channels, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, out_channels))
            self.fc = nn.Sequential(*layers)
            
            self.loss_reconstruction = build_loss(loss_cfg)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x).view(x.size(0), 3, 256, 192)

    def get_loss(self, preds, targets):
        # Ensure the targets and preds are of the same shape
        # print(f"Preds shape: {preds.shape}, Targets shape: {targets.shape}")
        return self.loss_reconstruction(preds, targets)

    def init_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m, 1)
