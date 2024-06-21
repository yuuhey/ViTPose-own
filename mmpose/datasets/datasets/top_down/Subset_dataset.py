from mmpose.datasets.builder import DATASETS
from .topdown_coco_dataset import TopDownCocoDataset

@DATASETS.register_module()
class CustomSubsetDataset(TopDownCocoDataset):
    def __init__(self, ann_file, img_prefix, data_cfg, pipeline, subset=None, **kwargs):
        super().__init__(ann_file, img_prefix, data_cfg, pipeline, **kwargs)
        
        if subset:
            num_samples = int(len(self.img_ids) * subset)
            self.img_ids = self.img_ids[:num_samples]
            self.db = self.db[:num_samples]