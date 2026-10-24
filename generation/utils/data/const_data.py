import random
import torch
from mmdet.datasets.builder import DATASETS
from utils.data.nuimage import NuImageDataset

@DATASETS.register_module()
class constData(NuImageDataset):
    CLASSES = ['HardHat', 'Hook', 'Person']
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            
            # pad_shape is the actual input shape even without padding
            bboxes = data['gt_bboxes'].data
            labels = [self.CLASSES[each].replace('-', ' ') for each in data['gt_labels'].data]
            
            pad_shape = data['img_metas'].data['pad_shape']
            img_shape = torch.tensor([pad_shape[1], pad_shape[0], pad_shape[1], pad_shape[0]])
            bboxes /= img_shape
            
            # random shuffle bbox annotations
            index = list(range(len(labels)))
            random.shuffle(index)
            index = index[:18] # 3+4*18+2=77
            
            # generate bbox mask and text prompt
            # constant: background -> 0, foreground -> self.foreground_loss_weight
            # area:     background -> 0, foreground -> 1 / area ^ self.foreground_loss_weight (for area, smaller weight, larger variance with respect to areas)
            objs = []
            bbox_mask = torch.zeros(self.FEAT_SIZE).float() # [H, W]
            for each in index:
                label = labels[each]
                bbox = bboxes[each]
                
                # generate bbox mask
                FEAT_SIZE = torch.tensor([self.FEAT_SIZE[1], self.FEAT_SIZE[0], self.FEAT_SIZE[1], self.FEAT_SIZE[0]])
                coord = torch.round(bbox * FEAT_SIZE).int().tolist()
                
                bbox_mask[coord[1]: coord[3], coord[0]: coord[2]] = self.foreground_loss_weight * 1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)), 0.2) if self.foreground_loss_mode == 'constant' else \
                                                                    1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)), self.foreground_loss_weight)
                # generate text prompt
                bbox = self.token_pair_from_bbox(bbox.tolist())
                objs.append(' '.join([label, bbox]))
            
            bbox_mask[bbox_mask == 0] = 1 * 1 / torch.pow(torch.tensor(self.FEAT_SIZE[0] * self.FEAT_SIZE[1]), 0.2) if self.foreground_loss_mode == 'constant' else 1 / torch.pow(torch.tensor(self.FEAT_SIZE[0] * self.FEAT_SIZE[1]), self.foreground_loss_weight)
            bbox_mask = bbox_mask / torch.sum(bbox_mask) * self.FEAT_SIZE[0] * self.FEAT_SIZE[1] if self.foreground_loss_norm else bbox_mask
            
            if self.uncond_prob > 0:
                text = 'An image with ' + ' '.join(objs) if random.random() > self.uncond_prob else ""
            else:
                text = 'An image with ' + ' '.join(objs)
            example = {}
            example["pixel_values"] = data['img'].data
            example["text"] = text
            if self.foreground_loss_mode is not None:
                example["bbox_mask"] = bbox_mask
            return example

@DATASETS.register_module()
class constDataSim(constData):
    CLASSES = ['hook', 'person']
