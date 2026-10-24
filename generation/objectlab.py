from pathlib import Path
from shutil import copy2
import numpy as np
from cleanlab.object_detection.summary import visualize
from ultralytics import YOLO

CLASS_NAMES = {0: 'Hard Hat', 1: 'Safety Hook', 2: 'Worker'}

def load_model(run_dir):
    return YOLO(Path(run_dir) / 'weights' / 'best.pt')

def label_path(image_path):
    return Path(image_path).with_suffix('.txt')

class CleanLabTask:
    def __init__(self, class_names=CLASS_NAMES):
        self.class_names = class_names
        
    def show(self, idx, images, labels, pred_probs, scores):
        print(f'{images[idx]} | idx {idx} | label quality score: {scores[idx]:.4f}')
        visualize(str(images[idx]),
                label=labels[idx],
                prediction=pred_probs[idx],
                class_names=self.class_names,
                overlay=False,
                figsize=(20, 20))
        
    def visualize_ordered(self, images, labels, pred_probs, scores, limit=30, worst_first=True):
        order = np.argsort(scores)
        if not worst_first:
            order = order[::-1]
        for idx in order[:limit]:
            self.show(int(idx), images, labels, pred_probs, scores)
            
    def visualize_issue(self, k, images, label_issue_idx, labels, pred_probs, scores):
        self.show(int(label_issue_idx[k]), images, labels, pred_probs, scores)
        
    @staticmethod
    def create_remove_noise_label_dataset(images, scores, target_dir, threshold=0.6):
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        kept = 0
        for image, score in zip(images, scores):
            if score < threshold:
                continue
            copy2(image, target_dir)
            copy2(label_path(image), target_dir)
            kept += 1
        print(f'kept {kept}/{len(images)} images (score >= {threshold})')
        return kept
    
    @staticmethod
    def create_auto_label_dataset(images, scores, pred_results, target_dir, threshold=0.8, keep_empty=False):
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        relabeled = dropped = 0
        for image, score, result in zip(images, scores, pred_results):
            if score >= threshold:
                copy2(image, target_dir)
                copy2(label_path(image), target_dir)
                continue
            
            boxes = result[0].boxes
            if len(boxes) == 0 and not keep_empty:
                dropped += 1
                continue
            
            lines = [
                f'{int(c)} ' + ' '.join(f'{v:.6f}' for v in xywhn)
                for c, xywhn in zip(boxes.cls.tolist(), boxes.xywhn.tolist())
            ]
            (target_dir / label_path(image).name).write_text(''.join(f'{l}\n' for l in lines))
            copy2(image, target_dir)
            relabeled += 1
        print(f'relabeled {relabeled}, dropped {dropped}, total {len(images)} (threshold {threshold})')
        return relabeled, dropped
