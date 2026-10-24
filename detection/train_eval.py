import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_TRAIN = 'training_results'
PROJECT_TEST = 'test_results'

NO_AUG = dict(
    scale=0, fliplr=0, mosaic=0, erasing=0, crop_fraction=0, translate=0,
    close_mosaic=0, mixup=0, hsv_h=0, hsv_s=0, hsv_v=0, perspective=0, copy_paste=0,
)

EXPERIMENTS = {
    'table2': {
        'case_a': 'remove_noise.yaml',
        'case_b': 'noise_refine_1.yaml',
        'case_c': 'noise_refine_2.yaml',
    },
    'table4': {
        'realset_300': 'real300.yaml',
        'synthset_300': 'unity300.yaml',
        'ours_300': 'geo300.yaml',
    },
    'table5': {
        'synthset': 'unity.yaml',
        'ourmethod': 'geo.yaml',
    },
    'table6': {
        'realset_synthset': 'hybrid_unity.yaml',
        'realset_ourmethod': 'geo12_real.yaml',
    },
}

@dataclass
class YoloTask:
    data: str = 'real.yaml'
    mode: str = 'n'
    epochs: int = 100
    batch_size: int = 512
    seed: int = 0
    wo_aug: bool = True
    devices: tuple = (0, 1, 2, 3)

    @property
    def name(self) -> str:
        prefix = 'const' if self.wo_aug else 'const_aug'
        return f'{prefix}_{self.epochs}_{self.mode}_{self.seed}_{Path(self.data).stem}'

    @property
    def run_dir(self) -> Path:
        return Path(PROJECT_TRAIN) / self.name

    @property
    def best_weights(self) -> Path:
        return self.run_dir / 'weights' / 'best.pt'

    def is_finished(self) -> bool:
        csv = self.run_dir / 'results.csv'
        if not (self.best_weights.exists() and csv.exists()):
            return False
        with open(csv) as f:
            n_rows = sum(1 for _ in f) - 1  
        return n_rows >= self.epochs

    def train(self, skip_finished: bool = False):
        if skip_finished and self.is_finished():
            print(f'[skip] {self.name} already finished')
            return
        print(f'[train] {self.name} (yolov12{self.mode}.pt)')
        model = YOLO(f'yolov12{self.mode}.pt')
        params = dict(
            data=self.data,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=640,
            optimizer='SGD',
            lr0=0.001,
            momentum=0.8,
            device=list(self.devices),
            workers=8,
            seed=self.seed,
            project=PROJECT_TRAIN,
            name=self.name,
            exist_ok=True,   
            patience=0,     
        )
        if self.wo_aug:
            params.update(NO_AUG)
        model.train(**params)

    def test(self):
        print(f'[test] {self.best_weights}')
        model = YOLO(str(self.best_weights))
        metrics = model.val(
            data=self.data,
            device=self.devices[0],
            save_json=True,
            project=PROJECT_TEST,
            name=self.name,
            exist_ok=True,
            split='test',
        )
        print_overall(metrics)
        return metrics


def print_overall(metrics):
    b = metrics.box
    print('-' * 50)
    for label, v in [('Precision', b.mp), ('Recall', b.mr), ('mAP50', b.map50), ('mAP50-95', b.map)]:
        print(f'{label}: {v:.4f} ({v * 100:.2f}%)')
    print('-' * 50)


def mean_std(metrics_list, names=None, ddof=0):
    names = names or metrics_list[0].names
    keys = ['Precision', 'Recall', 'F1', 'AP50']

    per_class = defaultdict(lambda: {k: [] for k in keys})
    for m in metrics_list:
        b = m.box
        for i, c in enumerate(b.ap_class_index):
            c = int(c)
            per_class[c]['Precision'].append(float(b.p[i]))
            per_class[c]['Recall'].append(float(b.r[i]))
            per_class[c]['F1'].append(float(b.f1[i]))
            per_class[c]['AP50'].append(float(b.ap50[i]))

    n_runs = len(metrics_list)
    summary = {}
    for k in keys:
        print(f'[{k}]')
        summary[k] = {}
        for c in sorted(per_class):
            vals = per_class[c][k]
            mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=ddof))
            note = '' if len(vals) == n_runs else f'  (n={len(vals)}/{n_runs})'
            print(f"  Class '{names[c]}': {mu:.4f} ± {sd:.4f} ({mu * 100:.2f}% ± {sd * 100:.2f}%){note}")
            summary[k][names[c]] = dict(mean=mu, std=sd, n=len(vals), values=vals)
        print('-' * 50)
    return summary


def main():
    seeds = [0, 42, 314, 1337, 2026]
    mode = 'm'
    batch_size = 256
    epochs = 100
    skip_finished = False

    # results[table][label] = [metrics per seed]
    results = {t: {label: [] for label in exps} for t, exps in EXPERIMENTS.items()}

    for seed in seeds:
        for table, exps in EXPERIMENTS.items():
            for label, data in exps.items():
                task = YoloTask(data=data, mode=mode, epochs=epochs,
                                batch_size=batch_size, seed=seed, wo_aug=True)
                task.train(skip_finished=skip_finished)
                results[table][label].append(task.test())

    summary = {}
    for table, exps in results.items():
        summary[table] = {}
        for label, metrics_list in exps.items():
            print(f'\n===== {table} / {label} =====')
            summary[table][label] = mean_std(metrics_list)

    out = Path(PROJECT_TEST) / f'summary_{mode}_e{epochs}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'\nSummary saved to {out}')


if __name__ == '__main__':
    main()