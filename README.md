# Experiments Repo

Repo này hiện được tách theo vai trò để dễ mở rộng cho các thí nghiệm mới:

- `experiments/`: script huấn luyện hoặc probing theo từng thí nghiệm.
- `runners/`: runner tạo metric/TSV từ checkpoint.
- `plots/`: script vẽ figure từ TSV hoặc artifact đã có.
- `analysis/`: script phân tích, tổng hợp, notebook/report generation.
- `scripts/`: workflow shell để chạy end-to-end.
- `sit_metrics/`: package metric dùng lại giữa các thí nghiệm.
- `outputs/`: artifact sinh ra từ các lần chạy.
- `SiT/`, `REPA/`, `HASTE/`: code upstream/vendorized.

Quy ước cho thí nghiệm mới:

- Nếu là một experiment mới, đặt entrypoint trong `experiments/`.
- Nếu cần runner riêng để sinh metric, đặt trong `runners/`.
- Nếu chỉ đọc TSV/PNG để phân tích, đặt trong `analysis/` hoặc `plots/`.
- Giữ artifact trong `outputs/<ten_thi_nghiem>/`.

Ví dụ:

```bash
python runners/run_sit_imagenet_metrics.py --help
python plots/plot_imagenet_metrics_compare.py --help
python experiments/exp1_frozen_ffn_probe.py --help
```

## Kaggle miniImageNet + SiT-XL/2

Pipeline Kaggle mới nằm trong `kaggle_sit_protocol/` và được điều phối bởi 3 notebook:

- `notebooks/00_bootstrap_and_manifest.ipynb`
- `notebooks/01_task0_cache.ipynb`
- `notebooks/02_tasks_1_to_10.ipynb`

Các notebook này thực hiện:

- auto-discover miniImageNet dưới `/kaggle/input`
- build manifest `500 main / 100 control / 16 preview`
- encode latent + fixed noise cache
- dựng `mean-common` và `tsvd-common (K=16,32,64)` mà không dump full activations
- chạy Task 1-10 kèm `Task 4B spatial norm` từ cache gọn để giữ output an toàn dưới giới hạn Kaggle
- xuất PCA-RGB panel cho `raw/common/residual` cả trước và sau spatial norm trong `Task 8`

Package entrypoints:

```python
from kaggle_sit_protocol import ProtocolConfig, run_bootstrap_stage, run_task0_stage, run_analysis_stage

cfg = ProtocolConfig.from_kaggle_defaults()
run_bootstrap_stage(cfg)
run_task0_stage(cfg)
run_analysis_stage(cfg)
```
