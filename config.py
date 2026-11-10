# -*- coding: utf-8 -*-
"""FF2 推理配置 —— 只需要改这一处路径就能跑"""

# ========== 必须改的 ==========

# 权重目录(里面应该有 v2_dual7_<view>_fold0.pth 和 v2_gen2_<view>_fold0.pth)
WEIGHTS_DIR = "weights"

# 类别清单(51 个种 + 18 个属的顺序)
# 训练时来自 splits_v2.json 的 class_names,这里单独存一份,避免依赖 splits 文件
CLASS_NAMES_FILE = "class_names.txt"


# ========== 一般不用改 ==========

BACKBONE = "dinov3l"          # dinov3l 或 dinov2l
IMG_SIZE = 592

# 归一化参数(与训练时 timm 的 resolve_data_config 一致,一般不用改)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

TAG_SPECIES = "v2_dual7"      # 种模型权重前缀
TAG_GENUS = "v2_gen2"         # 属模型权重前缀

# 使用的视角(顺序无所谓)
VIEWS = ["ventral", "dorsal", "lateral", "mand"]

# 冲突门控:对「与其他视角分歧大」的类做降权
USE_GATE = True
GATE_GAMMA = 20.0

# 层级判定:先用属模型定属,再在该属内取种
USE_HIERARCHICAL = True

# 批大小(显存不够就调小)
BATCH_SIZE = 16
NUM_WORKERS = 4

# 默认视角推断规则:文件名里含这些片段就认为是该视角
VIEW_PATTERNS = {
    "ventral": ["#s#v"],
    "dorsal": ["#s#d"],
    "lateral": ["#s#l"],
    "mand": ["#m#l"],
}
