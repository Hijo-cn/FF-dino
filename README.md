## 安装

```bash
pip install -r requirements.txt
```

## 权重

改 `config.py` 指向权重目录:

```python
WEIGHTS_DIR = "/home/yanhaiqiao/projects/ff2/weights"
```

## 输入

一个 CSV,三列(参考 `example_list.csv`):

```csv
image_path,specimen,view
/data/A001_ventral.JPG,A001,ventral
/data/A001_lateral.JPG,A001,lateral
```

- `image_path`:图片路径
- `specimen`:标本编号,同一编号的多张图会合并成一个标本
- `view`:`ventral` / `dorsal` / `lateral` / `mand`。
  留空则从文件名推断(含 `#s#v`→ventral,`#s#d`→dorsal,`#s#l`→lateral,`#m#l`→mand)

## 运行

```bash
python infer.py --input list.csv
```

## 输出

**`results.csv`**

| 列 | 含义 |
|---|---|
| `specimen` | 标本编号 |
| `final_pred` | 最终判定 |
| `pred_genus` | 属 |
| `prob` | 概率 |
| `n_views` | 用到的视角数 |
| `top3` | 前 3 候选 |

**`results_emb.npz`**

```python
import numpy as np
d = np.load("results_emb.npz")
d["specimen"]    # (N,)      标本编号
d["embedding"]   # (N, 2048) 特征向量
```

## 配置

`config.py`:

```python
WEIGHTS_DIR = "/home/yanhaiqiao/projects/ff2/weights"
BATCH_SIZE = 16           # 显存不够就调小
```

## 准确率

5 折交叉验证平均:扁平 91.8%,属 98.7%,层级 92.6%。

速度约 25 张/秒(RTX 3090)。
