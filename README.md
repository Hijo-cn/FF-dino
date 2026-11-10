# FF2 · 鼹鼠科标本图像识别

输入标本照片,输出**物种预测**和**2048 维特征向量**。

## 安装

```bash
pip install -r requirements.txt
```

## 放权重

8 个 `.pth`,约 9.6 GB,清单见 `weights/README.md`。

**能访问实验室服务器的话**,不用下载,直接改 `config.py`:

```python
WEIGHTS_DIR = "/home/yanhaiqiao/projects/ff2/weights"
```

**访问不了**,就把 `.pth` 下载后放进本目录的 `weights/`。

## 准备输入

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

**4 个视角都要给,少给会明显掉准确率**(4 视角 97.7%,3 视角 80%,2 视角 50%)。

## 运行

```bash
python infer.py --input list.csv
```

得到两个文件:

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

## 改配置

`config.py` 里通常只需要改:

```python
WEIGHTS_DIR = "weights"   # 权重目录
BATCH_SIZE = 16           # 显存不够就调小
```

## 准确率

5 折交叉验证平均:扁平 91.8%,属 98.7%,**层级 92.6%**。

自测:

```bash
export FF2_DATASET_DIR=/path/to/dataset   # 含 metadata.csv
python benchmark.py --fold 0
```

## 速度

约 25 张/秒(RTX 3090),默认参数下 1000 张图约 1 分钟。
