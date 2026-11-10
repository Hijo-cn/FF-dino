# 把 8 个权重文件放在这个目录

## 必需(种模型,4 个)

```
v2_dual7_ventral_fold0.pth
v2_dual7_dorsal_fold0.pth
v2_dual7_lateral_fold0.pth
v2_dual7_mand_fold0.pth
```

## 可选(属模型,4 个)

```
v2_gen2_ventral_fold0.pth
v2_gen2_dorsal_fold0.pth
v2_gen2_lateral_fold0.pth
v2_gen2_mand_fold0.pth
```

缺属模型也能跑,属级会从种概率推导,准确率低约 0.4 个百分点。

每个约 1.2 GB,合计约 9.6 GB。
