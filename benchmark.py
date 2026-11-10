#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
交付版准确率基准

只有某一折的「测试集」是模型没见过的,所以基准必须限定在该折的测试标本上。
(fold0 的模型在 fold0 的 train 上训练过,拿它去测 train 会虚高)

用法:
    python benchmark.py --fold 0                 # 造 CSV → 调 infer.py → 比对
    python benchmark.py --fold 0 --reuse         # 复用上次结果,只重新比对
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

TAG2VIEW = {
    "skull_ventral": "ventral", "skull_dorsal": "dorsal",
    "skull_lateral": "lateral", "mandible_lateral": "mand",
}


def build_csv(splits, fold, dataset, out_csv):
    sp = json.load(open(splits, encoding="utf-8"))
    test_ids = {sp["specimens"][i]["id"] for i in sp["folds"][fold]["test"]}
    truth = {s["id"]: s["species"] for s in sp["specimens"]}
    genus = {s["id"]: s["genus"] for s in sp["specimens"]}

    imgs = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(dataset, "metadata.csv"),
                                 encoding="utf-8-sig", newline="")):
        if r["specimen"] not in test_ids:
            continue
        v = TAG2VIEW.get(r["image_type"])
        if v:
            imgs[r["specimen"]][v] = os.path.join(dataset, r["image_path"])

    n_img = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "specimen", "view"])
        for s in sorted(imgs):
            for v, p in sorted(imgs[s].items()):
                w.writerow([p, s, v])
                n_img += 1

    with open(out_csv.replace(".csv", "_truth.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["specimen", "species", "genus"])
        for s in sorted(imgs):
            w.writerow([s, truth[s], genus[s]])

    print(f"fold{fold} 测试集: {len(imgs)} 个标本, {n_img} 张图 → {out_csv}")
    return out_csv


def compare(res_csv, truth_csv):
    truth = {}
    for r in csv.DictReader(open(truth_csv, encoding="utf-8-sig", newline="")):
        truth[r["specimen"]] = (r["species"], r["genus"])

    rows = []
    for r in csv.DictReader(open(res_csv, encoding="utf-8-sig", newline="")):
        if r["specimen"] in truth:
            ts, tg = truth[r["specimen"]]
            rows.append({
                "specimen": r["specimen"], "true": ts, "true_g": tg,
                "final": r["final_pred"], "species": r["pred_species"],
                "genus": r["pred_genus"], "n_views": int(r["n_views"]),
                "top3": r["top3"],
            })

    n = len(rows)
    if n == 0:
        print("没有可比对的行")
        return None
    ok_f = sum(r["final"] == r["true"] for r in rows)
    ok_s = sum(r["species"] == r["true"] for r in rows)
    ok_g = sum(r["genus"] == r["true_g"] for r in rows)

    print(f"\n{'='*62}")
    print(f"标本数 {n}")
    print(f"  扁平（纯种概率 argmax）  {ok_s/n*100:6.2f}%   ({ok_s}/{n})")
    print(f"  属级                    {ok_g/n*100:6.2f}%   ({ok_g}/{n})")
    print(f"  层级（先属后种，交付口径）{ok_f/n*100:6.2f}%   ({ok_f}/{n})")
    print(f"{'='*62}")

    # 按视角数分组
    print("\n按可用视角数:")
    for k in sorted({r["n_views"] for r in rows}):
        sub = [r for r in rows if r["n_views"] == k]
        acc = sum(r["final"] == r["true"] for r in sub) / len(sub) * 100
        print(f"  {k} 视角  n={len(sub):3d}   {acc:6.2f}%")

    # 逐属
    print("\n逐属（层级口径）:")
    gs = defaultdict(list)
    for r in rows:
        gs[r["true_g"]].append(r)
    for g in sorted(gs, key=lambda x: -len(gs[x])):
        sub = gs[g]
        acc = sum(r["final"] == r["true"] for r in sub) / len(sub) * 100
        print(f"  {g:16s} n={len(sub):3d}  {acc:6.1f}%")

    # 错误明细
    errs = [r for r in rows if r["final"] != r["true"]]
    if errs:
        print(f"\n错判明细（{len(errs)} 个）:")
        for r in errs[:25]:
            print(f"  {r['specimen']:14s} 真 {r['true']:28s} → {r['final']:28s} "
                  f"({r['n_views']}视角)")
        if len(errs) > 25:
            print(f"  ... 其余 {len(errs)-25} 个")

    return {"n": n, "flat": ok_s / n * 100, "genus": ok_g / n * 100,
            "hier": ok_f / n * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", default="",
                    help="多折一起跑,如 0,1,2,3,4;给出时会覆盖 --fold 并输出均值")
    ap.add_argument("--splits", default="../splits/splits_v2.json")
    ap.add_argument("--dataset", default=os.environ.get("FF2_DATASET_DIR", ""))
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--workdir", default="/tmp/ff2_bench")
    ap.add_argument("--ckpt-fold", type=int, default=-1,
                    help="用第几折的权重;默认与 --fold 相同(正确的交叉验证口径)")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    if not args.dataset:
        raise SystemExit("需要 export FF2_DATASET_DIR 或传 --dataset")
    folds = ([int(x) for x in args.folds.split(",") if x.strip()]
             if args.folds.strip() else [args.fold])
    os.makedirs(args.workdir, exist_ok=True)

    res = []
    for fold in folds:
        ck = fold if args.ckpt_fold < 0 else args.ckpt_fold
        csvp = os.path.join(args.workdir, f"fold{fold}.csv")
        outp = os.path.join(args.workdir, f"fold{fold}_res")
        print(f"\n{'#'*62}\n# fold {fold}   (权重 fold{ck})\n{'#'*62}")
        if not args.reuse:
            build_csv(args.splits, fold, args.dataset, csvp)
            cmd = [sys.executable, "infer.py", "--input", csvp,
                   "--output", outp, "--weights", args.weights,
                   "--ckpt-fold", str(ck)]
            print("\n运行:", " ".join(cmd), "\n")
            r = subprocess.run(cmd)
            if r.returncode != 0:
                raise SystemExit("infer.py 失败")
        st = compare(outp + ".csv", csvp.replace(".csv", "_truth.csv"))
        if st:
            st["fold"] = fold
            res.append(st)

    if len(res) > 1:
        print(f"\n{'='*62}")
        print(f"{'fold':>5} {'n':>5} {'扁平':>8} {'属':>8} {'层级':>8}")
        for s in res:
            print(f"{s['fold']:>5} {s['n']:>5} {s['flat']:>7.2f}% "
                  f"{s['genus']:>7.2f}% {s['hier']:>7.2f}%")
        n = sum(s["n"] for s in res)
        print(f"{'-'*62}")
        print(f"{'合计':>5} {n:>5} "
              f"{sum(s['flat']*s['n'] for s in res)/n:>7.2f}% "
              f"{sum(s['genus']*s['n'] for s in res)/n:>7.2f}% "
              f"{sum(s['hier']*s['n'] for s in res)/n:>7.2f}%")
        print(f"{'='*62}")


if __name__ == "__main__":
    main()
