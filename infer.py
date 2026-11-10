#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
FF2 标本识别推理

输入:一个 CSV,每行一张图
    image_path,specimen[,view]
    /data/a.JPG,SPEC001,ventral
    /data/b.JPG,SPEC001,lateral
  view 列可省略,此时从文件名推断(#s#v → ventral, #s#d → dorsal,
  #s#l → lateral, #m#l → mand)。

输出:
    <out>.csv      每标本一行:预测种 / 属 / 层级判定 / 概率 / top3 / 视角数
    <out>_emb.npz  specimen (N,) 与 embedding (N, 2048)

用法:
    python infer.py --input list.csv --output results
    python infer.py --input list.csv --output results --per-view-emb
"""
import argparse
import csv
import os
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config as C
from ff2_model import DualHeadNet, build_backbone

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x


class ImgDS(Dataset):
    def __init__(self, paths, size, mean, std):
        self.paths = paths
        self.size = size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        w, h = im.size
        s = self.size / min(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        im = im.resize((nw, nh), Image.BICUBIC)
        if nw != self.size or nh != self.size:
            l, t = (nw - self.size) // 2, (nh - self.size) // 2
            im = im.crop((l, t, l + self.size, t + self.size))
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
        return (x - self.mean) / self.std, i


def load_model(ckpt, n_classes, size, device):
    base, _, _, size = build_backbone(C.BACKBONE, n_classes, size)
    m = DualHeadNet(base, n_classes, pool="gem")
    sd = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(sd)
    return m.to(device).eval(), size, 2 * int(base.num_features)


@torch.inference_mode()
def predict_all(model, loader, device, n_classes, n_emb):
    P = np.zeros((len(loader.dataset), n_classes), dtype=np.float32)
    E = np.zeros((len(loader.dataset), n_emb), dtype=np.float32)
    done = 0
    for x, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=(device.type == "cuda")):
            logit = model(x)
        ii = idx.numpy()
        P[ii] = torch.softmax(logit.float(), 1).cpu().numpy()
        E[ii] = model.last_feat.float().cpu().numpy()
        done += len(ii)
    return P, E


def js_per_class(a, b):
    a = np.clip(a, 1e-12, 1.0)
    b = np.clip(b, 1e-12, 1.0)
    m = 0.5 * (a + b)
    return 0.5 * (a * (np.log(a) - np.log(m)) + b * (np.log(b) - np.log(m)))


def gated_mean(stack):
    """stack: (V, C) 多视角概率 → 按类 JS 分歧门控后平均"""
    m = stack.mean(0)
    if stack.shape[0] == 1 or not C.USE_GATE:
        return m
    js = np.mean([js_per_class(p, m) for p in stack], axis=0)
    return m * np.exp(-C.GATE_GAMMA * js)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV: image_path,specimen[,view]")
    ap.add_argument("--output", default="results", help="输出前缀(不带扩展名)")
    ap.add_argument("--weights", default=C.WEIGHTS_DIR)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个标本(调试)")
    ap.add_argument("--per-view-emb", action="store_true", help="额外保存逐视角 embedding")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    names = [l.strip() for l in open(C.CLASS_NAMES_FILE, encoding="utf-8") if l.strip()]
    genus_of = [n.split()[0] for n in names]
    genera = sorted(set(genus_of))
    gidx = {g: i for i, g in enumerate(genera)}
    sp2g = np.array([gidx[g] for g in genus_of])
    print(f"类别: {len(names)} 种 / {len(genera)} 属    设备: {device}")

    base_dir = os.path.dirname(os.path.abspath(args.input))
    rows = []
    with open(args.input, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            p = (r.get("image_path") or "").strip()
            if not p:
                continue
            if not os.path.isabs(p) and not os.path.exists(p):
                p = os.path.join(base_dir, p)
            spec = (r.get("specimen") or "").strip() or os.path.splitext(os.path.basename(p))[0]
            v = (r.get("view") or "").strip().lower() or None
            if v is None:
                nm = os.path.basename(p)
                for k, pats in C.VIEW_PATTERNS.items():
                    if any(pt in nm for pt in pats):
                        v = k
                        break
            rows.append({"path": p, "specimen": spec, "view": v or "unknown"})

    ok = [r for r in rows if os.path.exists(r["path"])]
    miss = len(rows) - len(ok)
    print(f"CSV {len(rows)} 行, 有效图片 {len(ok)}" + (f", 缺失 {miss}" if miss else ""))

    specs = sorted({r["specimen"] for r in ok})
    if args.limit:
        specs = specs[:args.limit]
        keep = set(specs)
        ok = [r for r in ok if r["specimen"] in keep]
    print(f"标本 {len(specs)}  图片 {len(ok)}")

    by_view = defaultdict(list)
    for r in ok:
        if r["view"] in C.VIEWS:
            by_view[r["view"]].append(r)
    for v in C.VIEWS:
        if v in by_view:
            print(f"  {v:9s} {len(by_view[v])} 张")

    spec_pos = {s: i for i, s in enumerate(specs)}
    n = len(specs)
    SP = {v: np.zeros((n, len(names)), dtype=np.float32) for v in by_view}
    GP = {v: np.zeros((n, len(genera)), dtype=np.float32) for v in by_view}
    EM = {v: np.zeros((n, 2048), dtype=np.float32) for v in by_view}
    CNT = {v: np.zeros(n) for v in by_view}
    has_genus = False

    t0 = time.time()
    for v in C.VIEWS:
        if v not in by_view:
            continue
        lst = by_view[v]
        paths = [r["path"] for r in lst]
        idx = np.array([spec_pos[r["specimen"]] for r in lst])
        print(f"\n[{v}] {len(paths)} 张 ...", flush=True)

        ds = ImgDS(paths, C.IMG_SIZE, C.MEAN, C.STD)
        dl = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=False,
                        num_workers=C.NUM_WORKERS, pin_memory=True)

        got = False
        for fold in C.FOLDS:
            ck = os.path.join(args.weights, f"{C.TAG_SPECIES}_{v}_fold{fold}.pth")
            if not os.path.exists(ck):
                print(f"  缺种模型 {ck}")
                continue
            model, _, n_emb = load_model(ck, len(names), C.IMG_SIZE, device)
            P, E = predict_all(model, dl, device, len(names), n_emb)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for j, si in enumerate(idx):
                SP[v][si] += P[j]
                EM[v][si] += E[j]
                CNT[v][si] += 1
            got = True
            print(f"  种模型 fold{fold} 完成  ({time.time()-t0:.0f}s)", flush=True)

        if not got:
            continue

        if C.USE_HIERARCHICAL:
            ck = os.path.join(args.weights, f"{C.TAG_GENUS}_{v}_fold0.pth")
            if os.path.exists(ck):
                model, _, _ = load_model(ck, len(genera), C.IMG_SIZE, device)
                G, _ = predict_all(model, dl, device, len(genera), 2048)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                for j, si in enumerate(idx):
                    GP[v][si] += G[j]
                has_genus = True
                print(f"  属模型完成  ({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"  缺属模型 {ck},属级将由种概率求和推导")

    out_rows = []
    emb_final = np.zeros((n, 2048), dtype=np.float32)

    for si, s in enumerate(specs):
        views_used, stack = [], []
        ev = np.zeros(2048)
        for v in C.VIEWS:
            if v in by_view and CNT[v][si] > 0:
                views_used.append(v)
                SP[v][si] /= CNT[v][si]
                EM[v][si] /= CNT[v][si]
                stack.append(SP[v][si])
                e = EM[v][si]
                ev += e / (np.linalg.norm(e) + 1e-12)
        if not stack:
            continue
        P = gated_mean(np.stack(stack))
        pn = P / max(P.sum(), 1e-12)
        emb_final[si] = ev / max(len(views_used), 1)

        order = np.argsort(-pn)[:3]
        top3 = ";".join(f"{names[i]}({pn[i]:.3f})" for i in order)

        final, pred_g = int(order[0]), genus_of[int(order[0])]
        if has_genus:
            Gv = np.zeros(len(genera))
            gc = 0
            for v in views_used:
                if GP[v][si].sum() > 0:
                    Gv += GP[v][si]
                    gc += 1
            if gc:
                gi = int(Gv.argmax())
                cand = np.where(sp2g == gi)[0]
                final = int(cand[int(np.argmax(pn[cand]))])
                pred_g = genera[gi]

        out_rows.append({
            "specimen": s,
            "final_pred": names[final],
            "pred_species": names[int(order[0])],
            "pred_genus": pred_g,
            "prob": f"{pn[int(order[0])]:.4f}",
            "n_views": len(views_used),
            "views": "|".join(views_used),
            "top3": top3,
        })

    csvp = args.output + ".csv"
    with open(csvp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["specimen", "final_pred", "pred_species",
                                          "pred_genus", "prob", "n_views", "views", "top3"])
        w.writeheader()
        w.writerows(out_rows)

    npzp = args.output + "_emb.npz"
    if args.per_view_emb:
        payload = {"specimen": np.array(specs), "embedding": emb_final}
        for v in by_view:
            payload[f"emb_{v}"] = EM[v]
        np.savez_compressed(npzp, **payload)
    else:
        np.savez_compressed(npzp, specimen=np.array(specs), embedding=emb_final)

    print(f"\n完成 {len(out_rows)} 个标本, 用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"  结果   {csvp}")
    print(f"  特征   {npzp}   embedding {emb_final.shape}")
    if len(out_rows) < len(specs):
        print(f"  [!] {len(specs)-len(out_rows)} 个标本没有可用视角,已跳过")


if __name__ == "__main__":
    main()
