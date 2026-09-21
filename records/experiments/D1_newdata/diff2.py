"""구/신 데이터셋 전수 대조 (md5 + 복셀/클래스 수준). 추론 금지, 측정만."""
import sys, os, glob, hashlib, json
import numpy as np, nibabel as nib
OLD, NEW = sys.argv[1], sys.argv[2]
SUB = ["images", "location_masks", "vessel_masks", "type_masks", "location_jsons"]
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
rep = {}
print("=" * 78)
for s in SUB:
    do, dn = f"{OLD}/{s}", f"{NEW}/{s}"
    fo = {os.path.basename(p) for p in glob.glob(f"{do}/*")}
    fn = {os.path.basename(p) for p in glob.glob(f"{dn}/*")}
    same = fo & fn
    ch = sorted(f for f in same if md5(f"{do}/{f}") != md5(f"{dn}/{f}"))
    rep[s] = {"old": len(fo), "new": len(fn), "removed": sorted(fo - fn),
              "added": sorted(fn - fo), "changed": ch}
    print(f"\n[{s}]  구 {len(fo)} · 신 {len(fn)}")
    print(f"  제거 {len(fo-fn)} · 추가 {len(fn-fo)} · 내용변경 {len(ch)}/{len(same)} ({len(ch)/max(len(same),1):.1%})")
    for f in sorted(fo - fn): print(f"    -  {f}")
    for f in sorted(fn - fo): print(f"    +  {f}")
    if s.endswith("masks") and ch:
        det = []
        for f in ch:
            a = np.asanyarray(nib.load(f"{do}/{f}").dataobj)
            b = np.asanyarray(nib.load(f"{dn}/{f}").dataobj)
            if a.shape != b.shape:
                det.append({"f": f, "kind": "shape", "old": list(a.shape), "new": list(b.shape)})
                print(f"    ~  {f}  shape {a.shape} -> {b.shape}")
                continue
            d = int((a != b).sum())
            ca = {int(x) for x in np.unique(a) if x}
            cb = {int(x) for x in np.unique(b) if x}
            det.append({"f": f, "kind": "vox", "ndiff": d, "frac": d / a.size,
                        "cls_old": sorted(ca), "cls_new": sorted(cb)})
            tag = "" if ca == cb else f"   클래스 {sorted(ca)} -> {sorted(cb)}"
            print(f"    ~  {f}  다른복셀 {d:,} ({d/a.size:.5%}){tag}")
        rep[s]["detail"] = det
json.dump(rep, open(os.environ.get("DIFF_OUT", "/tmp/diff.json"), "w"), indent=1)
print("\n" + "=" * 78)
print("요약")
for s in SUB:
    r = rep[s]
    print(f"  {s:16s} 제거 {len(r['removed'])} 추가 {len(r['added'])} 변경 {len(r['changed'])}")
v = rep["vessel_masks"]
print(f"\n>>> 혈관 마스크 실질 변경 {len(v['changed'])}건 / {v['new']}  "
      f"→ {'재학습 필요' if len(v['changed']) > 20 else '재학습 불필요 판정 근거'}")
