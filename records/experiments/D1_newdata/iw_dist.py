#!/usr/bin/env python3
"""iw_dist.py <tag> [<tag2>] — 오분류를 해부학 그래프 '거리'로 재분류하고 두 구성을 비교한다.

거리 1 = 바로 옆 분절, 2 = 한 분절 건너, ≥3 = 멀리. 1~2 는 '경계 판정 실패',
≥3 은 '피처가 위치를 못 잡음' 으로 성격이 다르다.
1.9(BA-SCA junction) ↔ 1.10(BA tip) 은 SCA 기시부가 BA 첨부 바로 아래라 이웃으로 넣는다
(intweak.py 최초판에서 빠져 있었다).
"""
import json, sys, collections
R="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; D=f"{R}/experiments/D1_newdata"
NAME={int(k):v for k,v in json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["location_classes"].items()}
def side(nm): return nm[0] if nm[:2] in ("R-","L-") else ""
def seg(nm):
    b=nm[2:] if nm[:2] in ("R-","L-") else nm; c=b.split()[0]
    return ("5.3j" if "M1-M2" in b else "5.3d") if c=="5.3" else c
PAIRS=[("3.1","3.2"),("3.2","3.3"),("3.3","3.4"),("3.4","3.5"),("3.5","3.6"),("3.6","3.7"),
 ("3.7","5.1"),("3.7","4.2"),("5.1","5.2"),("5.2","5.3j"),("5.3j","5.3d"),("5.1","5.3j"),
 ("4.2","4.1"),("4.1","4.3"),("4.3","4.4"),("4.4","4.5"),("4.2","4.3"),("2.1","2.2"),("1.10","2.1"),
 ("1.1","1.3"),("1.3","1.2"),("1.1","1.5"),("1.5","1.4"),("1.4","1.7"),("1.7","1.6"),
 ("1.4","1.9"),("1.9","1.8"),("1.4","1.10"),("1.9","1.10")]
G=collections.defaultdict(set)
for a,b in PAIRS: G[a].add(b); G[b].add(a)
def bfs(s):
    d={s:0}; q=[s]
    while q:
        x=q.pop(0)
        for y in G[x]:
            if y not in d: d[y]=d[x]+1; q.append(y)
    return d
DIST={k:bfs(k) for k in G}
def cat(gt,pr):
    if pr==0: return "미할당"
    gn,pn=NAME[gt],NAME[pr]; gs,ps=seg(gn),seg(pn)
    flip = side(gn)!=side(pn) and side(gn) and side(pn)
    if gs==ps: return "좌우반전" if flip else "정답"
    d=DIST.get(gs,{}).get(ps,99)
    base = "거리1 (바로 옆 분절)" if d==1 else "거리2 (한 분절 건너)" if d==2 else "거리≥3 (멀리)"
    return base + (" +좌우반전" if flip else "")
for TAG in sys.argv[1:]:
    d=json.load(open(f"{D}/intweak_{TAG}.json"))
    det=[x for x in d["lesions"] if x["detected"]]
    k=collections.Counter()
    for x in det:
        for p in x["preds"]: k[cat(x["cls"],p)]+=1
    tot=sum(k.values()); wrong=tot-k["정답"]
    print(f"\n## {TAG} — 검출병변 {len(det)} × 5시드 = {tot}시행\n")
    print("| 유형 | 시행 | 전체비 | 오답중 |"); print("|---|---|---|---|")
    for t,c in sorted(k.items(), key=lambda z:-z[1]):
        w=f"{100*c/wrong:.1f}%" if t!="정답" else "—"
        print(f"| {t} | {c} | {100*c/tot:.1f}% | {w} |")
    near=sum(c for t,c in k.items() if t.startswith("거리1") or t.startswith("거리2") or t=="좌우반전")
    print(f"\n정답 {k['정답']} ({100*k['정답']/tot:.1f}%) · 오답 {wrong} ({100*wrong/tot:.1f}%)")
    print(f"오답 중 **이웃 2칸 이내 + 좌우반전 = {near} ({100*near/wrong:.1f}%)** · 멀리/미할당 {wrong-near} ({100*(wrong-near)/wrong:.1f}%)")
