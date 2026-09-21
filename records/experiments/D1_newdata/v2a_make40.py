#!/usr/bin/env python3
"""V2-A 용 40클래스 파이프라인 복사본을 **원본에서 기계적으로** 만든다 (원본은 건드리지 않는다).

만드는 것 (code/sblee/nnunet/scripts/)
  d9xx_lib_v40.py   vessel_dense_names() 가 40클래스 fine 매핑을 읽는다
  c4_bp40.py        c4_branchpoint_graph.py 복사 · 이름표 40 · 인접표에 신규 id 37~40 추가
  c5_v40.py         c5_location_v2.py 복사 · d9xx_lib_v40 import · ICA 분기점 쌍/랜드마크 교체

바뀌는 분기점 쌍 (좌우 각각)
  기존 ICA-C6-C7 이 낀 7쌍 → 신규 8쌍
    C1-C5↔C6 · C6↔OA · **C6↔C7(신규 직렬경계 3.3|3.4)** · C7↔Pcom · C7↔AChA
    · **C7↔terminus(신규 직렬경계 3.6|3.7)** · terminus↔M1 · terminus↔A1A2
  BP_DIM 34 → 38 (ICA 12쌍 → 16쌍). 혈관 축 36 → 40 → 피처 112 → 124.
각 치환은 원문 문자열이 정확히 1번 나타나는지 assert 한다 — 원본이 바뀌면 조용히 틀리지 않고 멈춘다.
"""
import json, os, hashlib

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
S = f"{R}/code/sblee/nnunet/scripts"
FINE_JSON = "/home/sblee/TopAneu-26/labeling/vessel_mapping_fine.json"


def sub(src, old, new, what):
    assert src.count(old) == 1, f"{what}: 원문이 {src.count(old)}번"
    return src.replace(old, new)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


# ── d9xx_lib_v40 ────────────────────────────────────────────────────────────
s = open(f"{S}/d9xx_lib.py").read()
s = sub(s, '''def vessel_dense_names():
    """D800 dense id(int, 1..36) -> name."""
    lab = json.load(open(VES_RAW / "dataset.json"))["labels"]
    return {int(v): k for k, v in lab.items() if int(v) != 0}''',
        f'''def vessel_dense_names():
    """[V2-A] 40클래스 fine 매핑 -> name. (원본 d9xx_lib 은 D800 36클래스)"""
    lab = json.load(open("{FINE_JSON}"))["labels"]
    return {{int(v): k for k, v in lab.items() if int(v) != 0}}''', "d9xx names")
open(f"{S}/d9xx_lib_v40.py", "w").write("# 자동생성: experiments/D1_newdata/v2a_make40.py · 원본 d9xx_lib.py md5 "
                                        + md5(f"{S}/d9xx_lib.py") + "\n" + s)

# ── c4_bp40 ─────────────────────────────────────────────────────────────────
s = open(f"{S}/c4_branchpoint_graph.py").read()
s = sub(s, '''def vessel_names():
    lab = json.load(open(VES_RAW / "dataset.json"))["labels"]
    return {int(v): k for k, v in lab.items() if int(v) != 0}''',
        f'''def vessel_names():
    """[V2-A] 40클래스 fine 매핑"""
    lab = json.load(open("{FINE_JSON}"))["labels"]
    return {{int(v): k for k, v in lab.items() if int(v) != 0}}''', "c4 names")
s = sub(s, '''    adj = {}
    for k, v in p["classes"].items():
        adj[int(k)] = set(int(n) for n in v["neighbors"])
    return adj''',
        '''    adj = {}
    for k, v in p["classes"].items():
        adj[int(k)] = set(int(n) for n in v["neighbors"])
    # [V2-A] 기존 ICA-C6-C7(4/6)을 C6(4/6)·C7(37/38)·terminus(39/40)로 나눴다.
    # 세 구획 모두 기존 4/6 의 이웃을 물려받고 서로 이웃이다(허용적으로 — 검증은 엉뚱한 노드만 거른다).
    for old, trio in ((4, (4, 37, 39)), (6, (6, 38, 40))):
        base = set(adj.get(old, set()))
        for a in trio:
            adj[a] = set(base) | (set(trio) - {a})
            for nb in base:
                adj.setdefault(nb, set()).add(a)
    return adj''', "c4 adjacency")
open(f"{S}/c4_bp40.py", "w").write("# 자동생성: experiments/D1_newdata/v2a_make40.py · 원본 c4_branchpoint_graph.py md5 "
                                   + md5(f"{S}/c4_branchpoint_graph.py") + "\n" + s)

# ── c5_v40 ──────────────────────────────────────────────────────────────────
s = open(f"{S}/c5_location_v2.py").read()
s = sub(s, "import d9xx_lib as L", "import d9xx_lib_v40 as L", "c5 import")
s = sub(s, '''    ("R-ICA-C1-C5", "R-ICA-C6-C7"),                    # 3.1/3.3 경계 (serial)
    ("L-ICA-C1-C5", "L-ICA-C6-C7"),
    ("R-ICA-C6-C7", "R-OA"), ("L-ICA-C6-C7", "L-OA"),      # 3.2 C6-OA-junction
    ("R-ICA-C6-C7", "R-Pcom"), ("L-ICA-C6-C7", "L-Pcom"),  # 3.4 C7-Pcom-junction
    ("R-ICA-C6-C7", "R-AChA"), ("L-ICA-C6-C7", "L-AChA"),  # 3.5 C7-AChA-junction
    ("R-ICA-C6-C7", "R-M1"), ("L-ICA-C6-C7", "L-M1"),      # 3.7 terminus
    ("R-ICA-C6-C7", "R-A1A2"), ("L-ICA-C6-C7", "L-A1A2"),  # 3.7 terminus''',
        '''    # [V2-A] ICA-C6-C7 을 C6 / C7 / C7-terminus 로 나눈 40클래스판
    ("R-ICA-C1-C5", "R-ICA-C6"), ("L-ICA-C1-C5", "L-ICA-C6"),                    # 3.1/3.3 경계
    ("R-ICA-C6", "R-OA"), ("L-ICA-C6", "L-OA"),                                  # 3.2 C6-OA-junction
    ("R-ICA-C6", "R-ICA-C7"), ("L-ICA-C6", "L-ICA-C7"),                          # 3.3|3.4 경계 (신규)
    ("R-ICA-C7", "R-Pcom"), ("L-ICA-C7", "L-Pcom"),                              # 3.4 C7-Pcom-junction
    ("R-ICA-C7", "R-AChA"), ("L-ICA-C7", "L-AChA"),                              # 3.5 C7-AChA-junction
    ("R-ICA-C7", "R-ICA-C7-terminus"), ("L-ICA-C7", "L-ICA-C7-terminus"),        # 3.6|3.7 경계 (신규)
    ("R-ICA-C7-terminus", "R-M1"), ("L-ICA-C7-terminus", "L-M1"),                # 3.7 terminus
    ("R-ICA-C7-terminus", "R-A1A2"), ("L-ICA-C7-terminus", "L-A1A2"),            # 3.7 terminus''',
        "c5 junction pairs")
s = sub(s, '''    "Rterm": [{"R-ICA-C6-C7", "R-M1"}, {"R-ICA-C6-C7", "R-A1A2"}],
    "Lterm": [{"L-ICA-C6-C7", "L-M1"}, {"L-ICA-C6-C7", "L-A1A2"}],''',
        '''    # [V2-A] 종말부 노드는 terminus 가 M1/A1 과 만나는 곳. 구획이 짧아 C7 이 직접 닿는 경우도 허용
    "Rterm": [{"R-ICA-C7-terminus", "R-M1"}, {"R-ICA-C7-terminus", "R-A1A2"},
              {"R-ICA-C7", "R-M1"}, {"R-ICA-C7", "R-A1A2"}],
    "Lterm": [{"L-ICA-C7-terminus", "L-M1"}, {"L-ICA-C7-terminus", "L-A1A2"},
              {"L-ICA-C7", "L-M1"}, {"L-ICA-C7", "L-A1A2"}],''', "c5 landmarks")
open(f"{S}/c5_v40.py", "w").write("# 자동생성: experiments/D1_newdata/v2a_make40.py · 원본 c5_location_v2.py md5 "
                                  + md5(f"{S}/c5_location_v2.py") + "\n" + s)
print("생성: d9xx_lib_v40.py · c4_bp40.py · c5_v40.py")
