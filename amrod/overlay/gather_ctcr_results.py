#!/usr/bin/env python3
from pathlib import Path
from statistics import mean
import csv, re, socket

HOME = Path.home()
HOST = socket.gethostname()
VP = [("A2","ctcr_A2_v2_perbox_full_no_ctpv"),("A","ctcr_A_v2_full_no_ctpv"),("B","ctcr_B_v2_hard_t03_no_ctpv"),("C","ctcr_C_v2_soft_a02_no_ctpv")]

def parse(path):
    lines = path.read_text(errors="replace").splitlines()
    ap, mi = [], []
    for i,l in enumerate(lines):
        if "copypaste: Task: bbox" in l and i+2 < len(lines):
            try: ap.append(float(lines[i+2].split("copypaste:",1)[1].split(",")[1]))
            except: pass
        if "copypaste: Task: sem_seg" in l and i+2 < len(lines):
            try: mi.append(float(lines[i+2].split("copypaste:",1)[1].split(",")[0]))
            except: pass
    return ap, mi

def seed(name):
    m = re.search(r"_seed(\d+)", name)
    return int(m.group(1)) if m else 0

def variant(name):
    for v,p in VP:
        if p in name: return v
    return "?"

rows=[]; used=[]

def add_logs(pattern, bench, domains, expected):
    for p in sorted(HOME.glob(pattern)):
        if not any(pref in p.name for _,pref in VP): continue
        ap,mi=parse(p); n=min(len(ap),len(mi))
        if n==0: continue
        used.append((p,len(ap),len(mi)))
        s=seed(p.name); v=variant(p.name)
        for i in range(n):
            rows.append({
                "host":HOST,"benchmark":bench,"variant":v,"seed":s,
                "eval_idx":i+1,"round":i//len(domains)+1 if bench=="csc_mixed_lt_x10" else i+1,
                "domain":domains[i%len(domains)],
                "ap50":ap[i],"miou":mi[i],"log":p.name,
                "complete_expected":int(len(ap)==expected and len(mi)==expected),
                "n_ap50_in_log":len(ap),"n_miou_in_log":len(mi)
            })

add_logs("ctcr_*_csc_mixed_lt_x10*.log","csc_mixed_lt_x10",
         ["fog","motion_blur","snow","brightness","defocus_blur"],50)
add_logs("ctcr_*_acdc_seed*.log","acdc",["fog","night","rain","snow"],4)
add_logs("ctcr_*_fogx10.log","csc_fog_x10",["fog"],10)

detail=HOME/f"ctcr_results_detail_{HOST}.csv"
fields=["host","benchmark","variant","seed","eval_idx","round","domain","ap50","miou","log","complete_expected","n_ap50_in_log","n_miou_in_log"]
with detail.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

groups={}
for r in rows:
    for d in ("ALL",r["domain"]):
        groups.setdefault((r["benchmark"],r["variant"],r["seed"],d),[]).append(r)

summary=HOME/f"ctcr_results_summary_{HOST}.csv"
sfields=["host","benchmark","variant","seed","domain","n","mean_ap50","mean_miou","first_ap50","last_ap50","delta_ap50","first_miou","last_miou","delta_miou"]
srows=[]
for (b,v,s,d),rs in sorted(groups.items()):
    aps=[float(x["ap50"]) for x in rs]; mis=[float(x["miou"]) for x in rs]
    srows.append({"host":HOST,"benchmark":b,"variant":v,"seed":s,"domain":d,"n":len(rs),
                  "mean_ap50":mean(aps),"mean_miou":mean(mis),
                  "first_ap50":aps[0],"last_ap50":aps[-1],"delta_ap50":aps[-1]-aps[0],
                  "first_miou":mis[0],"last_miou":mis[-1],"delta_miou":mis[-1]-mis[0]})
with summary.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=sfields); w.writeheader(); w.writerows(srows)

print("="*100); print("HOST:",HOST); print("="*100)
for p,a,m in used: print(f"{p.name:<74} AP50={a:>2} mIoU={m:>2}")
print("\nOVERALL PER-RUN SUMMARY")
for r in srows:
    if r["domain"]=="ALL":
        print(f"{r['benchmark']:<18} {r['variant']} seed={r['seed']:<3} n={r['n']:<2} AP50={r['mean_ap50']:.4f} mIoU={r['mean_miou']:.4f}")
print("\nWROTE"); print(detail); print(summary)
