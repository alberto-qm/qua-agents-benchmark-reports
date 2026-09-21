import json, math
o=json.load(open('old.json')); n=json.load(open('new.json'))
def fin(x): return x is not None and not (isinstance(x,float) and math.isnan(x))
REF={'Q1':0.6929,'Q2':-0.9628,'Q3':0.5895,'Q4':0.228,'Q5':0.0187,'Q6':0.7468,'qB4':0.0418,'qC2':0.0459,'qC3':0.0454,'qA2':0.0167,'qB2':0.0415}
keys=[k for k in n if k in o and 'error' not in n[k] and 'error' not in o[k]]
cats={'same (<=10 mV)':0,'differ':0,'old only':0,'new only':0,'neither':0}
conc=[]; newonly=[]; fallback=[]
for k in keys:
    a,b=o[k]['idle'],n[k]['idle']; ra,rb=o[k]['r2'],n[k]['r2']
    if fin(a) and fin(b): cats['same (<=10 mV)' if abs(a-b)<=0.01 else 'differ']+=1
    elif fin(a): cats['old only']+=1
    elif fin(b): cats['new only']+=1
    else: cats['neither']+=1
    if fin(a) and not fin(b) and fin(ra) and ra>=0.8: conc.append(k)
    elif fin(a) and fin(b) and abs(a-b)>0.02 and fin(ra) and ra>=0.8: conc.append(k)
    elif not fin(a) and fin(b): newonly.append(k)
    if fin(b) and not n[k]['agrees']: fallback.append(k)
print(len(keys),'datasets',cats)
import statistics
r2o=[o[k]['r2'] for k in keys if fin(o[k]['r2'])]; r2n=[n[k]['r2'] for k in keys if fin(n[k]['r2'])]
print('median R2 old %.2f new %.2f; R2>=0.8: old %d new %d'%(statistics.median(r2o),statistics.median(r2n),sum(r>=0.8 for r in r2o),sum(r>=0.8 for r in r2n)))
def row(k):
    q=o[k]['qubit']; ref=REF.get(q); rs=f" ref={ref:+.3f}" if ref is not None else ""
    return f"  {k} {q:4s} [{o[k]['lo']:+.2f},{o[k]['hi']:+.2f}]{rs} old idle={o[k]['idle']!s:9.9} r2={o[k]['r2']:.2f} agr={o[k]['agrees']!s:5} | new idle={n[k]['idle']!s:9.9} r2={n[k]['r2']:.2f} agr={n[k]['agrees']!s:5} side={n[k]['side']!r:10} tracked={n[k]['tracked']}/{n[k]['cols']}"
print("concerning:", len(conc)); [print(row(k)) for k in conc]
print("new-only:", len(newonly)); [print(row(k)) for k in newonly]
print("new proposals from the measured-maximum fallback:", len(fallback)); [print(row(k)) for k in fallback]
