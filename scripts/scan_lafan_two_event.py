"""Human-level scan of LAFAN (30 fps) for windows with exactly TWO separated high-risk events
(high single-leg lift / kick, or flight), calm upright standing at both ends, small travel."""
import glob, os, sys, json
import numpy as np
FPS=30; SC=1.27/1.7   # human -> G1 scale used by the retargeter
J=dict(Hips=0,LF=3,LT=4,RF=7,RT=8,Neck=12,LH=17,RH=21)
def runs(m):
    out=[];i=0
    while i<len(m):
        if m[i]:
            j=i
            while j<len(m) and m[j]: j+=1
            out.append((i,j)); i=j
        else: i+=1
    return out
def analyze(f):
    p=np.load(f)*SC                       # [T,22,3]
    up=int(np.argmax(np.ptp(p[:, J['Neck']]-p[:, J['Hips']],axis=0)*0+np.abs((p[:, J['Neck']]-p[:, J['Hips']]).mean(0))))
    hz=[a for a in range(3) if a!=up]
    lf=np.minimum(p[:,J['LF'],up],p[:,J['LT'],up]+0.03); rf=np.minimum(p[:,J['RF'],up],p[:,J['RT'],up]+0.03)
    lfa=p[:,J['LF'],up]; rfa=p[:,J['RF'],up]
    g=np.percentile(np.minimum(lfa,rfa),1); lfa=lfa-g; rfa=rfa-g
    spine=p[:,J['Neck']]-p[:,J['Hips']]; tilt=np.degrees(np.arccos(np.clip(spine[:,up]/np.linalg.norm(spine,axis=1),-1,1)))
    sm=lambda x,k=9: np.convolve(x,np.ones(k)/k,mode='same')
    root=p[:,J['Hips']][:,hz]; pv=sm(np.linalg.norm(np.gradient(root,axis=0),axis=1)*FPS)
    rel=p-p[:,[J['Hips']]]; jv=sm(np.linalg.norm(np.gradient(rel,axis=0),axis=2).mean(1)*FPS)
    hipz=p[:,J['Hips'],up]-g
    hi=np.maximum(lfa,rfa); lo=np.minimum(lfa,rfa)
    return dict(p=p,hi=hi,lo=lo,tilt=tilt,pv=pv,jv=jv,root=root,hipz=hipz)
def scan(f, WMIN=6.0, WMAX=12.0, GAP=1.2):
    a=analyze(f); hi,lo,tilt,pv,jv,root,hipz=(a[k] for k in('hi','lo','tilt','pv','jv','root','hipz'))
    T=len(hi); stand=np.median(hipz)
    event=((hi>0.36)&(lo<0.12))|(lo>0.14)           # ankle heights (human ankle ~0.07 above ground when standing, scaled)
    ev=[(s,e) for s,e in runs(event) if e-s>=3]
    cl=[]
    for s,e in ev:
        if cl and s-cl[-1][1]<int(0.7*FPS): cl[-1]=(cl[-1][0],e)
        else: cl.append((s,e))
    calm=(hi<0.13)&(pv<0.35)&(jv<0.35)&(tilt<16)&(np.abs(hipz-stand)<0.08)
    calm_ok=np.array([calm[max(0,i-4):i+5].all() for i in range(T)])
    out=[]
    for i in range(len(cl)-1):
        (s1,e1),(s2,e2)=cl[i],cl[i+1]
        if s2-e1<GAP*FPS or e2-s1>(WMAX-2)*FPS: continue
        best=None
        for s in range(s1-FPS, max(0,e2-int(WMAX*FPS))-1, -3):
            if s<0: break
            if not calm_ok[s]: continue
            if i>0 and cl[i-1][1]>s: break
            for e in range(e2+FPS, min(T-1,s+int(WMAX*FPS))+1, 3):
                if i+2<len(cl) and cl[i+2][0]<e: break
                if not calm_ok[e] or e-s<WMIN*FPS: continue
                path=np.linalg.norm(root[s:e]-root[s],axis=1).max()
                if path>2.0 or tilt[s:e].max()>66 or hipz[s:e].min()<0.55*stand: continue
                cand=(e-s,s,e,path)
                if best is None or cand[0]<best[0]: best=cand
                break
            if best: break
        if best:
            L,s,e,path=best
            def desc(a_,b_): return dict(peak_foot=float(hi[a_:b_].max()),flight=float(lo[a_:b_].max()),tilt=float(tilt[a_:b_].max()),dur=(b_-a_)/FPS,phase=((a_+b_)/2-s)/(e-s))
            out.append(dict(clip=os.path.basename(f)[:-4],s=int(s),e=int(e),len_s=L/FPS,path=float(path),max_tilt=float(tilt[s:e].max()),e1=desc(s1,e1),e2=desc(s2,e2),gap_s=(s2-e1)/FPS,mid_calm=float(calm[e1:s2].mean()),jv_mean=float(jv[s:e].mean())))
    return out, cl, a
if __name__=="__main__":
    rows=[]
    for f in sorted(glob.glob("demo_data/lafan/*.npy")):
        r,cl,a=scan(f); rows+=r
    json.dump(rows,open(sys.argv[1],"w"),indent=1)
    print(len(rows),"two-event windows")
    for r in sorted(rows,key=lambda r:(r['clip'],r['s'])):
        e1,e2=r['e1'],r['e2']
        print(f"{r['clip']:28s} {r['s']:5d}-{r['e']:5d} {r['len_s']:4.1f}s path={r['path']:.2f} tilt={r['max_tilt']:3.0f}  E1: foot={e1['peak_foot']:.2f} fl={e1['flight']:.2f} @{e1['phase']:.2f}  E2: foot={e2['peak_foot']:.2f} fl={e2['flight']:.2f} @{e2['phase']:.2f}  gap={r['gap_s']:.1f}s")
