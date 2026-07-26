import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
os.chdir("/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV")
DOCS="platform/docs"
M=json.load(open(f"{DOCS}/repro_data.json"))
SW4=json.load(open(f"{DOCS}/s6_sweep_v4.json"))
SWF=json.load(open(f"{DOCS}/s6_sweep_candF.json"))
TWS=[0.,5.,10.,15.,20.,22.5,25.,27.5,30.,35.,40.,45.]; FS=[1.4,1.7,2.0,2.3,2.6]
FCOL={1.4:"#4477aa",1.7:"#66ccee",2.0:"#228833",2.3:"#cc3311",2.6:"#aa3377"}
UCOL={6.:"#228833",8.:"#cc3311",10.:"#4477aa"}
ACOL={0.:"#4477aa",5.:"#cc3311",10.:"#228833",15.:"#aa3377"}
def LT(SW,U,f,tw,aoa):
    v=SW.get(f"{U:g}_{f:g}_{tw:g}_{aoa:g}")
    return (None,None) if v is None else (v["L"], v["T"])   # (L, T) explicit
def Lrow(SW,U,f,tw,aoa): return LT(SW,U,f,tw,aoa)[0]
def Trow(SW,U,f,tw,aoa): return LT(SW,U,f,tw,aoa)[1]
def curve3(ax,mkey,xs,y4,y41,color):
    d=M.get(mkey)
    if d is not None:
        ax.plot(d["x"],d["exp"],"-",color=color,lw=2.6,marker="x",ms=8,mew=2,zorder=5)
    ok4=[(x,v) for x,v in zip(xs,y4) if v is not None]
    ok41=[(x,v) for x,v in zip(xs,y41) if v is not None]
    if ok4: x4,y4b=zip(*ok4); ax.plot(x4,y4b,":",color=color,lw=2.0,alpha=0.85,zorder=4)
    if ok41:
        x41,y41b=zip(*ok41); ax.plot(x41,y41b,"--",color=color,lw=2.4,marker="o",ms=8,mfc="white",mew=1.6,zorder=6)
        for xx,yy in zip(x41,y41b): ax.annotate(f"{yy:.1f}",(xx,yy),textcoords="offset points",xytext=(0,8),fontsize=6.5,color=color,ha="center")
def leg(ax,extra):
    base=[Line2D([0],[0],color="k",lw=2.6,marker="x",ms=8,mew=2,label="solid x = MEASURED"),
          Line2D([0],[0],color="k",lw=2.0,ls=":",label="dotted = v4"),
          Line2D([0],[0],color="k",lw=2.4,ls="--",marker="o",ms=8,mfc="white",label="dashed o = v4.1")]
    ax.legend(handles=base+extra,fontsize=7.5,loc="best")
# Fig17: U8 AoA5 twist sweep, 5 freq
fig,(aT,aL)=plt.subplots(1,2,figsize=(17,6))
for f in FS:
    T4=[Trow(SW4,8,f,tw,5) for tw in TWS]; T41=[Trow(SWF,8,f,tw,5) for tw in TWS]
    L4=[Lrow(SW4,8,f,tw,5) for tw in TWS]; L41=[Lrow(SWF,8,f,tw,5) for tw in TWS]
    curve3(aT,f"17|a|{f:.1f}",TWS,T4,T41,FCOL[f])
    curve3(aL,f"17|b|{f:.1f}",TWS,L4,L41,FCOL[f])
aT.set(xlabel="twist (deg nominal)",ylabel="Thrust (N)",title="(a) Thrust vs twist"); aT.grid(alpha=0.3)
aL.set(xlabel="twist (deg nominal)",ylabel="Lift (N)",title="(b) Lift vs twist"); aL.grid(alpha=0.3)
leg(aT,[Line2D([0],[0],color=FCOL[f],lw=2,label=f"f={f:g}") for f in FS]); leg(aL,[Line2D([0],[0],color=FCOL[f],lw=2,label=f"f={f:g}") for f in FS])
fig.suptitle("Fig17 MEASURED vs v4 vs v4.1 (U=8, AoA=5)",fontsize=13); fig.tight_layout()
fig.savefig(f"{DOCS}/fig17_en.png",dpi=140); plt.close(fig)
# Fig18: tw22.5 AoA5 freq sweep, 3 speeds
fig,(aT,aL)=plt.subplots(1,2,figsize=(17,6))
for U in (6.,8.,10.):
    T4=[Trow(SW4,U,f,22.5,5) for f in FS]; T41=[Trow(SWF,U,f,22.5,5) for f in FS]
    L4=[Lrow(SW4,U,f,22.5,5) for f in FS]; L41=[Lrow(SWF,U,f,22.5,5) for f in FS]
    curve3(aT,f"18|a|{U}",FS,T4,T41,UCOL[U]); curve3(aL,f"18|b|{U}",FS,L4,L41,UCOL[U])
aT.set(xlabel="flap freq (Hz)",ylabel="Thrust (N)",title="(a) Thrust vs freq @tw22.5"); aT.grid(alpha=0.3)
aL.set(xlabel="flap freq (Hz)",ylabel="Lift (N)",title="(b) Lift vs freq @tw22.5"); aL.grid(alpha=0.3)
leg(aT,[Line2D([0],[0],color=UCOL[U],lw=2,label=f"U={U:g}") for U in (6.,8.,10.)]); leg(aL,[Line2D([0],[0],color=UCOL[U],lw=2,label=f"U={U:g}") for U in (6.,8.,10.)])
fig.suptitle("Fig18 MEASURED vs v4 vs v4.1 (AoA=5, tw=22.5)",fontsize=13); fig.tight_layout()
fig.savefig(f"{DOCS}/fig18_en.png",dpi=140); plt.close(fig)
# Fig19
fig,ax=plt.subplots(2,2,figsize=(17,11))
for a in (0.,5.,10.,15.):
    T4=[Trow(SW4,8,f,22.5,a) for f in FS]; T41=[Trow(SWF,8,f,22.5,a) for f in FS]
    L4=[Lrow(SW4,8,f,22.5,a) for f in FS]; L41=[Lrow(SWF,8,f,22.5,a) for f in FS]
    curve3(ax[0,0],f"19|a|{a:g}",FS,T4,T41,ACOL[a]); curve3(ax[0,1],f"19|b|{a:g}",FS,L4,L41,ACOL[a])
for a in (0.,5.,10.,15.):
    T4=[Trow(SW4,8,2.6,tw,a) for tw in TWS]; T41=[Trow(SWF,8,2.6,tw,a) for tw in TWS]
    L4=[Lrow(SW4,8,2.6,tw,a) for tw in TWS]; L41=[Lrow(SWF,8,2.6,tw,a) for tw in TWS]
    curve3(ax[1,0],f"19|c|{a:g}",TWS,T4,T41,ACOL[a]); curve3(ax[1,1],f"19|d|{a:g}",TWS,L4,L41,ACOL[a])
for axi,t,xl in zip(ax.ravel(),["(a) Thrust vs freq","(b) Lift vs freq","(c) Thrust vs twist @f2.6","(d) Lift vs twist @f2.6"],["flap freq (Hz)","flap freq (Hz)","twist (deg)","twist (deg)"]):
    axi.set(xlabel=xl,ylabel="N",title=t); axi.grid(alpha=0.3)
for axi in ax.ravel(): leg(axi,[Line2D([0],[0],color=ACOL[a],lw=2,label=f"AoA={a:g}") for a in (0.,5.,10.,15.)])
fig.suptitle("Fig19 MEASURED vs v4 vs v4.1 (numbers=v4.1 values in N)",fontsize=13); fig.tight_layout()
fig.savefig(f"{DOCS}/fig19_en.png",dpi=140); plt.close(fig)
print("saved fig17/18/19_en.png (FIXED: L/T no longer swapped)")
