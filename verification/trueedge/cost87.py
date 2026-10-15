import time
exec(open('lip87/unit87.py').read().split("E._P['occ_img_refine'] = 1.0\nimg, gt")[0])
E._P['occ_img_refine']=1.0
for n in (360, 500):
    h=w=n; globals().update(h=n,w=n); yy_, xx_ = np.mgrid[0:n, 0:n].astype(np.float32)
    globals()['yy'], globals()['xx'] = yy_, xx_
    img, gt = scene((235,235,235),(120,150,200), cx=n//2, cy=int(n*.58), r=int(n*.19)); k0 = soft_k0(gt)
    ts=[]
    for _ in range(15):
        t=time.perf_counter(); E._refine_keep_img(k0, img); ts.append(time.perf_counter()-t)
    t=time.perf_counter(); E._refine_keep_img(np.ones_like(k0), img); t0=time.perf_counter()-t
    print(f"face region {n}px: object present {1000*np.median(ts):.1f} ms, no object {1000*t0:.2f} ms")
