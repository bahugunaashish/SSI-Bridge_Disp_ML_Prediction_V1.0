"""Fixed-size FFT, Prony and time-window descriptors for bidirectional EQs.

Acceleration is expected in m/s^2 and time in seconds. The model receives no
sample-by-sample acceleration: it receives 24 ordered window summaries and a
fixed-size record summary.
"""
import numpy as np

WINDOWS = 24
WINDOW_FEATURES = 10  # per component
PRONY_MODES = 3
_AXIS_GLOBAL_NAMES=("PGA_m_per_s2","RMS_m_per_s2","Arias_intensity","CAV_m_per_s","D5_95_s","D5_75_s","FFT_dominant_Hz","FFT_centroid_Hz","FFT_bandwidth_Hz","FFT_energy_0_1_Hz","FFT_energy_1_3_Hz","FFT_energy_3_6_Hz","FFT_energy_over_6_Hz")+tuple(x for i in range(1,PRONY_MODES+1) for x in (f"Prony{i}_frequency_Hz",f"Prony{i}_damping_ratio",f"Prony{i}_relative_amplitude"))
GLOBAL_FEATURE_NAMES=tuple([f"X_{x}" for x in _AXIS_GLOBAL_NAMES]+[f"Y_{x}" for x in _AXIS_GLOBAL_NAMES]+["XY_correlation","dt_s","record_duration_s"])
_WINDOW_NAMES=("RMS_m_per_s2","peak_abs_m_per_s2","mean_abs_m_per_s2","Arias_energy","FFT_dominant_Hz","FFT_centroid_Hz","FFT_energy_0_1_Hz","FFT_energy_1_3_Hz","FFT_energy_3_6_Hz","FFT_energy_over_6_Hz")
WINDOW_FEATURE_NAMES=tuple([f"X_{x}" for x in _WINDOW_NAMES]+[f"Y_{x}" for x in _WINDOW_NAMES])


def _spectrum(x, dt):
    x = np.asarray(x, dtype=np.float64)
    x = x - (x.mean() if len(x) else 0.0)
    if len(x) < 4 or dt <= 0:
        return np.array([0.0]), np.array([0.0])
    win = np.hanning(len(x))
    amp = np.abs(np.fft.rfft(x * win))
    freq = np.fft.rfftfreq(len(x), dt)
    if len(amp): amp[0] = 0.0
    return freq, amp


def _prony(x, dt, modes=PRONY_MODES):
    """Estimate damped modes with an AR/Prony polynomial and least squares.

    Unstable roots, DC roots and roots above Nyquist are discarded. Outputs
    are (frequency Hz, damping ratio, relative amplitude), strongest first.
    This is a compact descriptive estimate, not a substitute for a structural
    modal analysis.
    """
    x = np.asarray(x, dtype=np.float64)
    order = min(20, max(4, len(x) // 20))
    if len(x) <= order + 2 or dt <= 0 or np.std(x) < 1e-14:
        return np.zeros((modes, 3), dtype=np.float64)
    x = x - x.mean()
    # Downsample only the fitting window to keep the least-squares problem bounded.
    stride = max(1, len(x) // 12000)
    x = x[::stride]
    p = min(order, max(2, len(x)//20))
    design = np.column_stack([x[p-j-1:len(x)-j-1] for j in range(p)])
    try:
        coeff, *_ = np.linalg.lstsq(design, -x[p:], rcond=1e-8)
        roots = np.roots(np.r_[1.0, coeff])
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return np.zeros((modes, 3), dtype=np.float64)
    dts = dt * stride
    nyquist = .5 / dts
    found = []
    for z in roots:
        radius = abs(z)
        angle=np.angle(z)
        # A real signal has conjugate roots; keep one root per pair.
        if not (0.0 < radius < 1.0) or angle <= 1e-8:
            continue
        freq = angle / (2*np.pi*dts)
        if not (0.0 < freq <= nyquist):
            continue
        decay = -np.log(radius) / dts
        damp = decay / np.sqrt(decay*decay + (2*np.pi*freq)**2)
        found.append((freq, damp, z))
    if found:
        try:
            zpos=np.asarray([item[2] for item in found],dtype=np.complex128)
            powers=np.arange(len(x),dtype=np.float64)
            vand=np.column_stack([zpos[None,:]**powers[:,None],np.conj(zpos)[None,:]**powers[:,None]])
            residues,*_=np.linalg.lstsq(vand,x,rcond=1e-8)
            amp=2*np.abs(residues[:len(found)])
            found=[(f,d,z,float(a)) for (f,d,z),a in zip(found,amp)]
        except (ValueError,np.linalg.LinAlgError,FloatingPointError):
            found=[(f,d,z,float(abs(z))) for f,d,z in found]
        found.sort(key=lambda a:a[3],reverse=True)
    out = np.zeros((modes, 3), dtype=np.float64)
    if found:
        vals = np.asarray([[f,d,a] for f,d,_z,a in found[:modes]], dtype=np.float64)
        vals[:, 2] /= max(vals[:, 2].sum(), 1e-12)
        out[:len(vals)] = vals
    return out


def _duration(x, dt, fraction):
    energy = np.square(np.asarray(x, dtype=np.float64))
    total = energy.sum()
    if total <= 1e-20 or len(x) < 2:
        return 0.0
    c = np.cumsum(energy) / total
    return float((np.searchsorted(c, fraction[1]) - np.searchsorted(c, fraction[0])) * dt)


def _axis_global(x, dt):
    x = np.asarray(x, dtype=np.float64)
    f, a = _spectrum(x, dt)
    power = a*a
    total = power.sum() + 1e-20
    centroid = float(np.sum(f*power)/total)
    dominant = float(f[np.argmax(a)]) if len(a) else 0.0
    bandwidth = float(np.sqrt(np.sum(np.square(f-centroid)*power)/total))
    nyq = .5/dt if dt > 0 else 0
    bands = [(0,1),(1,3),(3,6),(6,nyq+1e-12)]
    ratios = [float(power[(f>=lo)&(f<hi)].sum()/total) for lo,hi in bands]
    arias = float(np.pi/(2*9.80665) * dt * np.square(x).sum())
    pga = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x*x))) if len(x) else 0.0
    cav = float(dt*np.abs(x).sum())
    pr = _prony(x,dt).reshape(-1).tolist()
    # PGA, RMS, Arias intensity, CAV, D5-95, D5-75, FFT peak/centroid/bandwidth,
    # four FFT-band ratios, followed by three (frequency,damping,relative amp) modes.
    return [pga,rms,arias,cav,_duration(x,dt,(.05,.95)),_duration(x,dt,(.05,.75)),dominant,centroid,bandwidth,*ratios,*pr]


def _window_axis(x, dt, count=WINDOWS):
    x=np.asarray(x,dtype=np.float64); out=np.zeros((count,WINDOW_FEATURES),dtype=np.float32)
    bounds=np.linspace(0,len(x),count+1,dtype=int)
    for i,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:])):
        w=x[lo:hi]
        if not len(w): continue
        f,a=_spectrum(w,dt); p=a*a; total=p.sum()+1e-20
        bands=[(0,1),(1,3),(3,6),(6,.5/dt+1e-12)]
        ratios=[float(p[(f>=a0)&(f<b0)].sum()/total) for a0,b0 in bands]
        arias=float(np.pi/(2*9.80665)*dt*np.square(w).sum())
        centroid=float(np.sum(f*p)/total); dom=float(f[np.argmax(a)]) if len(a) else 0.0
        out[i]=[float(np.sqrt(np.mean(w*w))),float(np.max(np.abs(w))),float(np.mean(np.abs(w))),arias,dom,centroid,*ratios]
    return out


def extract_eq_descriptors(acceleration, time, sequence_windows=WINDOWS):
    """Return global Prony/FFT descriptors and ordered window-feature sequence.

    ``acceleration`` must have two horizontal columns (X,Y), in m/s^2.
    ``time`` is seconds. Irregular sampling is rejected because FFT/Prony
    descriptors require a uniform time step.
    """
    a=np.asarray(acceleration,dtype=np.float64); t=np.asarray(time,dtype=np.float64).reshape(-1)
    if a.ndim!=2 or a.shape[1]<2 or len(a)!=len(t) or len(t)<4: raise ValueError("EQ input must be [T,2+] and match a time vector")
    d=np.diff(t); dt=float(np.median(d))
    if dt<=0 or not np.allclose(d,dt,rtol=1e-3,atol=max(1e-8,dt*1e-4)): raise ValueError("EQ time samples must be uniformly spaced")
    axes=a[:,:2]
    global_features=np.asarray([*_axis_global(axes[:,0],dt),*_axis_global(axes[:,1],dt),
        float(np.corrcoef(axes.T)[0,1]) if np.std(axes[:,0])>1e-12 and np.std(axes[:,1])>1e-12 else 0.0,
        dt,float(max(0,len(t)-1)*dt)],dtype=np.float32)
    seq=np.concatenate([_window_axis(axes[:,i],dt,sequence_windows) for i in range(2)],axis=1)
    return global_features,seq
