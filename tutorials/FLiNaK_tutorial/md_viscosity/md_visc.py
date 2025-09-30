# Python script to parse LAMMPS log file and compute viscosity from thermo `v_v11,v_v22,v_v33` outputs.
# It will also optionally parse S0St700.dat in the same folder if present and compute viscosity from correlations
# using a scale factor extracted from the log (if available) or a user-provided scale.
#
# Saves results to md_visc_results.csv and prints summary.
#
# Usage (in this notebook): it will run on the uploaded file /mnt/data/log.lammps by default.
# The script is written to be saved as md_visc.py as well.

import re, csv, os, sys, math
from statistics import mean, stdev, median
import numpy as np

LOGPATH = "log.lammps"
DATNAME = "S0St700.dat"
OUTCSV = "md_visc_results.csv"

def parse_log_for_thermo(filename):
    header_pat = re.compile(r'^\s*Step\b', re.IGNORECASE)
    number_start = re.compile(r'^\s*[-+0-9]')
    header = None
    rows = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if header is None:
                if header_pat.search(line) and ('v_v11' in line or 'v11' in line or 'v_v22' in line):
                    header = re.split(r'\s+', line.strip())
                    header = [h.strip() for h in header if h.strip()!='']
                    continue
            else:
                if number_start.search(line):
                    toks = re.split(r'\s+', line.strip())
                    if len(toks) < 2:
                        try:
                            nums = [float(t) for t in toks]
                        except:
                            break
                        rows.append(nums); continue
                    try:
                        nums = [float(toks[i]) for i in range(min(len(toks), len(header)))]
                    except:
                        break
                    rows.append(nums)
                else:
                    if rows:
                        break
                    else:
                        continue
    return header, rows

def compute_eta_from_rows(header, rows):
    hdr_low = [h.lower() for h in header]
    def find_index(names):
        for name in names:
            if name.lower() in hdr_low:
                return hdr_low.index(name.lower())
        for name in names:
            for i,h in enumerate(hdr_low):
                if name.lower() in h:
                    return i
        return None
    idx1 = find_index(['v_v11','v11','v_v11'])
    idx2 = find_index(['v_v22','v22','v_v22'])
    idx3 = find_index(['v_v33','v33','v_v33'])
    if idx1 is None or idx2 is None or idx3 is None:
        raise RuntimeError("Could not find v_v11/v_v22/v_v33 columns in header: " + str(header))
    steps = []
    eta_series = []
    for r in rows:
        try:
            v1 = float(r[idx1]); v2 = float(r[idx2]); v3 = float(r[idx3])
        except:
            continue
        eta = (v1 + v2 + v3) / 3.0
        eta_series.append(eta)
        try:
            si = hdr_low.index('step'); steps.append(int(r[si]))
        except ValueError:
            steps.append(None)
    return {'header':header, 'idxs':(idx1,idx2,idx3), 'steps':steps, 'eta':eta_series}

def parse_scale_from_log(filename):
    # search for a line like: variable v11 equal trap(f_SS[3])*5.95036785757622e-12
    pat = re.compile(r'variable\s+\w+\s+equal\s+trap\(\s*f_SS\[\d+\]\s*\)\s*\*\s*([0-9.eE+-]+)')
    pat2 = re.compile(r'variable\s+scale\s+equal\s+([0-9.eE+\-*/\(\)\sA-Za-z0-9_.]+)')
    scale = None
    dt = None; s=None; V=None; T=None
    with open(filename,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            m = pat.search(line)
            if m:
                try:
                    scale = float(m.group(1))
                    return scale, dt, s, V, T
                except:
                    pass
            m2 = pat2.search(line)
            if m2 and scale is None:
                # try to evaluate simple numeric expression if present
                expr = m2.group(1).strip()
                # skip complex expressions containing variable names, but catch numeric literal
                try:
                    # remove accidental variable names like ${dt} etc -> not safe to eval raw
                    cleaned = re.sub(r'[^\d.eE+\-*/().]', '', expr)
                    if cleaned and any(c.isdigit() for c in cleaned):
                        scale = float(eval(cleaned))
                        return scale, dt, s, V, T
                except Exception:
                    pass
            # parse dt, s, V, T variables if defined
            mdt = re.search(r'variable\s+dt\s+equal\s+([0-9.eE+\-]+)', line)
            if mdt:
                try: dt = float(mdt.group(1))
                except: pass
            ms = re.search(r'variable\s+s\s+equal\s+([0-9.eE+\-]+)', line)
            if ms:
                try: s = float(ms.group(1))
                except: pass
            mV = re.search(r'variable\s+V\s+equal\s+([0-9.eE+\-]+)', line)
            if mV:
                try: V = float(mV.group(1))
                except: pass
            mT = re.search(r'variable\s+T\s+equal\s+([0-9.eE+\-]+)', line)
            if mT:
                try: T = float(mT.group(1))
                except: pass
    return scale, dt, s, V, T

def compute_eta_from_dat(datpath, scale=None, s=1, dt=0.001):
    # try to read numeric file, skip comments
    data = np.loadtxt(datpath, comments='#')
    if data.ndim ==1:
        data = data.reshape(-1, data.shape[0])
    # heuristics: if first column looks like time (in ps) or index, else use index
    ncols = data.shape[1]
    # choose columns for correlations: often columns 1..3 after time
    if ncols >=4:
        times = data[:,0]
        c1 = data[:,1]; c2 = data[:,2]; c3 = data[:,3]
        # if time looks integer indices starting 0..N-1, convert to real time
        if np.allclose(times, np.arange(len(times))):
            times = times * s * dt
    elif ncols==3:
        # no time column
        times = np.arange(data.shape[0]) * s * dt
        c1 = data[:,0]; c2=data[:,1]; c3=data[:,2]
    else:
        raise RuntimeError("S0St700.dat has too few columns: {}".format(ncols))
    if scale is None:
        raise RuntimeError("Scale factor required to convert integrated correlator to Pa.s")
    eta1 = np.trapz(c1, x=times) * scale
    eta2 = np.trapz(c2, x=times) * scale
    eta3 = np.trapz(c3, x=times) * scale
    return eta1, eta2, eta3, times, c1, c2, c3

def summarize_series(series, tail=50):
    n = len(series)
    if n==0:
        return None
    tail = min(tail, n)
    tail_vals = series[-tail:]
    return {
        'n': n,
        'last': series[-1],
        'tail_n': tail,
        'tail_mean': float(np.mean(tail_vals)),
        'tail_median': float(np.median(tail_vals)),
        'tail_std': float(np.std(tail_vals, ddof=1)) if len(tail_vals)>1 else 0.0,
        'mean': float(np.mean(series)),
        'std': float(np.std(series, ddof=1)) if len(series)>1 else 0.0
    }

# Run parsing on uploaded log
print("Parsing log:", LOGPATH)
hdr, rows = parse_log_for_thermo(LOGPATH)
if hdr is None or len(rows)==0:
    print("ERROR: thermo block with v_v11 etc not found in", LOGPATH)
else:
    parsed = compute_eta_from_rows(hdr, rows)
    stats = summarize_series(parsed['eta'], tail=50)
    print("Found thermo header:", hdr)
    print("Data points:", len(parsed['eta']))
    print("Indices for v_v11,v_v22,v_v33:", parsed['idxs'])
    print("Last eta (Pa.s):", stats['last'])
    print("Tail mean (last 50):", stats['tail_mean'], "Pa.s")
    print("Overall mean:", stats['mean'], "Pa.s")
    # save series CSV
    csv_out = OUTCSV
    with open(csv_out, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['step','eta_Pa_s'])
        for s,e in zip(parsed['steps'], parsed['eta']):
            w.writerow([s if s is not None else '', "{:.12e}".format(e)])
    print("Saved eta series to:", csv_out)

# Try to parse scale from log
scale, dt, svar, Vvar, Tvar = parse_scale_from_log(LOGPATH)
print("\nParsed scale from log (if found):", scale)
if dt is not None: print("Parsed dt:", dt)
if svar is not None: print("Parsed s:", svar)
if Vvar is not None: print("Parsed V:", Vvar)
if Tvar is not None: print("Parsed T:", Tvar)

# If S0St700.dat present, try to compute eta from it using parsed scale
datpath = os.path.join(os.path.dirname(LOGPATH), DATNAME)
if os.path.exists(datpath):
    print("\nFound", DATNAME, " — attempting integration using scale.")
    try:
        used_s = 1 if svar is None else svar
        used_dt = 0.001 if dt is None else dt
        if scale is None:
            # try to find numeric scale in log explicitly (search numeric literal)
            # fallback: ask user but here we abort with message
            raise RuntimeError("Scale not found in log; cannot compute from S0St700.dat without scale.")
        eta1, eta2, eta3, times, c1, c2, c3 = compute_eta_from_dat(datpath, scale=scale, s=used_s, dt=used_dt)
        print("Computed from S0St700.dat -> eta1,eta2,eta3 (Pa.s):", eta1, eta2, eta3)
        eta_avg = (eta1+eta2+eta3)/3.0
        print("Average eta from S0St700.dat:", eta_avg, "Pa.s")
    except Exception as e:
        print("Failed to compute from S0St700.dat:", e)
else:
    print("\nNo", DATNAME, "found in same folder. If you want integration from correlations, upload that file as well.")

print("\nDone.")

