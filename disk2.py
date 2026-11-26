"""disk_pipeline_v2.py
Revised protoplanetary-disk gap pipeline:
  1. Geometry (center / inclination / PA).
  2. Deprojection + azimuthal averaging.
  3. Multi-scale, significance-filtered DSHARP-style gap detection.
"""
from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass, field
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.modeling import models, fitting
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from scipy.optimize import least_squares
warnings.filterwarnings("ignore", category=FITSFixedWarning)
def load_fits_image(path: str):
    with fits.open(path) as hdul:
        data = np.squeeze(hdul[0].data)
        header = hdul[0].header
    if data.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {data.shape}.")
    wcs = WCS(header).celestial
    try:
        pixel_scale_arcsec = abs(wcs.wcs.cdelt[1]) * 3600.0
    except Exception:
        pixel_scale_arcsec = abs(header.get("CDELT2", np.nan)) * 3600.0
    bmaj = header.get("BMAJ", np.nan) * 3600.0
    bmin = header.get("BMIN", np.nan) * 3600.0
    beam_fwhm_arcsec = pixel_scale_arcsec if np.isnan(bmaj) or np.isnan(bmin) \
        else np.sqrt(bmaj * bmin)

    return data, header, wcs, pixel_scale_arcsec, beam_fwhm_arcsec
@dataclass
class Geometry:
    x0_pix: float
    y0_pix: float
    pa_deg: float
    inclination_deg: float
    theta_major_rad: float
def _robust_noise(data: np.ndarray) -> float:
    """Image rms from the median absolute deviation. Unlike a sigma-clipped
    std of the whole frame, this is not inflated by a bright disk that fills
    a sizeable part of the image."""
    f = data[np.isfinite(data)]
    if f.size == 0:
        return 0.0
    med = np.median(f)
    return float(1.4826 * np.median(np.abs(f - med)))


def fit_geometry(data: np.ndarray, beam_fwhm_pix: float = 3.0,
                 center_pix: tuple = None, nsigma: float = 5.0,
                 thresh_frac: float = 0.1) -> Geometry:
    """Robust centre / PA / inclination from image moments.

    1. Smooth the image by ~one beam and take its peak as a seed.
    2. Keep only the region connected to that peak above
       max(nsigma * rms, thresh_frac * peak) -- this excludes noise islands,
       distant companions and the faint halo that bias a global fit.
    3. Intensity-weighted (above threshold) centroid and second moments
       give the centre and the major/minor axes; inc = arccos(b/a).

    Unlike the old Gaussian2D fit there is no wide starting guess, no hard
    5-sigma truncation of the fitted pixels and no minimiser that can
    wander off to a false solution. Pass `center_pix=(x0, y0)` to pin the
    centre to a literature position; inc/PA still come from the moments
    unless the caller overrides them. Prefer literature inc/PA for the
    outer disk whenever they exist.
    """
    from scipy.ndimage import label, find_objects
    d = np.nan_to_num(data, nan=0.0)
    noise = _robust_noise(data)
    sigma_s = max(1.5, 0.5 * beam_fwhm_pix)
    sm = gaussian_filter(d, sigma_s)
    ipk = int(np.argmax(sm))
    ypk, xpk = np.unravel_index(ipk, sm.shape)
    peak = sm[ypk, xpk]

    thr = max(nsigma * noise, thresh_frac * peak)
    lab, _ = label(sm > thr)
    lab_pk = lab[ypk, xpk]
    comp = (lab == lab_pk) if lab_pk > 0 else (sm >= 0.5 * peak)
    box = find_objects(comp.astype(int))[0]
    ys, xs = np.mgrid[box]
    w = np.clip(sm[box] - thr, 0.0, None) * comp[box]
    W = w.sum()
    if W <= 0 or w.size < 5:
        xc, yc = float(xpk), float(ypk)
        if center_pix is not None:
            xc, yc = float(center_pix[0]), float(center_pix[1])
        return Geometry(x0_pix=xc, y0_pix=yc, pa_deg=0.0, inclination_deg=0.0,
                        theta_major_rad=np.radians(90.0))
    xc, yc = float((w * xs).sum() / W), float((w * ys).sum() / W)
    mxx = float((w * (xs - xc) ** 2).sum() / W)
    myy = float((w * (ys - yc) ** 2).sum() / W)
    mxy = float((w * (xs - xc) * (ys - yc)).sum() / W)
    tr, det = mxx + myy, mxx * myy - mxy ** 2
    disc = np.sqrt(max(tr ** 2 / 4 - det, 0.0))
    lam1, lam2 = tr / 2 + disc, max(tr / 2 - disc, 1e-12)
    theta_major = 0.5 * np.arctan2(2 * mxy, mxx - myy)   # major axis, from +x
    inc_deg = float(np.degrees(np.arccos(np.clip(np.sqrt(lam2 / lam1), 0, 1))))
    pa_deg = float((np.degrees(theta_major) - 90.0) % 180.0)

    if center_pix is not None:
        xc, yc = float(center_pix[0]), float(center_pix[1])
    return Geometry(x0_pix=xc, y0_pix=yc, pa_deg=pa_deg,
                    inclination_deg=inc_deg, theta_major_rad=float(theta_major))

@dataclass
class RadialProfile:
    r_arcsec: np.ndarray
    intensity: np.ndarray
    intensity_err: np.ndarray
    source: str  # "gofish" or "manual_fallback"


def _pixel_offset_to_arcsec_offset(wcs, x0_pix, y0_pix):
    """GoFish's x0/y0 are RA/Dec offsets (arcsec) from the FITS phase
    centre, not pixel coordinates -- convert the fitted pixel center here.
    Sign convention (RA increases to the left) matches GoFish's
    own internal handling, but ALWAYS sanity check against a quick plot the
    first time you run this on a new dataset."""
    # GoFish builds its position axes as (index - crpix + 1) * cdelt, i.e.
    # offsets from the FITS *reference pixel* (phase centre), NOT from the
    # array centre, and with the sign of cdelt (negative for RA, so +x points
    # east/left). Mirror that exactly. The old version measured from nx/2,
    # which is only right if the phase centre happens to sit at the middle.
    crpix1, crpix2 = wcs.wcs.crpix[0], wcs.wcs.crpix[1]
    m = wcs.pixel_scale_matrix
    sx = np.sign(m[0, 0]) or -1.0
    sy = np.sign(m[1, 1]) or 1.0
    sx_scale = np.hypot(m[0, 0], m[1, 0]) * 3600.0
    sy_scale = np.hypot(m[0, 1], m[1, 1]) * 3600.0
    dra = (x0_pix - (crpix1 - 1.0)) * sx * sx_scale
    ddec = (y0_pix - (crpix2 - 1.0)) * sy * sy_scale
    return dra, ddec


def radial_profile_gofish(fits_path: str, geom: Geometry, wcs, data_shape,
                           dr_arcsec: float = None) -> RadialProfile:
    from gofish import imagecube  # raises ImportError if not installed

    cube = imagecube(fits_path)
    x0_arcsec, y0_arcsec = _pixel_offset_to_arcsec_offset(wcs, geom.x0_pix, geom.y0_pix)

    x, y, dy = cube.radial_profile(
        x0=x0_arcsec, y0=y0_arcsec,
        inc=geom.inclination_deg, PA=geom.pa_deg,
        dr=dr_arcsec,               # None -> GoFish default (~1/4 beam)
        unit="Jy/beam",
        assume_correlated=True,     # accounts for beam-correlated pixels
    )
    return RadialProfile(r_arcsec=np.asarray(x), intensity=np.asarray(y),
                          intensity_err=np.asarray(dy), source="gofish")


def radial_profile_manual(data: np.ndarray, geom: Geometry,
                           pixel_scale_arcsec: float,
                           bin_width_arcsec: float) -> RadialProfile:
    ny, nx = data.shape
    if bin_width_arcsec <= 0 or pixel_scale_arcsec <= 0:
        raise ValueError("Pixel and radial-bin scales must be positive.")

    # The deprojected radius is largest at one of the image corners. This
    # avoids allocating full-image coordinate grids just to determine bins.
    corners_x = np.array([0, nx - 1, 0, nx - 1], dtype=float)
    corners_y = np.array([0, 0, ny - 1, ny - 1], dtype=float)
    corner_r, _ = _deproject_px(
        corners_x, corners_y, geom.x0_pix, geom.y0_pix,
        np.radians(geom.inclination_deg), geom.theta_major_rad)
    n_bins = max(1, int(np.ceil(np.max(corner_r) * pixel_scale_arcsec
                                / bin_width_arcsec)))
    centers = (np.arange(n_bins, dtype=float) + 0.5) * bin_width_arcsec
    counts = np.zeros(n_bins, dtype=np.int64)
    sums = np.zeros(n_bins, dtype=float)
    sums_sq = np.zeros(n_bins, dtype=float)

    inc, rot = np.radians(geom.inclination_deg), geom.theta_major_rad
    cos_rot, sin_rot = np.cos(rot), np.sin(rot)
    inv_cos_inc = 1.0 / max(np.cos(inc), 1e-3)
    x = np.arange(nx, dtype=float)[None, :]
    chunk_rows = max(1, min(ny, 256))
    for y0 in range(0, ny, chunk_rows):
        y1 = min(ny, y0 + chunk_rows)
        y = np.arange(y0, y1, dtype=float)[:, None]
        dx, dy = x - geom.x0_pix, y - geom.y0_pix
        x_p = dx * cos_rot + dy * sin_rot
        y_p = (-dx * sin_rot + dy * cos_rot) * inv_cos_inc
        radius_pix = np.hypot(x_p, y_p)
        values = data[y0:y1]
        valid = np.isfinite(values)
        bin_idx = np.floor(radius_pix * pixel_scale_arcsec
                            / bin_width_arcsec).astype(np.int64)
        valid &= bin_idx < n_bins
        idx = bin_idx[valid]
        val = values[valid]
        counts += np.bincount(idx, minlength=n_bins)
        sums += np.bincount(idx, weights=val, minlength=n_bins)
        sums_sq += np.bincount(idx, weights=val * val, minlength=n_bins)

    I = np.full(n_bins, np.nan)
    dI = np.full(n_bins, np.nan)
    occupied = counts > 0
    I[occupied] = sums[occupied] / counts[occupied]
    variance = np.maximum(sums_sq[occupied] / counts[occupied] - I[occupied] ** 2, 0.0)
    dI[occupied] = np.sqrt(variance / counts[occupied])
    return RadialProfile(r_arcsec=centers, intensity=I, intensity_err=dI,
                          source="manual_fallback")


def _deproject_px(x, y, x0, y0, inc_rad, rot):
    dx, dy = x - x0, y - y0
    x_p = dx * np.cos(rot) + dy * np.sin(rot)
    y_d = (-dx * np.sin(rot) + dy * np.cos(rot)) / max(np.cos(inc_rad), 1e-3)
    return np.hypot(x_p, y_d), np.arctan2(y_d, x_p)


def _ring_symmetry_cost(x, y, d, x0, y0, inc_rad, rot, r_lo, r_hi, dr):
    r, _ = _deproject_px(x, y, x0, y0, inc_rad, rot)
    m = (r > r_lo) & (r < r_hi)
    if m.sum() < 100:
        return np.inf
    k = ((r[m] - r_lo) / dr).astype(int)
    dm = d[m]
    n = np.bincount(k)
    s1 = np.bincount(k, dm)
    s2 = np.bincount(k, dm * dm)
    good = n > 1
    within = (s2[good] - s1[good] ** 2 / n[good]).sum()
    total = dm.var() * dm.size
    return within / max(total, 1e-30)


def _looks_like_ring_source(data: np.ndarray, geom: Geometry,
                            pixel_scale_arcsec: float,
                            beam_fwhm_arcsec: float,
                            min_peak_radius_beams: float = 1.5,
                            max_central_ratio: float = 0.9) -> bool:
    """Data-driven test for 'should I run ring refinement?'.

    A ring/cavity source (GG Tau, SY Cha, HD 163296's outer ring, ...) shows
    a resolved brightness peak at r > ~1.5 beams, with the innermost region
    of the deprojected profile at most ~90% of that peak. A centrally-peaked
    disk (HL Tau, TW Hya core, ...) has its maximum at r ~ 0 and fails both
    tests. Nothing here is keyed on object name.

    Uses the *manual* radial profile deliberately: it does not require
    gofish to be installed, and its binning is coarse enough that a single
    bright pixel cannot fake a ring.
    """
    try:
        prof = radial_profile_manual(data, geom, pixel_scale_arcsec,
                                     bin_width_arcsec=0.5 * beam_fwhm_arcsec)
    except Exception:
        return False
    r, I = prof.r_arcsec, prof.intensity
    ok = np.isfinite(I) & (r > 0)
    if ok.sum() < 5:
        return False
    r, I = r[ok], I[ok]

    peak_idx = int(np.argmax(I))
    r_peak = r[peak_idx]
    if r_peak < min_peak_radius_beams * beam_fwhm_arcsec:
        return False  # peak is at the center -> filled disk

    I_peak = I[peak_idx]
    if I_peak <= 0:
        return False

    # Ring peak must be a genuine local maximum, not an edge spike.
    if peak_idx in (0, len(I) - 1):
        return False
    if not (I[peak_idx] > I[peak_idx - 1] and I[peak_idx] > I[peak_idx + 1]):
        return False

    # Central bins: how bright is the core relative to the ring peak?
    central = I[r <= 0.5 * r_peak]
    if central.size == 0:
        return False
    if np.nanmedian(central) / I_peak > max_central_ratio:
        return False

    return True


def refine_ring_geometry(data, geom, pixel_scale_arcsec, beam_fwhm_arcsec,
                         fit_inc=True, fit_pa=True, max_shift_frac=0.8,
                         verbose=True):
    """Re-fit centre (and optionally inc/PA) with a *ring* model instead of a
    single 2-D Gaussian.  Meant for ring/cavity sources, where a filled
    Gaussian is the wrong shape.  Returns a NEW Geometry, or the input
    geometry unchanged if anything looks unsafe.

      1. coarse grid search over the centre for the most circular ring
         (robust to a poor starting centre, uses no model shape at all);
      2. bounded least-squares polish of
             I = B + A exp[-(r-r0)^2/(2 s^2)] (1 + a cos(phi) + b sin(phi))
         in deprojected polar coords.  The (a, b) term soaks up a lopsided
         ring so that brightness asymmetry is not traded against centre;
      3. the fit is repeated on a narrow r0 +/- 3.5 s annulus about the new
         centre, which also keeps compact central emission out of the fit.
    """
    ny, nx = data.shape
    beam_pix = beam_fwhm_arcsec / pixel_scale_arcsec
    inc0, rot0 = np.radians(geom.inclination_deg), geom.theta_major_rad

    # ring radius guess from the current geometry's azimuthal profile
    prof = radial_profile_manual(data, geom, pixel_scale_arcsec,
                                 bin_width_arcsec=0.5 * beam_fwhm_arcsec)
    ok = np.isfinite(prof.intensity) & (prof.r_arcsec > beam_fwhm_arcsec)
    if ok.sum() < 5:
        return geom
    r_peak = prof.r_arcsec[ok][np.argmax(prof.intensity[ok])] / pixel_scale_arcsec

    half = int(2.6 * r_peak)
    x_lo, x_hi = int(max(0, geom.x0_pix - half)), int(min(nx, geom.x0_pix + half))
    y_lo, y_hi = int(max(0, geom.y0_pix - half)), int(min(ny, geom.y0_pix + half))
    sub = data[y_lo:y_hi, x_lo:x_hi]
    sample_step = max(1, int(np.ceil(np.sqrt(sub.size / 100_000))))
    sampled = sub[::sample_step, ::sample_step]
    yy, xx = np.mgrid[y_lo:y_hi:sample_step, x_lo:x_hi:sample_step]
    fin = np.isfinite(sampled)
    xf, yf, df = xx[fin].astype(float), yy[fin].astype(float), sampled[fin]
    scale = np.nanmax(df)

    step = max(1.0, 0.5 * beam_pix)
    dr = max(1.0, 0.25 * beam_pix)
    cx, cy, inc_c, rot_c = geom.x0_pix, geom.y0_pix, inc0, rot0
    p = [cx, cy, inc0, rot0, r_peak, 0.2 * r_peak, 1.0, 0.0, 0.0, 0.0]
    d_inc, d_rot = np.radians(35.0), np.radians(90.0)
    converged = False

    for outer in range(4):
        # --- 1. coarse centre grid search (ring-circularity cost, inc/PA held) ---
        smax = (max_shift_frac if outer == 0 else 0.25) * r_peak
        offs = np.arange(-smax, smax + step, step)
        best = (np.inf, cx, cy)
        for oy in offs:
            for ox in offs:
                c = _ring_symmetry_cost(xf, yf, df, cx + ox, cy + oy, inc_c, rot_c,
                                        0.6 * r_peak, 1.4 * r_peak, dr)
                if c < best[0]:
                    best = (c, cx + ox, cy + oy)
        xc, yc = best[1], best[2]
        p[0], p[1], p[2], p[3] = xc, yc, inc_c, rot_c

        # --- 2. least-squares polish: wide annulus, then narrow r0 +/- 3.5 s ---
        for stage in range(2):
            r_map, _ = _deproject_px(xf, yf, p[0], p[1], p[2], p[3])
            if stage == 0:
                sel = (r_map > 0.5 * r_peak) & (r_map < 1.6 * r_peak)
            else:
                sel = np.abs(r_map - p[4]) < 3.5 * max(p[5], 0.5 * beam_pix / 2.3548)
            x, y, d = xf[sel], yf[sel], df[sel] / scale
            if d.size < 200:
                return geom
            sm = 0.3 * r_peak
            lo = [xc - sm, yc - sm, max(0.0, inc0 - d_inc), rot0 - d_rot,
                  0.5 * r_peak, 0.5 * beam_pix / 2.3548, 0.0, -0.5, -0.9, -0.9]
            hi = [xc + sm, yc + sm, min(np.radians(80.0), inc0 + d_inc), rot0 + d_rot,
                  1.5 * r_peak, r_peak, 5.0, 0.5, 0.9, 0.9]
            if not fit_inc:
                lo[2], hi[2] = inc0 - 1e-9, inc0 + 1e-9
            if not fit_pa:
                lo[3], hi[3] = rot0 - 1e-9, rot0 + 1e-9
            q0 = np.clip(p, np.array(lo) + 1e-12, np.array(hi) - 1e-12)

            def resid(q):
                r, phi = _deproject_px(x, y, q[0], q[1], q[2], q[3])
                return (q[7] + q[6] * np.exp(-(r - q[4]) ** 2 / (2 * q[5] ** 2))
                        * (1 + q[8] * np.cos(phi) + q[9] * np.sin(phi))) - d
            try:
                fit = least_squares(resid, q0, bounds=(lo, hi),
                                    x_scale=[r_peak, r_peak, 0.1, 0.1, r_peak, r_peak, 1, 0.1, 0.5, 0.5])
            except Exception as exc:
                if verbose:
                    print(f"[ring-refine] fit failed ({exc}); keeping original geometry.")
                return geom
            p = list(fit.x)

        moved = np.hypot(p[0] - cx, p[1] - cy)
        cx, cy, inc_c, rot_c = p[0], p[1], p[2], p[3]
        if moved < 0.2:
            converged = True
            break

    cost_start = _ring_symmetry_cost(xf, yf, df, geom.x0_pix, geom.y0_pix, inc0, rot0,
                                     0.6 * r_peak, 1.4 * r_peak, dr)
    cost_end = _ring_symmetry_cost(xf, yf, df, cx, cy, inc_c, rot_c,
                                   0.6 * r_peak, 1.4 * r_peak, dr)
    if not converged or not (0.5 * r_peak < p[4] < 1.5 * r_peak) or not cost_end < cost_start:
        if verbose:
            print("[ring-refine] no converged, more-circular ring solution; keeping original geometry.")
        return geom

    theta_major = p[3] if fit_pa else geom.theta_major_rad
    inclination = np.degrees(p[2]) if fit_inc else geom.inclination_deg
    new = Geometry(x0_pix=float(p[0]), y0_pix=float(p[1]),
                   pa_deg=float((np.degrees(theta_major) - 90.0) % 180.0),
                   inclination_deg=float(inclination),
                   theta_major_rad=float(theta_major))
    if verbose:
        shift = np.hypot(new.x0_pix - geom.x0_pix, new.y0_pix - geom.y0_pix)
        print(f"[ring-refine] centre ({geom.x0_pix:.1f},{geom.y0_pix:.1f}) -> ({new.x0_pix:.1f},{new.y0_pix:.1f}) pix "
              f"(shift {shift:.1f} pix = {shift * pixel_scale_arcsec:.3f}\"); "
              f"inc {geom.inclination_deg:.1f}->{new.inclination_deg:.1f}, PA {geom.pa_deg:.1f}->{new.pa_deg:.1f}; "
              f"ring r0={p[4] * pixel_scale_arcsec:.3f}\", lopsidedness={np.hypot(p[8], p[9]):.2f}")
    return new


def _ring_cost_for_geom(data, geom, pixel_scale_arcsec, beam_fwhm_arcsec):
    """Ring-circularity cost of `geom` around its own azimuthal-profile peak
    (lower = more circular ring). Returns (cost, r_peak_pix) or (inf, nan)."""
    prof = radial_profile_manual(data, geom, pixel_scale_arcsec,
                                 bin_width_arcsec=0.5 * beam_fwhm_arcsec)
    ok = np.isfinite(prof.intensity) & (prof.r_arcsec > beam_fwhm_arcsec)
    if ok.sum() < 5:
        return np.inf, np.nan
    r_peak = prof.r_arcsec[ok][np.argmax(prof.intensity[ok])] / pixel_scale_arcsec
    ny, nx = data.shape
    half = int(2.0 * r_peak)
    x_lo, x_hi = int(max(0, geom.x0_pix - half)), int(min(nx, geom.x0_pix + half))
    y_lo, y_hi = int(max(0, geom.y0_pix - half)), int(min(ny, geom.y0_pix + half))
    sub = data[y_lo:y_hi, x_lo:x_hi]
    sample_step = max(1, int(np.ceil(np.sqrt(sub.size / 100_000))))
    sampled = sub[::sample_step, ::sample_step]
    yy, xx = np.mgrid[y_lo:y_hi:sample_step, x_lo:x_hi:sample_step]
    fin = np.isfinite(sampled)
    dr = max(1.0, 0.25 * beam_fwhm_arcsec / pixel_scale_arcsec)
    cost = _ring_symmetry_cost(xx[fin].astype(float), yy[fin].astype(float), sampled[fin],
                               geom.x0_pix, geom.y0_pix,
                               np.radians(geom.inclination_deg), geom.theta_major_rad,
                               0.6 * r_peak, 1.4 * r_peak, dr)
    return cost, r_peak


def choose_geometry(data, geom0, pixel_scale_arcsec, beam_fwhm_arcsec,
                    fit_inc=True, fit_pa=True, center_fixed=False,
                    phase_center_pix=None, verbose=True):
    """Decide ring-vs-disk WITHOUT trusting a single, possibly wrong centre.

    The old logic classified the source from the profile at the Gaussian-fit
    centre. A centrally peaked disk seen from an off-centre position has a
    profile that is low at r=0 and peaks at r ~ offset, which passes every
    ring test -- so a bad centre made the source *look* like a ring, and the
    ring refinement then locked onto the bad centre.

    Ring refinement is started from independent image-based seeds and the
    FITS phase centre. A fit is accepted when the refined profile retains a
    resolved ring, or when a detected compact core is centered on the fitted
    position. The latter allows a bright ring with a dim central point source
    to be centered on that source without accepting off-center false rings.
    Returns (geometry, is_ring).
    """
    beam_pix = beam_fwhm_arcsec / pixel_scale_arcsec
    if center_fixed:           # literature centre: refine only if it is a ring
        seeds = [geom0]
    else:
        d = np.nan_to_num(data, nan=0.0)
        sm = gaussian_filter(d, max(1.5, 0.5 * beam_pix))
        ypk, xpk = np.unravel_index(int(np.argmax(sm)), sm.shape)
        seeds = [geom0]
        extra = [(float(xpk), float(ypk))]
        if phase_center_pix is not None:
            extra.append((float(phase_center_pix[0]), float(phase_center_pix[1])))
        for (sx, sy) in extra:
            if all(np.hypot(sx - s.x0_pix, sy - s.y0_pix) > 0.5 * beam_pix for s in seeds):
                seeds.append(Geometry(x0_pix=sx, y0_pix=sy, pa_deg=geom0.pa_deg,
                                      inclination_deg=geom0.inclination_deg,
                                      theta_major_rad=geom0.theta_major_rad))

    phase_ring_radius_pix = None
    if phase_center_pix is not None:
        phase_geom = Geometry(float(phase_center_pix[0]), float(phase_center_pix[1]),
                              geom0.pa_deg, geom0.inclination_deg,
                              geom0.theta_major_rad)
        if _looks_like_ring_source(data, phase_geom, pixel_scale_arcsec,
                                   beam_fwhm_arcsec):
            phase_profile = radial_profile_manual(
                data, phase_geom, pixel_scale_arcsec,
                bin_width_arcsec=0.5 * beam_fwhm_arcsec)
            radial = phase_profile.r_arcsec
            brightness = phase_profile.intensity
            annulus = np.isfinite(brightness) & (radial > 1.5 * beam_fwhm_arcsec)
            if annulus.any():
                phase_ring_radius_pix = (
                    radial[annulus][np.argmax(brightness[annulus])] / pixel_scale_arcsec)

    core_smooth = gaussian_filter(np.nan_to_num(data), max(1.0, 0.4 * beam_pix))
    results = []   # (cost, geometry, r_peak_pix)
    for s in seeds:
        c_start, _ = _ring_cost_for_geom(data, s, pixel_scale_arcsec, beam_fwhm_arcsec)
        new = refine_ring_geometry(data, s, pixel_scale_arcsec, beam_fwhm_arcsec,
                                   fit_inc=fit_inc, fit_pa=fit_pa, verbose=False)
        if new is s:
            continue                                   # refinement rejected itself
        c_end, r_pk = _ring_cost_for_geom(data, new, pixel_scale_arcsec, beam_fwhm_arcsec)
        if not np.isfinite(c_end) or not c_end < c_start:
            continue
        if phase_ring_radius_pix is not None:
            phase_shift = np.hypot(new.x0_pix - phase_center_pix[0],
                                   new.y0_pix - phase_center_pix[1])
            if phase_shift > max(0.5 * phase_ring_radius_pix, 2.0 * beam_pix):
                continue
        if not _looks_like_ring_source(data, new, pixel_scale_arcsec, beam_fwhm_arcsec):
            radius = max(2.0 * beam_pix, 0.45 * r_pk if np.isfinite(r_pk) else 0.0)
            core = core_centroid(data, beam_pix, near=(new.x0_pix, new.y0_pix),
                                 radius_pix=radius)
            x0, y0 = int(round(new.x0_pix)), int(round(new.y0_pix))
            half = int(np.ceil(radius))
            xlo, xhi = max(0, x0 - half), min(data.shape[1], x0 + half + 1)
            ylo, yhi = max(0, y0 - half), min(data.shape[0], y0 + half + 1)
            yy, xx = np.mgrid[ylo:yhi, xlo:xhi]
            sub = core_smooth[ylo:yhi, xlo:xhi]
            in_core = np.hypot(xx - new.x0_pix, yy - new.y0_pix) <= radius
            core_peak = np.max(sub[in_core]) if np.any(in_core) else 0.0
            if core_peak < 5.0 * _robust_noise(data) or \
                    np.hypot(core[0] - new.x0_pix, core[1] - new.y0_pix) > beam_pix:
                continue
        results.append((c_end, new, r_pk))

    if not results:
        if verbose:
            print("[geometry] no seed gave a verified ring solution -> treating as a "
                  "disk and keeping the moment-based geometry.")
        return geom0, False

    results.sort(key=lambda t: t[0])
    best_cost, best, r_pk = results[0]
    spread = max(np.hypot(g.x0_pix - best.x0_pix, g.y0_pix - best.y0_pix)
                 for _, g, _ in results)
    if verbose:
        print(f"[geometry] ring verified from {len(results)}/{len(seeds)} seed(s); "
              f"centre ({best.x0_pix:.1f},{best.y0_pix:.1f}) pix, "
              f"ring r0={r_pk * pixel_scale_arcsec:.3f}\"")
        if spread > 0.5 * r_pk:
            print(f"[geometry] WARNING: ring solutions from different seeds disagree "
                  f"by up to {spread:.0f} pix (> r0/2). Check the centre by eye or "
                  f"pass the literature centre with --center-pix / --center-radec.")
    return best, True


def core_centroid(data: np.ndarray, beam_fwhm_pix: float,
                  near: tuple = None, radius_pix: float = None) -> tuple:
    """Centroid (x, y) of the compact central emission: intensity-weighted
    centre of the pixels above half the local peak within ~2 beams of the
    brightest smoothed pixel. For a ring/cavity source this is the central
    point source (the star/inner disk), which need not coincide with the
    ring's geometric centre.

    The ring is often BRIGHTER than the core, so the global maximum would
    land on the ring. Pass `near=(x, y)` (the ring centre) and `radius_pix`
    (inside the ring, e.g. ~0.45 of the ring radius) to search only the
    cavity."""
    d = np.nan_to_num(data, nan=0.0)
    sm = gaussian_filter(d, max(1.0, 0.4 * beam_fwhm_pix))
    if near is not None and radius_pix is not None:
        yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        search = np.where(np.hypot(xx - near[0], yy - near[1]) <= radius_pix, sm, -np.inf)
    else:
        search = sm
    ypk, xpk = np.unravel_index(int(np.argmax(search)), sm.shape)
    half = int(np.ceil(2.0 * beam_fwhm_pix))
    y_lo, y_hi = max(0, ypk - half), min(d.shape[0], ypk + half + 1)
    x_lo, x_hi = max(0, xpk - half), min(d.shape[1], xpk + half + 1)
    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    sub = sm[y_lo:y_hi, x_lo:x_hi]
    w = np.clip(sub - 0.5 * sm[ypk, xpk], 0.0, None)
    W = w.sum()
    if W <= 0:
        return float(xpk), float(ypk)
    return float((w * xx).sum() / W), float((w * yy).sum() / W)



def find_companions(data: np.ndarray, geom: Geometry, beam_fwhm_pix: float,
                    noise: float = None, min_peak_sigma: float = 15.0,
                    max_azimuth_span_deg: float = 150.0,
                    separation_factor: float = 1.3) -> list:
    """Locate compact, spatially separate sources (e.g. the AS 205 S binary
    companion) that must NOT enter the azimuthal average of the primary disk.

    A smoothed >5-sigma component is flagged as a companion only if ALL hold:
      * it is disconnected from the component containing the disk centre;
      * its peak is >= `min_peak_sigma` (a real source, not a noise island);
      * it lies beyond `separation_factor` x the primary component's
        furthest extent (so it cannot be an outer part of the same disk);
      * it subtends < `max_azimuth_span_deg` about the disk centre -- a ring
        or outer-disk fragment wraps around the star, a companion does not.
    Returns a list of dicts (x, y, sep_pix, peak_sigma, mask) where `mask`
    is the boolean footprint (2-sigma halo, dilated by ~2 beams).
    """
    from scipy.ndimage import label, binary_dilation
    d = np.nan_to_num(data, nan=0.0)
    if noise is None:
        noise = _robust_noise(data)
    if noise <= 0:
        return []
    sm = gaussian_filter(d, max(1.5, 0.5 * beam_fwhm_pix))
    lab5, n5 = label(sm > 5.0 * noise)
    if n5 < 2:
        return []
    ix, iy = int(round(geom.x0_pix)), int(round(geom.y0_pix))
    ix = min(max(ix, 0), d.shape[1] - 1)
    iy = min(max(iy, 0), d.shape[0] - 1)
    # primary = component containing the centre, else the nearest one
    main = lab5[iy, ix]
    if main == 0:
        yy, xx = np.nonzero(lab5 > 0)
        main = lab5[yy[np.argmin(np.hypot(xx - ix, yy - iy))],
                    xx[np.argmin(np.hypot(xx - ix, yy - iy))]]
    my, mx = np.nonzero(lab5 == main)
    main_extent = float(np.max(np.hypot(mx - geom.x0_pix, my - geom.y0_pix)))
    lab2, _ = label(sm > 2.0 * noise)
    main_footprint = lab2 == lab2[my[0], mx[0]] if lab2[my[0], mx[0]] > 0 else (lab5 == main)
    min_area = max(20, int(0.5 * beam_fwhm_pix ** 2))
    out = []
    for k in range(1, n5 + 1):
        if k == main:
            continue
        comp = lab5 == k
        if comp.sum() < min_area:
            continue
        peak_sigma = float(sm[comp].max() / noise)
        if peak_sigma < min_peak_sigma:
            continue
        cy, cx = np.nonzero(comp)
        dx, dy = cx - geom.x0_pix, cy - geom.y0_pix
        sep = float(np.hypot(dx.mean(), dy.mean()))
        if sep < separation_factor * main_extent:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        # azimuthal span = 360 - largest empty angular gap
        a = np.sort(ang)
        gaps_a = np.diff(np.r_[a, a[0] + 360.0])
        span = 360.0 - float(gaps_a.max())
        if span > max_azimuth_span_deg:
            continue
        footprint = lab2 == lab2[cy[0], cx[0]]
        footprint = binary_dilation(footprint, iterations=int(np.ceil(2.0 * beam_fwhm_pix)))
        footprint &= ~binary_dilation(lab5 == main, iterations=int(np.ceil(1.0 * beam_fwhm_pix)))
        out.append(dict(x=float(cx.mean()), y=float(cy.mean()), sep_pix=sep,
                        peak_sigma=peak_sigma, azimuth_span_deg=span, mask=footprint))
    return out


def build_radial_profile(fits_path: str, data: np.ndarray, geom: Geometry,
                          wcs, pixel_scale_arcsec: float,
                          beam_fwhm_arcsec: float) -> RadialProfile:
    try:
        prof = radial_profile_gofish(fits_path, geom, wcs, data.shape,
                                      dr_arcsec=0.25 * beam_fwhm_arcsec)
        # Cross-check against the self-contained deprojection on the SAME
        # geometry. A wrong centre/sign convention in the GoFish call is
        # invisible in the image overlay (which draws the pixel centre, not
        # what GoFish used), but it shows up here as a profile mismatch.
        ref = radial_profile_manual(data, geom, pixel_scale_arcsec,
                                    bin_width_arcsec=0.25 * beam_fwhm_arcsec)
        ok = np.isfinite(ref.intensity) & np.isfinite(ref.r_arcsec)
        okp = np.isfinite(prof.intensity)
        if ok.sum() > 5 and okp.sum() > 5:
            rr = ref.r_arcsec[ok]
            sel = (rr > beam_fwhm_arcsec) & (rr < prof.r_arcsec[okp].max())
            if sel.sum() > 5:
                g_i = np.interp(rr[sel], prof.r_arcsec[okp], prof.intensity[okp])
                scale = max(np.nanmax(np.abs(ref.intensity[ok])), 1e-30)
                mismatch = np.nanmedian(np.abs(g_i - ref.intensity[ok][sel])) / scale
                if mismatch > 0.05:
                    print(f"[profile] WARNING: GoFish profile disagrees with the "
                          f"manual deprojection on the same geometry "
                          f"(median |diff| = {100 * mismatch:.0f}% of peak) -- "
                          f"the GoFish centre/sign convention is probably off. "
                          f"Using the manual profile instead.")
                    return ref
        print("[profile] using GoFish radial_profile() (agrees with manual check).")
        return prof
    except ImportError:
        print("[profile] `gofish` not installed -- falling back to manual "
              "deprojection. `pip install gofish` to use the requested "
              "workflow as specified.")
        return radial_profile_manual(data, geom, pixel_scale_arcsec,
                                      bin_width_arcsec=0.25 * beam_fwhm_arcsec)


@dataclass
class Gap:
    r_gap_arcsec: float
    r_gap_err_arcsec: float
    width_arcsec: float
    depth: float
    contrast: float  # I_gap / brighter adjacent ring peak
    ring_ref_arcsec: float
    kind: str
    reliable: bool
    reference_intensity: float = np.nan
    bracket_inner_arcsec: float = np.nan
    bracket_outer_arcsec: float = np.nan
    significance: float = np.nan
    azimuth_fraction: float = np.nan
    scales_detected: int = 0
    tier: str = "firm"
    cavity_ring_peak_arcsec: float = np.nan
    cavity_ring_fwhm_arcsec: float = np.nan
    cavity_radius_arcsec: float = np.nan
    cavity_radius_err_arcsec: float = np.nan
    cavity_resolution_limited: bool = False


_KNOWN_CAVITY_SOURCE_MARKERS = (
    "gm_aur", "lkca_15", "dm_tau", "ux_tau_a", "pds_70",
    "2mass_j16042165", "mwc758", "hd169142",
)


def _known_cavity_source(fits_path):
    # These user-identified transition disks can be unresolved at this beam.
    source_name = fits_path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return any(marker in source_name for marker in _KNOWN_CAVITY_SOURCE_MARKERS)


def _crossing_radius(r, I, i_from, i_to, I_target):
    """Linear-interpolated radius where the profile between index i_from and
    i_to crosses I_target -- used for Huang et al. (2018)'s
    I_mean = 0.5*(I_dark + I_bright) edge definition."""
    lo, hi = (i_from, i_to) if i_from < i_to else (i_to, i_from)
    seg_r, seg_I = r[lo:hi + 1], I[lo:hi + 1]
    if len(seg_I) < 2:
        return r[i_from]
    diffs = seg_I - I_target
    sign_change = np.where(np.diff(np.sign(diffs)) != 0)[0]
    if len(sign_change) == 0:
        return seg_r[np.argmin(np.abs(diffs))]
    k = sign_change[0]
    r1, r2 = seg_r[k], seg_r[k + 1]
    I1, I2 = seg_I[k], seg_I[k + 1]
    frac = 0.0 if I2 == I1 else (I_target - I1) / (I2 - I1)
    return r1 + frac * (r2 - r1)



def _noise_floor(r, I, err, source, beam_fwhm_arcsec):
    """Estimate a profile noise floor without mistaking the radial taper for noise."""
    good = np.isfinite(err) & (err > 0)
    floor = float(np.median(err[good])) if good.any() else 0.0
    if source == "gofish":
        return max(floor, 1e-30)

    dr = float(np.median(np.diff(r)))
    smoothed = gaussian_filter1d(
        I, max(0.5, 0.5 * beam_fwhm_arcsec / max(dr, 1e-12)))
    residual = I - smoothed
    residual_mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    return max(floor, float(residual_mad), 1e-30)


def _detect_one_scale(r, I, err, beam, smooth_beams, min_sigma, min_depth_frac,
                      max_gap_radius):
    dr = float(np.median(np.diff(r)))
    s = gaussian_filter1d(I, max(0.5, smooth_beams * beam / dr))
    min_flank = max(dr, 0.25 * beam)
    sep = max(1, int(np.ceil(min_flank / dr)))
    minima, _ = find_peaks(-s, distance=sep)
    maxima, _ = find_peaks(s, distance=sep)
    if s[0] > s[1]:
        maxima = np.unique(np.append(maxima, 0))
    found = []
    for m in minima:
        L, R = maxima[maxima < m], maxima[maxima > m]
        if not L.size or not R.size:
            continue
        l = int(L[-1])
        rt = int(R[0])
        if r[m] - r[l] < min_flank or r[rt] - r[m] < min_flank:
            continue
        if max_gap_radius is not None and r[rt] > max_gap_radius:
            continue
        g, lp, rp = float(s[m]), float(s[l]), float(s[rt])
        ring_ref = l if lp >= rp else rt
        rim_ref = l if lp <= rp else rt
        ring, rim = max(lp, rp), min(lp, rp)
        if ring <= 0 or rim <= 0:
            continue
        depth = ring - g
        local_depth = rim - g
        denom = np.sqrt(err[m] ** 2 + err[rim_ref] ** 2)
        sig = local_depth / max(denom, 1e-30)
        if local_depth / rim < min_depth_frac:
            continue
        half = 0.5 * (g + rim)
        r_in = _crossing_radius(r, s, l, int(m), half)
        r_out = _crossing_radius(r, s, int(m), rt, half)
        w = abs(r_out - r_in)
        if not np.isfinite(w) or w <= 0:
            continue
        found.append(dict(m=int(m), l=l, rt=rt, w=w, g=g, rim=rim, base=ring,
                          sig=float(sig),
                          ring_ref=float(r[ring_ref]), top=ring))
    return found


def _sector_fraction(gap, sectors, beam, smooth_beams, min_dip_sigma):
    """Fraction of azimuthal sectors where the radial dip is independently present.
    sectors is a list of (r, I, sigma_sector) tuples."""
    if not sectors:
        return np.nan
    ok = 0
    for rs, Is, sg in sectors:
        rs, Is = np.asarray(rs, float), np.asarray(Is, float)
        good = np.isfinite(rs) & np.isfinite(Is)
        rs, Is = rs[good], Is[good]
        if rs.size < 3 or not np.isfinite(sg) or sg <= 0:
            continue
        order = np.argsort(rs)
        rs, Is = rs[order], Is[order]
        dr = float(np.median(np.diff(rs)))
        if dr <= 0:
            continue
        smoothed = gaussian_filter1d(np.nan_to_num(Is), max(0.5, smooth_beams * beam / dr))
        a = np.interp(gap.bracket_inner_arcsec, rs, smoothed)
        b = np.interp(gap.bracket_outer_arcsec, rs, smoothed)
        lo, hi = sorted((gap.bracket_inner_arcsec, gap.bracket_outer_arcsec))
        win = (rs >= lo) & (rs <= hi)
        if not win.any():
            continue
        base = np.interp(gap.r_gap_arcsec,
                         [gap.bracket_inner_arcsec, gap.bracket_outer_arcsec], [a, b])
        dip = base - float(np.min(smoothed[win]))
        if dip > 0 and dip >= min_dip_sigma * np.sqrt(2.0) * sg:
            ok += 1
    return ok / len(sectors)


def _peak_fwhm(r, intensity, peak_index):
    """Measure a peak's half-maximum width using its nearest crossings."""
    peak = float(intensity[peak_index])
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    half_peak = 0.5 * peak

    left = np.flatnonzero(intensity[:peak_index] <= half_peak)
    right = np.flatnonzero(intensity[peak_index + 1:] <= half_peak)
    if not left.size or not right.size:
        return np.nan

    i_left = int(left[-1])
    i_right = peak_index + 1 + int(right[0])

    def crossing(i1, i2):
        y1, y2 = intensity[i1], intensity[i2]
        if y2 == y1:
            return float(r[i1])
        return float(r[i1] + (half_peak - y1) * (r[i2] - r[i1]) / (y2 - y1))

    return abs(crossing(i_left, i_left + 1) - crossing(i_right - 1, i_right))


def _cavity_feature(r, intensity, errors, beam_fwhm_arcsec, smooth,
                    cavity_min_depth_frac, cavity_min_span_beams,
                    cavity_hint=False, max_radius_arcsec=None):
    """Measure a central clearing and its outer ring from a radial profile."""
    dr = float(np.median(np.diff(r)))
    inner_peak_limit = max(0.5 * beam_fwhm_arcsec, dr)
    outer_peaks, _ = find_peaks(smooth)
    outer_peaks = outer_peaks[r[outer_peaks] >= inner_peak_limit]
    if max_radius_arcsec is not None:
        outer_peaks = outer_peaks[r[outer_peaks] <= max_radius_arcsec]
    if not outer_peaks.size:
        return None

    peak_index = int(outer_peaks[np.argmax(smooth[outer_peaks])])
    peak_intensity = float(smooth[peak_index])
    if peak_intensity <= 0:
        return None

    center_intensity = max(0.0, float(smooth[0]))
    if center_intensity >= peak_intensity:
        return None
    contrast = center_intensity / peak_intensity
    ring_radius = float(r[peak_index])
    half_clearance = center_intensity + 0.5 * (peak_intensity - center_intensity)
    below_half = np.flatnonzero(smooth[:peak_index] < half_clearance)
    if below_half.size:
        inner_index = int(below_half[-1])
        cavity_radius = _crossing_radius(
            r, smooth, inner_index, min(inner_index + 1, peak_index), half_clearance)
    else:
        cavity_radius = float(r[0])

    resolved_by_profile = (
        ring_radius >= cavity_min_span_beams * beam_fwhm_arcsec
        and cavity_radius >= beam_fwhm_arcsec
        and contrast <= 1.0 - cavity_min_depth_frac
    )
    hinted_source = cavity_hint and ring_radius >= inner_peak_limit
    if not resolved_by_profile and not hinted_source:
        return None

    center_error = float(errors[0])
    peak_error = float(errors[peak_index])
    significance = ((peak_intensity - center_intensity)
                    / max(np.hypot(center_error, peak_error), 1e-30))
    radius_error = max(0.5 * beam_fwhm_arcsec, 0.5 * dr)
    ring_fwhm = _peak_fwhm(r, smooth, peak_index)
    confidently_resolved = resolved_by_profile and significance >= 5.0
    return Gap(
        r_gap_arcsec=float(cavity_radius),
        r_gap_err_arcsec=float(radius_error),
        width_arcsec=float(ring_fwhm) if np.isfinite(ring_fwhm) else np.nan,
        depth=float(peak_intensity - center_intensity),
        contrast=float(contrast),
        ring_ref_arcsec=ring_radius,
        kind="cavity",
        reliable=confidently_resolved,
        reference_intensity=peak_intensity,
        bracket_inner_arcsec=0.0,
        bracket_outer_arcsec=ring_radius,
        significance=float(significance),
        tier="firm" if confidently_resolved else "tentative",
        cavity_ring_peak_arcsec=ring_radius,
        cavity_ring_fwhm_arcsec=float(ring_fwhm) if np.isfinite(ring_fwhm) else np.nan,
        cavity_radius_arcsec=float(cavity_radius),
        cavity_radius_err_arcsec=float(radius_error),
        cavity_resolution_limited=not resolved_by_profile,
    )


def find_gaps_robust(profile, beam_fwhm_arcsec, sectors=None,
                     smooth_beams=0.1, min_depth_frac=0.01, min_sigma=3.0,
                     firm_sigma=5.0, firm_azimuth_frac=0.75,
                     reject_azimuth_frac=0.5, sector_dip_sigma=1.0,
                     max_gap_radius_arcsec=None, cavity_min_depth_frac=0.5,
                     cavity_min_span_beams=2.0,
                     cavity_center_bracket_beams=1.0, cavity_hint=False):
    """Find annular gaps and measure central cavities separately.

    Dips must be significant relative to both neighboring peaks, so a fading
    outer edge cannot qualify solely by comparison with a much brighter inner
    peak. Candidate dips are checked at several smoothing scales and merged
    when they refer to the same feature.
    """
    r = np.asarray(profile.r_arcsec, float)
    I = np.asarray(profile.intensity, float)
    err = np.asarray(profile.intensity_err, float)
    v = np.isfinite(r) & np.isfinite(I)
    r, I, err = r[v], I[v], err[v]
    if r.size < 8:
        return []
    order = np.argsort(r)
    r, I, err = r[order], I[order], err[order]
    err = np.maximum(np.where(np.isfinite(err), err, 0.0),
                     _noise_floor(r, I, err, profile.source, beam_fwhm_arcsec))
    if not np.all(np.diff(r) > 0):
        unique_r, unique_idx = np.unique(r, return_index=True)
        r, I, err = unique_r, I[unique_idx], err[unique_idx]
    if r.size < 8 or np.any(np.diff(r) <= 0):
        return []

    dr = float(np.median(np.diff(r)))
    smoothed = gaussian_filter1d(
        I, max(0.5, 0.5 * beam_fwhm_arcsec / max(dr, 1e-12)))
    cavity = _cavity_feature(
        r, I, err, beam_fwhm_arcsec, smoothed, cavity_min_depth_frac,
        cavity_min_span_beams, cavity_hint=cavity_hint,
        max_radius_arcsec=max_gap_radius_arcsec)

    gap_radius_limit = max_gap_radius_arcsec
    if cavity is not None and gap_radius_limit is not None:
        if gap_radius_limit > 10.0 * cavity.cavity_ring_peak_arcsec:
            gap_radius_limit = 8.0 * cavity.cavity_ring_peak_arcsec

    scales = sorted({float(smooth_beams), float(2.0 * smooth_beams),
                     float(4.0 * smooth_beams)})
    candidates = []
    for scale in scales:
        for candidate in _detect_one_scale(
                r, I, err, beam_fwhm_arcsec, scale, min_sigma,
                min_depth_frac, gap_radius_limit):
            candidate["scale"] = scale
            candidates.append(candidate)

    clusters = []
    for candidate in sorted(candidates, key=lambda item: r[item["m"]]):
        match = next((cluster for cluster in clusters
                      if abs(r[candidate["m"]] - r[cluster[0]["m"]])
                      <= 0.75 * beam_fwhm_arcsec), None)
        if match is None:
            clusters.append([candidate])
        else:
            match.append(candidate)

    gaps = [cavity] if cavity is not None else []
    for cluster in clusters:
        f = max(cluster, key=lambda item: item["sig"])
        detected_scales = {item["scale"] for item in cluster}
        if f["sig"] < min_sigma and len(detected_scales) < 2:
            continue
        m, l, rt = f["m"], f["l"], f["rt"]
        gap = Gap(r_gap_arcsec=float(r[m]), r_gap_err_arcsec=0.5 * f["w"],
                  width_arcsec=f["w"], depth=f["base"] - f["g"],
                  contrast=max(0.0, f["g"] / f["top"]),
                  ring_ref_arcsec=f["ring_ref"],
                  kind="local_minimum", reliable=False, reference_intensity=f["top"],
                  bracket_inner_arcsec=float(r[l]), bracket_outer_arcsec=float(r[rt]),
                  significance=float(f["sig"]),
                  scales_detected=len(detected_scales))
        tier = "firm" if f["sig"] >= firm_sigma else "tentative"
        if gap.width_arcsec < beam_fwhm_arcsec:
            tier = "tentative"
        if sectors:
            gap.azimuth_fraction = _sector_fraction(
                gap, sectors, beam_fwhm_arcsec, smooth_beams, sector_dip_sigma)
            if np.isfinite(gap.azimuth_fraction):
                if gap.azimuth_fraction < reject_azimuth_frac:
                    continue
                if gap.azimuth_fraction < firm_azimuth_frac:
                    tier = "tentative"
        central_bracket = gap.bracket_inner_arcsec <= (
            cavity_center_bracket_beams * beam_fwhm_arcsec)
        broad_clearing = (gap.bracket_outer_arcsec - gap.bracket_inner_arcsec
                          >= cavity_min_span_beams * beam_fwhm_arcsec)
        deep_clearing = gap.contrast <= 1.0 - cavity_min_depth_frac
        if cavity is not None and central_bracket and broad_clearing:
            continue
        central_clearing = smoothed[0] <= (
            (1.0 - cavity_min_depth_frac) * smoothed[rt])
        if central_bracket and broad_clearing and deep_clearing and central_clearing:
            continue
        gap.tier, gap.reliable = tier, tier == "firm"
        gaps.append(gap)
    return gaps


def estimate_image_noise(data: np.ndarray, header, region: tuple = None) -> float:
    """Image rms. Order of preference: header RMS keyword, an explicit
    emission-free `region` (y0, y1, x0, x1), else a MAD-based estimate over
    the whole frame (robust to a disk that covers part of the image; the
    old sigma-clipped std of the whole frame was inflated by the disk)."""
    if "RMS" in header:
        return float(header["RMS"])
    sample = data if region is None else data[region[0]:region[1], region[2]:region[3]]
    return _robust_noise(sample)


def _max_fully_sampled_radius_arcsec(geom: Geometry, image_shape: tuple,
                                     pixel_scale_arcsec: float) -> float:
    """Largest radius whose major-axis points remain inside the image."""
    ny, nx = image_shape
    ux, uy = np.cos(geom.theta_major_rad), np.sin(geom.theta_major_rad)
    distances = []
    for dx, dy in ((ux, uy), (-ux, -uy)):
        tx = ((nx - 1 - geom.x0_pix) / dx if dx > 0 else
              (0 - geom.x0_pix) / dx if dx < 0 else np.inf)
        ty = ((ny - 1 - geom.y0_pix) / dy if dy > 0 else
              (0 - geom.y0_pix) / dy if dy < 0 else np.inf)
        distances.append(min(tx, ty))
    return max(0.0, min(distances) * pixel_scale_arcsec)


def estimate_disk_radius(profile: RadialProfile, noise_level: float,
                          beam_fwhm_arcsec: float, threshold_sigma: float = 5.0,
                          flux_fraction: float = 0.95,
                          max_radius_arcsec: float = None):
    """Outer radius enclosing `flux_fraction` (default 95%) of the disk flux.

    This is the standard "R95" definition and, unlike a threshold on a
    noisy tail, does not jump around with the last above-noise bin.

      * The flux integral 2*pi*r*I(r) dr runs from the centre out to the
        end of the last sustained, significant emission region. Looking for
        the last region (rather than the first threshold crossing after the
        brightest bin) prevents a bright compact core from truncating a
        fainter outer ring at the cavity.
      * That noise is taken from the profile itself (median error bar and
        the scatter of its outermost bins), NOT from the single-pixel image
        rms, which overstates the noise of an azimuthal average.
    Returns (radius_arcsec, err_arcsec).
    """
    from scipy.ndimage import gaussian_filter1d
    r_all, I_all, dI_all = profile.r_arcsec, profile.intensity, profile.intensity_err
    profile_valid = np.isfinite(I_all)
    I_for_noise = I_all[profile_valid]
    dI_for_noise = dI_all[profile_valid]
    if len(I_for_noise) < 5:
        return (float(r_all[profile_valid][-1]) if np.any(profile_valid) else 0.0), 0.0

    n_tail = max(5, len(I_for_noise) // 5)
    tail = I_for_noise[-n_tail:]
    tail_median = np.nanmedian(tail)
    tail_noise = 1.4826 * np.nanmedian(np.abs(tail - tail_median))
    finite_err = dI_for_noise[np.isfinite(dI_for_noise) & (dI_for_noise > 0)]
    profile_err = float(np.nanmedian(finite_err)) if finite_err.size else 0.0
    sigma_prof = max(profile_err, float(tail_noise), 1e-30)

    r, I, dI = r_all, I_all, dI_all
    valid = np.isfinite(I)
    if max_radius_arcsec is not None:
        valid &= r <= max_radius_arcsec
    r, I, dI = r[valid], I[valid], dI[valid]
    if len(r) < 5:
        return (float(r[-1]) if len(r) else 0.0), 0.0
    dr = float(np.nanmedian(np.diff(r)))

    Is = gaussian_filter1d(I, max(1.0, 0.5 * beam_fwhm_arcsec / max(dr, 1e-12)))

    significant = Is >= threshold_sigma * sigma_prof
    min_run = max(3, int(np.ceil(1.5 * beam_fwhm_arcsec / max(dr, 1e-12))))
    runs = []
    start = None
    for i, is_significant in enumerate(np.r_[significant, False]):
        if is_significant and start is None:
            start = i
        elif not is_significant and start is not None:
            if i - start >= min_run:
                runs.append((start, i - 1))
            start = None
    if runs:
        i_end = runs[-1][1]
    else:
        i_end = int(np.argmax(Is))
    i_end = max(i_end, min(len(r) - 1, int(np.argmax(Is)) + 2))

    flux = np.cumsum(np.clip(I[:i_end + 1], 0.0, None) * 2.0 * np.pi * r[:i_end + 1] * dr)
    if flux[-1] <= 0:
        return float(r[i_end]), dr / 2.0
    k = int(np.searchsorted(flux, flux_fraction * flux[-1]))
    k = min(k, i_end)
    return float(r[k]), float(max(dr / 2.0, 0.5 * beam_fwhm_arcsec))


def estimate_outer_ring_edge(profile: RadialProfile, beam_fwhm_arcsec: float,
                             threshold_sigma: float = 3.0,
                             min_ring_ratio: float = 0.15,
                             edge_fraction: float = 0.15,
                             min_peak_fraction: float = 0.01):
    """Estimate the edge beyond the last coherent ring in a multi-ring disk.

    Faint outer-profile bumps are not treated as new rings unless they reach
    at least ``min_ring_ratio`` of the preceding ring peak and
    ``min_peak_fraction`` of the profile maximum. The edge is the sustained
    outward crossing of ``edge_fraction`` of the last accepted ring peak (or
    the profile-noise threshold, if higher).
    """
    from scipy.ndimage import gaussian_filter1d

    r = np.asarray(profile.r_arcsec, dtype=float)
    intensity = np.asarray(profile.intensity, dtype=float)
    errors = np.asarray(profile.intensity_err, dtype=float)
    valid = np.isfinite(r) & np.isfinite(intensity)
    r, intensity, errors = r[valid], intensity[valid], errors[valid]
    if r.size < 5:
        return None
    order = np.argsort(r)
    r, intensity, errors = r[order], intensity[order], errors[order]
    dr = float(np.nanmedian(np.diff(r)))
    if not np.isfinite(dr) or dr <= 0:
        return None

    tail = intensity[-max(5, len(intensity) // 5):]
    tail_median = np.nanmedian(tail)
    tail_noise = 1.4826 * np.nanmedian(np.abs(tail - tail_median))
    finite_errors = errors[np.isfinite(errors) & (errors > 0)]
    profile_error = float(np.nanmedian(finite_errors)) if finite_errors.size else 0.0
    noise = max(float(tail_noise), profile_error, 1e-30)
    smooth = gaussian_filter1d(intensity, max(1.0, 0.5 * beam_fwhm_arcsec / dr))
    min_distance = max(1, int(np.ceil(beam_fwhm_arcsec / dr)))
    peaks, _ = find_peaks(smooth, distance=min_distance,
                          prominence=threshold_sigma * noise)
    ring_peaks = [int(i) for i in peaks if r[i] >= 1.5 * beam_fwhm_arcsec
                  and smooth[i] > threshold_sigma * noise
                  and smooth[i] >= min_peak_fraction * np.nanmax(smooth)]
    if not ring_peaks:
        return None

    accepted = [ring_peaks[0]]
    for i in ring_peaks[1:]:
        if smooth[i] >= min_ring_ratio * smooth[accepted[-1]]:
            accepted.append(i)
    last_peak = accepted[-1]
    edge_level = max(threshold_sigma * noise, edge_fraction * smooth[last_peak])
    min_run = max(3, int(np.ceil(beam_fwhm_arcsec / dr)))
    for i in range(last_peak + 1, len(smooth) - min_run + 1):
        if np.all(smooth[i:i + min_run] < edge_level):
            if smooth[i] == smooth[i - 1]:
                edge = r[i]
            else:
                fraction = (edge_level - smooth[i - 1]) / (smooth[i] - smooth[i - 1])
                edge = r[i - 1] + fraction * (r[i] - r[i - 1])
            return float(edge), float(max(dr / 2.0, 0.5 * beam_fwhm_arcsec))
    return None



def save_results_csv(result: dict, output_path: str):
    import csv

    geom, gaps = result["geometry"], result["gaps"]
    distance_pc = result.get("distance_pc")

    def finite_or_empty(value):
        return value if np.isfinite(value) else ""

    def au(val_arcsec):
        return val_arcsec * distance_pc if distance_pc else ""

    common = {
        "inclination_deg": geom.inclination_deg,
        "position_angle_deg": geom.pa_deg,
        "center_x_pix": geom.x0_pix,
        "center_y_pix": geom.y0_pix,
        "disk_radius_arcsec": result.get("disk_radius_arcsec", ""),
        "disk_radius_definition": result.get("disk_radius_definition", ""),
        "disk_radius_err_arcsec": result.get("disk_radius_err_arcsec", ""),
        "disk_radius_AU": au(result.get("disk_radius_arcsec")) if result.get("disk_radius_arcsec") else "",
        "pixel_scale_arcsec_pix": result.get("pixel_scale_arcsec", ""),
        "beam_fwhm_arcsec": result.get("beam_fwhm_arcsec", ""),
        "distance_pc": distance_pc if distance_pc else "",
        "profile_source": result.get("measure_profile").source if result.get("measure_profile") else "",
    }
    field_order = list(common.keys()) + [
        "feature_number", "kind", "reliable", "tier", "significance",
        "azimuth_fraction", "scales_detected",
        "gap_radius_arcsec", "gap_radius_err_arcsec", "gap_radius_AU",
        "gap_width_arcsec", "gap_width_AU", "gap_depth_image_units",
        "gap_reference_ring_intensity", "gap_contrast_Igap_over_ring",
        "gap_bracket_inner_arcsec", "gap_bracket_outer_arcsec",
        "reference_ring_radius_arcsec", "reference_ring_radius_AU",
        "cavity_depth_image_units", "cavity_contrast_Icavity_over_ring",
        "cavity_radius_arcsec", "cavity_radius_err_arcsec", "cavity_radius_AU",
        "cavity_resolution_limited",
        "cavity_ring_peak_radius_arcsec", "cavity_ring_peak_radius_AU",
        "cavity_ring_fwhm_arcsec", "cavity_ring_fwhm_AU",
    ]
    rows = []
    if gaps:
        for i, g in enumerate(gaps, 1):
            row = dict(common)
            row.update({
                "feature_number": i, "kind": g.kind, "reliable": g.reliable,
                "tier": g.tier, "significance": g.significance,
                "azimuth_fraction": g.azimuth_fraction, "scales_detected": g.scales_detected,
                "gap_radius_arcsec": "" if g.kind == "cavity" else g.r_gap_arcsec,
                "gap_radius_err_arcsec": "" if g.kind == "cavity" else g.r_gap_err_arcsec,
                "gap_radius_AU": "" if g.kind == "cavity" else au(g.r_gap_arcsec),
                "gap_width_arcsec": "" if g.kind == "cavity" else g.width_arcsec,
                "gap_width_AU": "" if g.kind == "cavity" else au(g.width_arcsec),
                "gap_depth_image_units": "" if g.kind == "cavity" else g.depth,
                "gap_reference_ring_intensity":
                    "" if g.kind == "cavity" else g.reference_intensity,
                "gap_contrast_Igap_over_ring": "" if g.kind == "cavity" else g.contrast,
                "gap_bracket_inner_arcsec": "" if g.kind == "cavity" else g.bracket_inner_arcsec,
                "gap_bracket_outer_arcsec": "" if g.kind == "cavity" else g.bracket_outer_arcsec,
                "reference_ring_radius_arcsec": "" if g.kind == "cavity" else g.ring_ref_arcsec,
                "reference_ring_radius_AU": "" if g.kind == "cavity" else au(g.ring_ref_arcsec),
                "cavity_depth_image_units": g.depth if g.kind == "cavity" else "",
                "cavity_contrast_Icavity_over_ring": g.contrast if g.kind == "cavity" else "",
                "cavity_radius_arcsec":
                    g.cavity_radius_arcsec if g.kind == "cavity" else "",
                "cavity_radius_err_arcsec":
                    g.cavity_radius_err_arcsec if g.kind == "cavity" else "",
                "cavity_radius_AU":
                    au(g.cavity_radius_arcsec) if g.kind == "cavity" else "",
                "cavity_resolution_limited":
                    g.cavity_resolution_limited if g.kind == "cavity" else "",
                "cavity_ring_peak_radius_arcsec":
                    g.cavity_ring_peak_arcsec if g.kind == "cavity" else "",
                "cavity_ring_peak_radius_AU":
                    au(g.cavity_ring_peak_arcsec) if g.kind == "cavity" else "",
                "cavity_ring_fwhm_arcsec":
                    finite_or_empty(g.cavity_ring_fwhm_arcsec)
                    if g.kind == "cavity" else "",
                "cavity_ring_fwhm_AU":
                    au(g.cavity_ring_fwhm_arcsec)
                    if g.kind == "cavity" and np.isfinite(g.cavity_ring_fwhm_arcsec) else "",
            })
            rows.append(row)
    else:
        row = dict(common)
        for key in field_order[len(common):]:
            row[key] = ""
        row["feature_number"] = 0
        rows.append(row)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[output] wrote {output_path}")


def _overlay_ellipse(ax, geom: Geometry, r_pix, theta, **kwargs):
    inc_rad = np.radians(geom.inclination_deg)
    rot = geom.theta_major_rad
    x_p = r_pix * np.cos(theta)
    y_p = r_pix * np.sin(theta) * np.cos(inc_rad)
    dx = x_p * np.cos(rot) - y_p * np.sin(rot)
    dy = x_p * np.sin(rot) + y_p * np.cos(rot)
    ax.plot(geom.x0_pix + dx, geom.y0_pix + dy, **kwargs)


def plot_results(data: np.ndarray, result: dict, save_path: str = None):
    """Plot fitted disk geometry, measured gap dips, and the outer disk edge."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.ticker import FuncFormatter

    geom = result["geometry"]
    profile = result["measure_profile"]
    gaps = result["gaps"]
    pixel_scale_arcsec = result["pixel_scale_arcsec"]
    disk_radius_arcsec = result.get("disk_radius_arcsec")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    ax = axes[0]

    r_zoom_pix = ((disk_radius_arcsec or (profile.r_arcsec[np.isfinite(profile.intensity)][-1]))
                  / pixel_scale_arcsec)
    margin = 1.5
    xlim = (geom.x0_pix - margin * r_zoom_pix, geom.x0_pix + margin * r_zoom_pix)
    ylim = (geom.y0_pix - margin * r_zoom_pix, geom.y0_pix + margin * r_zoom_pix)

    ny, nx = data.shape
    x_lo, x_hi = int(max(0, xlim[0])), int(min(nx, xlim[1]))
    y_lo, y_hi = int(max(0, ylim[0])), int(min(ny, ylim[1]))
    crop = data[y_lo:y_hi, x_lo:x_hi]
    vmax = np.nanpercentile(crop, 99.5)
    vmin = max(0.0, np.nanpercentile(crop, 1.0))
    norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    im = ax.imshow(data, origin="lower", cmap="inferno", norm=norm)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{(value - geom.x0_pix) * pixel_scale_arcsec:.2f}"))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{(value - geom.y0_pix) * pixel_scale_arcsec:.2f}"))
    ax.set_xlabel("Offset from disk centre (arcsec)")
    ax.set_ylabel("Offset from disk centre (arcsec)")

    theta = np.linspace(0, 2 * np.pi, 200)
    for g in gaps:
        if g.kind == "cavity":
            _overlay_ellipse(ax, geom, g.cavity_radius_arcsec / pixel_scale_arcsec,
                             theta, color="cyan", lw=1, ls=":")
            marker_radius = g.cavity_ring_peak_arcsec
        else:
            marker_radius = g.r_gap_arcsec
        r_pix = marker_radius / pixel_scale_arcsec
        color = "lime"
        ls = "-"
        alpha_v = 1.0 if g.reliable else 0.45
        _overlay_ellipse(ax, geom, r_pix, theta, color=color, lw=1, ls=ls, alpha=alpha_v)
    if disk_radius_arcsec:
        _overlay_ellipse(ax, geom, disk_radius_arcsec / pixel_scale_arcsec, theta,
                          color="white", lw=1.5, ls="--")
    center_label = f"Disk centre used ({result.get('center_mode', 'ring')} mode)"
    gc = result.get("gofish_center_arcsec")
    if gc is not None:
        center_label += f" (GoFish x0={gc[0]:.3f}\", y0={gc[1]:.3f}\")"
    ax.plot(geom.x0_pix, geom.y0_pix, marker="+", color="cyan", ms=12,
            mew=1.5, ls="none", label=center_label)
    rc, cc = result.get("ring_center_pix"), result.get("core_center_pix")
    if rc is not None:
        ax.plot(*rc, marker="D", mfc="none", mec="lime", ms=8, mew=1.3, ls="none",
                label="Fitted geometry centre")
    if cc is not None:
        ax.plot(*cc, marker="o", mfc="none", mec="magenta", ms=8, mew=1.3, ls="none",
                label="Core centroid")
    wcs_obj = result.get("wcs")
    if wcs_obj is not None:
        # FITS phase centre = the (0,0) of GoFish's offset axes. If the disk
        # centre in the plot is not where (x0,y0) says it should be relative
        # to this marker, the offset conversion is wrong.
        ax.plot(wcs_obj.wcs.crpix[0] - 1.0, wcs_obj.wcs.crpix[1] - 1.0, marker="x",
                color="white", ms=8, mew=1.2, ls="none", label="FITS phase centre (GoFish 0,0)")
    ax.legend(loc="lower right", fontsize=7)
    comps = result.get("companions") or []
    if comps:
        ax.contour(np.isfinite(data) & ~np.any([c["mask"] for c in comps], axis=0),
                   levels=[0.5], colors="cyan", linewidths=0.8, linestyles=":")
        ax.text(0.02, 0.97,
                "Masked companion(s): " + ", ".join(
                    f"{c['sep_pix'] * pixel_scale_arcsec:.2f}\" away" for c in comps),
                transform=ax.transAxes, va="top", ha="left", fontsize=7, color="cyan",
                bbox=dict(fc="black", alpha=0.5, ec="none", pad=2))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_title("Continuum image + fitted geometry (substructures / disk edge)")
    plt.colorbar(im, ax=ax, label="Jy/beam", fraction=0.046)

    ax = axes[1]
    inside = np.isfinite(profile.intensity)
    r_plot = profile.r_arcsec[inside]
    I_plot = profile.intensity[inside]
    Ierr_plot = profile.intensity_err[inside]

    ax.errorbar(r_plot, I_plot, yerr=Ierr_plot, fmt="-", color="k", lw=1, ms=0,
                ecolor="gray", elinewidth=0.7, label="Azimuthal average (GoFish)"
                if profile.source == "gofish" else "Azimuthal average (manual fallback)")

    legend_done = {"gap": False, "cavity": False, "cavity_edge": False}
    for g in gaps:
        color = "red"
        is_cavity = g.kind == "cavity"
        feature_type = "cavity" if is_cavity else "gap"
        r_c = g.cavity_ring_peak_arcsec if is_cavity else g.r_gap_arcsec
        feature_width = g.cavity_ring_fwhm_arcsec if is_cavity else g.width_arcsec
        if not np.isfinite(feature_width):
            feature_width = 0.0
        half_w = 0.5 * feature_width
        alpha_v = 1.0 if g.reliable else 0.4
        label = None
        if not legend_done[feature_type]:
            label = "Cavity ring (peak/FWHM)" if is_cavity else "Detected gap"
            legend_done[feature_type] = True
        if is_cavity:
            cavity_label = None
            if not legend_done["cavity_edge"]:
                cavity_label = "Cavity edge (half contrast)"
                legend_done["cavity_edge"] = True
            ax.axvspan(0.0, g.cavity_radius_arcsec, color="cyan", alpha=0.06,
                       label=cavity_label)
            ax.axvline(g.cavity_radius_arcsec, color="cyan", ls=":", alpha=0.9)
        ax.axvline(r_c, color=color, ls="--", alpha=0.8 * alpha_v)
        ax.axvspan(r_c - half_w, r_c + half_w, color=color, alpha=0.15 * alpha_v, label=label)

    radius_label = result.get("disk_radius_definition", "outer radius")
    if disk_radius_arcsec:
        ax.axvline(disk_radius_arcsec, color="blue", ls=":",
                   label=f"Disk outer radius ({radius_label})")
        xmax = 1.2 * disk_radius_arcsec
    else:
        xmax = r_plot[-1] if len(r_plot) else 1.0

    ax.set_xlim(0.0, xmax)
    ax.set_xlabel("Deprojected radius (arcsec)")
    ax.set_ylabel("Intensity (Jy/beam)")
    ax.legend(fontsize=8)

    # Companion logarithmic profile: it is diagnostic only; detection and all
    # reported depths/widths remain measured in linear intensity units.
    ax_log = axes[2]
    positive = inside & np.isfinite(profile.intensity) & (profile.intensity > 0)
    r_log = profile.r_arcsec[positive]
    I_log = profile.intensity[positive]
    err_log = profile.intensity_err[positive]
    ax_log.plot(r_log, I_log, color="black", lw=1.0,
                label="Positive radial intensity")
    good_err = np.isfinite(err_log) & (err_log >= 0) & (I_log - err_log > 0)
    if np.any(good_err):
        ax_log.errorbar(r_log[good_err], I_log[good_err], yerr=err_log[good_err],
                        fmt="none", ecolor="gray", elinewidth=0.6, alpha=0.7)
    for g in gaps:
        is_cavity = g.kind == "cavity"
        r_c = g.cavity_ring_peak_arcsec if is_cavity else g.r_gap_arcsec
        feature_width = g.cavity_ring_fwhm_arcsec if is_cavity else g.width_arcsec
        if not np.isfinite(feature_width):
            feature_width = 0.0
        half_w = 0.5 * feature_width
        alpha_v = 1.0 if g.reliable else 0.4
        if is_cavity:
            ax_log.axvspan(0.0, g.cavity_radius_arcsec, color="cyan", alpha=0.06)
            ax_log.axvline(g.cavity_radius_arcsec, color="cyan", ls=":", alpha=0.9)
        ax_log.axvline(r_c, color="red", ls="--", alpha=0.8 * alpha_v)
        ax_log.axvspan(r_c - half_w, r_c + half_w, color="red",
                       alpha=0.12 * alpha_v)
    if disk_radius_arcsec:
        ax_log.axvline(disk_radius_arcsec, color="blue", ls=":",
                       label=f"Disk outer radius ({radius_label})")
    if len(I_log):
        ymin = np.nanpercentile(I_log, 5)
        ymax = np.nanpercentile(I_log, 99)
        if not np.isfinite(ymin) or ymin <= 0:
            ymin = np.nanmin(I_log[I_log > 0])
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin > 0:
            ax_log.set_ylim(ymin * 0.8, ymax * 1.2)
    ax_log.set_yscale("log")
    ax_log.set_xlim(0.0, xmax)
    ax_log.set_xlabel("Deprojected radius (arcsec)")
    ax_log.set_ylabel("Intensity (Jy/beam; log scale)")
    ax_log.set_title("Log intensity (visual diagnostic only)")
    ax_log.legend(fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[output] wrote {save_path}")
        plt.close(fig)
    return fig


def run_pipeline(fits_path: str, distance_pc: float = None,
                  inclination_deg: float = None, pa_deg: float = None,
                  max_disk_radius_arcsec: float = None,
                  threshold_sigma: float = 5.0,
                  refine_ring_center: bool = None,
                  center_pix: tuple = None, center_radec: str = None,
                  noise_region: tuple = None, center_mode: str = "ring",
                  mask_companions: bool = True,
                  gap_smoothing_beams: float = 0.1):

    if not np.isfinite(gap_smoothing_beams) or gap_smoothing_beams <= 0:
        raise ValueError("Gap smoothing scale must be a positive finite number of beams.")
    data, header, wcs, pixel_scale_arcsec, beam_fwhm_arcsec = load_fits_image(fits_path)
    beam_fwhm_pix = beam_fwhm_arcsec / pixel_scale_arcsec

    # Literature centre (preferred when known): either pixel coords or a
    # sky position such as "16h32m22.9s -24d28m32s" / "248.095 -24.476" (deg).
    if center_radec is not None:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        try:
            sc = SkyCoord(center_radec, unit=(u.hourangle, u.deg)) if ":" in center_radec \
                or "h" in center_radec else SkyCoord(*[float(v) for v in center_radec.split()],
                                                     unit="deg")
        except Exception as exc:
            raise ValueError(f"Could not parse --center-radec {center_radec!r}: {exc}")
        cx, cy = wcs.world_to_pixel(sc)
        center_pix = (float(cx), float(cy))
        print(f"[geometry] using supplied sky centre -> pixel ({cx:.1f}, {cy:.1f})")
    elif center_pix is not None:
        print(f"[geometry] using supplied pixel centre {center_pix}")

    geom = fit_geometry(data, beam_fwhm_pix=beam_fwhm_pix, center_pix=center_pix)
    if inclination_deg is not None:
        geom.inclination_deg = inclination_deg
    if pa_deg is not None:
        geom.pa_deg = pa_deg
        geom.theta_major_rad = np.radians((pa_deg + 90.0) % 180.0)

    # ------------------------------------------------------------------
    # Ring-centre refinement is now data-driven, not source-name-driven.
    # `_looks_like_ring_source` inspects the deprojected radial profile and
    # decides whether the object is a ring/cavity source (peak at r > 1.5
    # beams, core suppressed relative to that peak). If so -> refine; if
    # not -> keep the Gaussian-fit geometry. Callers can still force the
    # decision on or off via the `refine_ring_center` argument.
    # ------------------------------------------------------------------
    if refine_ring_center is None:
        # Verified multi-seed decision: the ring test is applied at the
        # REFINED centre of several independent seeds, never at an
        # unverified starting centre.
        geom, is_ring = choose_geometry(data, geom, pixel_scale_arcsec, beam_fwhm_arcsec,
                                        fit_inc=inclination_deg is None,
                                        fit_pa=pa_deg is None,
                                        center_fixed=center_pix is not None,
                                        phase_center_pix=(float(wcs.wcs.crpix[0] - 1.0),
                                                          float(wcs.wcs.crpix[1] - 1.0)))
    elif refine_ring_center:
        geom = refine_ring_geometry(data, geom, pixel_scale_arcsec, beam_fwhm_arcsec,
                                    fit_inc=inclination_deg is None,
                                    fit_pa=pa_deg is None)
    # Two candidate centres, both drawn on the plot: the ring's geometric
    # centre (what the geometry fit gave) and the compact-core centroid.
    # `center_mode` picks which one the deprojection uses. A user-supplied
    # centre always wins.
    if center_mode not in ("ring", "core"):
        raise ValueError("center_mode must be 'ring' or 'core'")
    ring_center_pix = (geom.x0_pix, geom.y0_pix)
    _, r_pk_pix = _ring_cost_for_geom(data, geom, pixel_scale_arcsec, beam_fwhm_arcsec)
    core_radius = max(2.0 * beam_fwhm_pix,
                      0.45 * r_pk_pix if np.isfinite(r_pk_pix) else 0.0)
    core_center_pix = core_centroid(data, beam_fwhm_pix, near=ring_center_pix,
                                    radius_pix=core_radius)
    if center_mode == "core" and center_pix is None and center_radec is None:
        geom.x0_pix, geom.y0_pix = core_center_pix
    print(f"[geometry] inc={geom.inclination_deg:.1f} deg, PA={geom.pa_deg:.1f} deg, "
          f"center=({geom.x0_pix:.1f},{geom.y0_pix:.1f}) pix [mode: {center_mode}]")

    gofish_center_arcsec = _pixel_offset_to_arcsec_offset(wcs, geom.x0_pix, geom.y0_pix)
    print(f"[geometry] centre passed to GoFish: x0={gofish_center_arcsec[0]:.3f}\", "
          f"y0={gofish_center_arcsec[1]:.3f}\" from the phase centre")

    # Spatially separate compact sources (binary companions such as AS 205 S)
    # are not part of the disk: left in, they add a spurious bump to the
    # azimuthal average and drag the R95 radius out to the companion
    # separation. Mask them (NaN) before the profile is built.
    companions = []
    profile_data, profile_path, _tmp_fits = data, fits_path, None
    if mask_companions:
        companions = find_companions(data, geom, beam_fwhm_pix,
                                     noise=estimate_image_noise(data, header,
                                                                region=noise_region))
        if companions:
            profile_data = data.copy()
            for c in companions:
                profile_data[c["mask"]] = np.nan
                print(f"[companion] masked off-disk source at pixel "
                      f"({c['x']:.0f}, {c['y']:.0f}), {c['sep_pix'] * pixel_scale_arcsec:.2f}\" "
                      f"from the disk centre, peak {c['peak_sigma']:.0f} sigma, "
                      f"azimuth span {c['azimuth_span_deg']:.0f} deg")
            # GoFish reads the file itself, so give it a masked copy
            # (only needed when GoFish is actually installed).
            try:
                import gofish  # noqa: F401
                _have_gofish = True
            except ImportError:
                _have_gofish = False
        if companions and _have_gofish:
            import tempfile, os
            fd, _tmp_fits = tempfile.mkstemp(suffix=".fits")
            os.close(fd)
            hdr_out = header.copy()
            fits.PrimaryHDU(data=profile_data.astype(np.float32).reshape(
                [1] * (hdr_out.get("NAXIS", 2) - 2) + list(profile_data.shape)),
                header=hdr_out).writeto(_tmp_fits, overwrite=True)
            profile_path = _tmp_fits

    try:
        measure_profile = build_radial_profile(profile_path, profile_data, geom, wcs,
                                                pixel_scale_arcsec, beam_fwhm_arcsec)
    finally:
        if _tmp_fits is not None:
            try:
                os.remove(_tmp_fits)
            except OSError:
                pass

    # Keep the outer radius on the R95-flux definition used by the literature.
    # Ring-edge estimates are a different radius definition and can truncate
    # both the plotted disk and outer gaps in multi-ring systems.
    noise = estimate_image_noise(data, header, region=noise_region)
    max_profile_radius_arcsec = _max_fully_sampled_radius_arcsec(
        geom, data.shape, pixel_scale_arcsec)
    disk_radius_arcsec, disk_radius_err_arcsec = estimate_disk_radius(
        measure_profile, noise, beam_fwhm_arcsec, threshold_sigma=threshold_sigma,
        max_radius_arcsec=max_profile_radius_arcsec)
    disk_radius_definition = "R95 flux"
    coherent_edge = estimate_outer_ring_edge(measure_profile, beam_fwhm_arcsec)
    if coherent_edge is not None and disk_radius_arcsec > (
            coherent_edge[0] + 1.5 * beam_fwhm_arcsec):
        disk_radius_arcsec, disk_radius_err_arcsec = estimate_disk_radius(
            measure_profile, noise, beam_fwhm_arcsec, threshold_sigma=threshold_sigma,
            max_radius_arcsec=coherent_edge[0])
        disk_radius_definition = "R95 flux (within coherent edge)"
    if max_disk_radius_arcsec is None:
        max_disk_radius_arcsec = disk_radius_arcsec
    gap_radius_limit = max_disk_radius_arcsec
    if coherent_edge is not None:
        gap_radius_limit = min(gap_radius_limit, coherent_edge[0])
    gaps = find_gaps_robust(
        measure_profile, beam_fwhm_arcsec,
        smooth_beams=gap_smoothing_beams,
        max_gap_radius_arcsec=gap_radius_limit,
        cavity_hint=_known_cavity_source(fits_path))

    print(f"[disk radius] {disk_radius_arcsec:.3f} +/- "
          f"{disk_radius_err_arcsec:.3f} arcsec ({disk_radius_definition})")
    print(f"\n{len(gaps)} feature(s) accepted:")
    for i, g in enumerate(gaps, 1):
        if g.kind == "cavity":
            r_report = g.cavity_radius_arcsec
            width_report = g.cavity_ring_fwhm_arcsec
            description = (f"cavity radius={r_report:.3f}\"; "
                           f"ring peak={g.cavity_ring_peak_arcsec:.3f}\"")
            width_description = (
                f"ring FWHM={width_report:.3f}\""
                if np.isfinite(width_report) else "ring FWHM=unresolved")
        else:
            r_report = g.r_gap_arcsec
            width_report = g.width_arcsec
            description = f"gap radius/width={r_report:.3f}\""
            width_description = f"width={width_report:.3f}\""
        au_str = (f" ({r_report * distance_pc:.1f} au, "
                  f"width {width_report * distance_pc:.1f} au)"
                  if distance_pc and np.isfinite(width_report) else "")
        print(f"  {i}. [{g.kind}] {description}{au_str} | "
              f"{width_description} | depth={g.depth:.4g} | "
              f"contrast={g.contrast:.3f} | significance={g.significance:.1f} sigma | {g.tier}")

    return dict(geometry=geom, measure_profile=measure_profile,
                gaps=gaps,
                disk_radius_arcsec=disk_radius_arcsec,
                disk_radius_definition=disk_radius_definition,
                disk_radius_err_arcsec=disk_radius_err_arcsec,
                pixel_scale_arcsec=pixel_scale_arcsec,
                beam_fwhm_arcsec=beam_fwhm_arcsec,
                distance_pc=distance_pc, data=data, wcs=wcs,
                gofish_center_arcsec=gofish_center_arcsec,
                ring_center_pix=ring_center_pix, core_center_pix=core_center_pix,
                center_mode=center_mode, companions=companions)


if __name__ == "__main__":
    import argparse
    import csv
    import time
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fits_path", nargs="?", default=None,
                   help="one FITS image, or a folder containing FITS images")
    p.add_argument("--batch-dir", default=None,
                   help="process every .fits/.fit/.fts file in this folder")
    p.add_argument("--output-dir", default="disk2_results",
                   help="output folder used in batch mode (default: disk2_results)")
    p.add_argument("distance_pc", nargs="?", type=float, default=None)
    p.add_argument("--inclination-deg", type=float, default=None)
    p.add_argument("--pa-deg", type=float, default=None)
    p.add_argument("--center-pix", nargs=2, type=float, default=None, metavar=("X0", "Y0"),
                   help="disk centre in pixel coordinates (literature / manual)")
    p.add_argument("--center-radec", default=None,
                   help='disk centre on the sky, e.g. "16h32m22.9s -24d28m32s" or "248.095 -24.476" (deg)')
    p.add_argument("--center-mode", choices=["ring", "core"], default="ring",
                   help="centre used for deprojection: the ring's geometric centre "
                        "(default) or the compact-core centroid. Both are drawn on the plot.")
    p.add_argument("--noise-region", nargs=4, type=int, default=None,
                   metavar=("Y0", "Y1", "X0", "X1"),
                   help="emission-free pixel box used for the image rms")
    p.add_argument("--gap-smoothing-beams", type=float, default=0.1,
                   help="starting radial-profile Gaussian smoothing in beam FWHM; "
                        "detection also checks 2x and 4x this scale (default: 0.1)")
    p.add_argument("--no-mask-companions", dest="mask_companions", action="store_false",
                   help="do not mask separate compact companion sources before profiling")
    center_group = p.add_mutually_exclusive_group()
    center_group.add_argument("--refine-ring-center", dest="refine_ring_center",
                              action="store_true",
                              help="force ring-based centre/inc/PA refinement")
    center_group.add_argument("--no-refine-ring-center", dest="refine_ring_center",
                              action="store_false",
                              help="disable ring refinement even if the data looks ring-like")
    p.set_defaults(refine_ring_center=None)   # None -> auto-classify from data
    p.add_argument("--csv-out", default="disk_gap_results.csv")
    p.add_argument("--png-out", default="disk_gap_results.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    if args.batch_dir is not None and args.fits_path is not None:
        p.error("provide either a FITS path/folder or --batch-dir, not both")
    input_path = args.batch_dir or args.fits_path
    if input_path is None:
        p.error("provide a FITS path/folder or --batch-dir")

    input_path = Path(input_path).expanduser()
    if input_path.is_dir():
        batch_dir = input_path
    else:
        batch_dir = None

    pipeline_options = dict(
        distance_pc=args.distance_pc,
        inclination_deg=args.inclination_deg,
        pa_deg=args.pa_deg,
        refine_ring_center=args.refine_ring_center,
        center_pix=tuple(args.center_pix) if args.center_pix else None,
        center_radec=args.center_radec,
        center_mode=args.center_mode,
        noise_region=tuple(args.noise_region) if args.noise_region else None,
        mask_companions=args.mask_companions,
        gap_smoothing_beams=args.gap_smoothing_beams,
    )

    if batch_dir is None:
        result = run_pipeline(str(input_path), **pipeline_options)
        save_results_csv(result, args.csv_out)
        plot_results(result["data"], result, save_path=args.png_out)
    else:
        files = sorted({f for pattern in ("*.fits", "*.fit", "*.fts")
                        for f in batch_dir.rglob(pattern) if f.is_file()})
        if not files:
            p.error(f"no FITS images found in {batch_dir}")
        output_dir = Path(args.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        failures = []
        for index, source_path in enumerate(files, 1):
            stem = source_path.name
            for suffix in (".fits", ".fit", ".fts"):
                if stem.lower().endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            csv_path = output_dir / f"{stem}_gaps.csv"
            png_path = output_dir / f"{stem}_disk_gaps.png"
            print(f"\n[batch {index}/{len(files)}] {source_path}")
            started = time.perf_counter()
            try:
                result = run_pipeline(str(source_path), **pipeline_options)
                save_results_csv(result, str(csv_path))
                plot_results(result["data"], result, save_path=str(png_path))
                summary_rows.append({
                    "source": str(source_path),
                    "center_x_pix": result["geometry"].x0_pix,
                    "center_y_pix": result["geometry"].y0_pix,
                    "inclination_deg": result["geometry"].inclination_deg,
                    "position_angle_deg": result["geometry"].pa_deg,
                    "disk_radius_arcsec": result["disk_radius_arcsec"],
                    "disk_radius_definition": result["disk_radius_definition"],
                    "gap_count": sum(g.kind != "cavity" for g in result["gaps"]),
                    "cavity_count": sum(g.kind == "cavity" for g in result["gaps"]),
                    "csv": str(csv_path),
                    "plot": str(png_path),
                    "status": "ok",
                    "error": "",
                })
                print(f"[batch] finished in {time.perf_counter() - started:.1f}s")
            except Exception as exc:
                import traceback
                traceback.print_exc()
                failures.append((source_path, exc))
                summary_rows.append({
                    "source": str(source_path),
                    "center_x_pix": "",
                    "center_y_pix": "",
                    "inclination_deg": "",
                    "position_angle_deg": "",
                    "disk_radius_arcsec": "",
                    "disk_radius_definition": "",
                    "gap_count": "",
                    "cavity_count": "",
                    "csv": "",
                    "plot": "",
                    "status": "failed",
                    "error": str(exc),
                })
        summary_path = output_dir / "batch_summary.csv"
        with summary_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[batch] summary written to {summary_path}")
        print(f"[batch] completed {len(files) - len(failures)}/{len(files)} source(s)")
        if failures:
            raise SystemExit(1)