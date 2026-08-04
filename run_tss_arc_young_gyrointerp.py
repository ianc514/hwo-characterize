import os

# gyrointerp writes cache files under ~/.gyrointerp_cache. Keep the cache local
# to this project folder.
os.environ["HOME"] = os.getcwd()

import argparse

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.mast import Catalogs
from gyrointerp import gyro_age_posterior_list, get_summary_statistics


def clean_value(value):
    if value is None or np.ma.is_masked(value):
        return ""
    text = str(value)
    if text in ["--", "nan", "None"]:
        return ""
    return text


def float_or_nan(value):
    if value is None or np.ma.is_masked(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def query_tic_for_row(row, radius_arcsec):
    coord = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)
    matches = Catalogs.query_region(
        coord,
        catalog="TIC",
        radius=radius_arcsec * u.arcsec,
    )

    if matches is None or len(matches) == 0:
        return "", np.nan, np.nan

    if "dstArcSec" in matches.colnames:
        matches.sort("dstArcSec")

    best = matches[0]
    tic_id = clean_value(best["ID"]) if "ID" in matches.colnames else ""
    teff = float_or_nan(best["Teff"]) if "Teff" in matches.colnames else np.nan
    distance = (
        float_or_nan(best["dstArcSec"])
        if "dstArcSec" in matches.colnames
        else np.nan
    )

    return tic_id, teff, distance


def prepare_young_tss_arc(input_csv, max_age_gyr, limit):
    arc = Table.read(input_csv, format="csv")
    age = np.array(arc["Age"], dtype=float)

    if hasattr(arc["Age"], "mask"):
        valid_age = ~arc["Age"].mask
    else:
        valid_age = np.isfinite(age)

    young = arc[valid_age & (age < max_age_gyr)]
    return young[:limit]


def add_tic_teff(table, radius_arcsec):
    tic_id_values = []
    tic_teff = []
    tic_match_distance_arcsec = []
    tic_note = []

    for i, row in enumerate(table):
        name = clean_value(row["star_name"])
        try:
            tic_id, teff, distance = query_tic_for_row(row, radius_arcsec)
            if tic_id and np.isfinite(teff):
                note = ""
            elif tic_id:
                note = "TIC match found but no Teff"
            else:
                note = "no TIC match found"
        except Exception as exc:
            tic_id = ""
            teff = np.nan
            distance = np.nan
            note = f"TIC query failed: {exc}"

        tic_id_values.append(tic_id)
        tic_teff.append(teff)
        tic_match_distance_arcsec.append(distance)
        tic_note.append(note)

        print(
            f"{i + 1}/{len(table)} {name}: "
            f"TIC={tic_id or 'none'}, Teff={teff if np.isfinite(teff) else 'nan'}"
        )

    out = table.copy()
    out["tic_id"] = tic_id_values
    out["tic_teff"] = tic_teff
    out["tic_match_distance_arcsec"] = tic_match_distance_arcsec
    out["tic_note"] = tic_note
    return out


def run_gyrointerp(table, output_csv, cache_id, nworkers):
    age_grid = np.linspace(0, 5000, 500)

    gyro_note = []
    valid_indices = []
    valid_prots = []
    valid_teffs = []
    valid_star_ids = []

    for i, row in enumerate(table):
        prot = float_or_nan(row["prot_adopt"])
        teff = float_or_nan(row["tic_teff"])

        if not np.isfinite(prot):
            gyro_note.append("missing prot_adopt")
            continue

        if not np.isfinite(teff):
            gyro_note.append("missing TIC Teff")
            continue

        if teff < 3800 or teff > 6200:
            gyro_note.append("outside gyrointerp Teff range")
            continue

        valid_indices.append(i)
        valid_prots.append(prot)
        valid_teffs.append(teff)
        valid_star_ids.append(f"{i:04d}_{clean_value(row['star_name']).replace(' ', '_')}")
        gyro_note.append("")

    print(f"Computing gyro ages for {len(valid_indices)} valid stars...")
    posterior_paths = gyro_age_posterior_list(
        cache_id,
        np.array(valid_prots),
        np.array(valid_teffs),
        star_ids=np.array(valid_star_ids),
        age_grid=age_grid,
        bounds_error="4gyrextrap",
        nworkers=nworkers,
    )

    posterior_by_star_id = {}
    for path in posterior_paths:
        for star_id in valid_star_ids:
            if f"{star_id}_" in path:
                posterior_by_star_id[star_id] = path
                break

    gyro_age_myr = np.full(len(table), np.nan)
    gyro_age_gyr = np.full(len(table), np.nan)
    gyro_age_peak_myr = np.full(len(table), np.nan)
    gyro_age_mean_myr = np.full(len(table), np.nan)
    gyro_age_plus1sigma_myr = np.full(len(table), np.nan)
    gyro_age_minus1sigma_myr = np.full(len(table), np.nan)

    for i, star_id in zip(valid_indices, valid_star_ids):
        if star_id not in posterior_by_star_id:
            gyro_note[i] = "posterior file not found"
            continue

        posterior = pd.read_csv(posterior_by_star_id[star_id])
        stats = get_summary_statistics(age_grid, posterior["age_post"].to_numpy())

        gyro_age_myr[i] = float(stats["median"])
        gyro_age_gyr[i] = gyro_age_myr[i] / 1000
        gyro_age_peak_myr[i] = float(stats["peak"])
        gyro_age_mean_myr[i] = float(stats["mean"])
        gyro_age_plus1sigma_myr[i] = float(stats["+1sigma"])
        gyro_age_minus1sigma_myr[i] = float(stats["-1sigma"])

        print(
            f"{i + 1}/{len(table)} {table['star_name'][i]}: "
            f"Prot={float(table['prot_adopt'][i]):.4f} d, "
            f"Teff={float(table['tic_teff'][i]):.0f} K, "
            f"gyro age={gyro_age_myr[i]:.1f} Myr"
        )

    out = table.copy()
    out["gyro_age_myr"] = gyro_age_myr
    out["gyro_age_gyr"] = gyro_age_gyr
    out["gyro_age_peak_myr"] = gyro_age_peak_myr
    out["gyro_age_mean_myr"] = gyro_age_mean_myr
    out["gyro_age_plus1sigma_myr"] = gyro_age_plus1sigma_myr
    out["gyro_age_minus1sigma_myr"] = gyro_age_minus1sigma_myr
    out["gyro_age_note"] = gyro_note
    out.write(output_csv, format="csv", overwrite=True)
    print(f"wrote {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="TSS_ARC.csv")
    parser.add_argument("--max-age-gyr", type=float, default=4)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--tic-radius-arcsec", type=float, default=5)
    parser.add_argument(
        "--matched-output",
        default="tss_arc_age_lt4_first1000_tic.csv",
    )
    parser.add_argument(
        "--gyro-output",
        default="tss_arc_age_lt4_first1000_gyro_ages.csv",
    )
    parser.add_argument("--cache-id", default="tss_arc_age_lt4_first1000_gyro_ages")
    parser.add_argument("--nworkers", type=int, default=4)
    args = parser.parse_args()

    young = prepare_young_tss_arc(args.input, args.max_age_gyr, args.limit)
    print(f"Selected {len(young)} rows from {args.input}")
    young_tic = add_tic_teff(young, args.tic_radius_arcsec)
    young_tic.write(args.matched_output, format="csv", overwrite=True)
    print(f"wrote {args.matched_output}")

    run_gyrointerp(
        young_tic,
        output_csv=args.gyro_output,
        cache_id=args.cache_id,
        nworkers=args.nworkers,
    )


if __name__ == "__main__":
    main()
