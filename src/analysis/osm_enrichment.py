import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import Point
from tqdm import tqdm

ACTIVITIES = Path(__file__).parents[2] / "dataset" / "preprocessed" / "activities.parquet"
OUT = Path(__file__).parents[2] / "dataset" / "external" / "osm_poi_features.parquet"
RESULT = Path(__file__).parents[2] / "result" / "analysis"
RESULT.mkdir(parents=True, exist_ok=True)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RADII_KM = [1.0, 5.0]
TOP_N_DEST = 30          # fetch OSM for top N destinations by activity count
RETRY_DELAY = 10         # seconds between Overpass retries
MAX_RETRIES = 3

OSM_TAGS = [
    '"tourism"~"hotel|attraction|museum|viewpoint|artwork|zoo|theme_park"',
    '"amenity"~"restaurant|cafe|bar|theatre|cinema|nightclub"',
    '"leisure"~"park|beach_resort|marina|sports_centre"',
]


def bbox_from_centroid(lat: float, lon: float, pad_km: float = 8.0) -> tuple:
    pad_deg_lat = pad_km / 111.0
    pad_deg_lon = pad_km / (111.0 * np.cos(np.radians(lat)))
    return (lat - pad_deg_lat, lon - pad_deg_lon,
            lat + pad_deg_lat, lon + pad_deg_lon)


def fetch_pois(lat: float, lon: float, dest_name: str) -> list:
    s, w, n, e = bbox_from_centroid(lat, lon, pad_km=8.0)
    tag_filters = "".join(f"node[{t}]({s},{w},{n},{e});" for t in OSM_TAGS)
    query = f"[out:json][timeout:30];({tag_filters});out center;"

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=40)
            if resp.status_code == 200:
                elements = resp.json().get("elements", [])
                return [(el["lat"], el["lon"]) for el in elements
                        if "lat" in el and "lon" in el]
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  Overpass error ({dest_name}): {e}")
            time.sleep(RETRY_DELAY)
    return []


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def poi_density(act_lat, act_lon, poi_coords: np.ndarray, radius_km: float) -> int:
    if len(poi_coords) == 0:
        return 0
    dists = haversine_km(act_lat, act_lon, poi_coords[:, 0], poi_coords[:, 1])
    return int((dists <= radius_km).sum())


def main():
    print("Loading activities...")
    df = pd.read_parquet(ACTIVITIES)
    df = df.dropna(subset=["lat", "lon", "destination_display_name"]).reset_index(drop=True)
    print(f"Activities with coords: {len(df):,}")

    dest_counts = df["destination_display_name"].value_counts()
    top_dests = dest_counts.head(TOP_N_DEST).index.tolist()
    print(f"Processing top {TOP_N_DEST} destinations: {[d.split(',')[0] for d in top_dests]}")

    dest_centroid = (
        df[df["destination_display_name"].isin(top_dests)]
        .groupby("destination_display_name")[["lat", "lon"]]
        .mean()
    )

    dest_poi_cache = {}
    for dest in tqdm(top_dests, desc="Fetching OSM POIs"):
        clat = dest_centroid.loc[dest, "lat"]
        clon = dest_centroid.loc[dest, "lon"]
        pois = fetch_pois(clat, clon, dest)
        dest_poi_cache[dest] = np.array(pois) if pois else np.empty((0, 2))
        tqdm.write(f"  {dest.split(',')[0]}: {len(pois)} POIs fetched")
        time.sleep(1.5)   # polite rate limiting

    col_names = [f"poi_density_{int(r)}km" for r in RADII_KM]
    results = []

    sub = df[df["destination_display_name"].isin(top_dests)].copy()
    print(f"\nComputing POI density for {len(sub):,} activities...")

    for dest, grp in tqdm(sub.groupby("destination_display_name"), desc="Computing density"):
        poi_coords = dest_poi_cache.get(dest, np.empty((0, 2)))
        for idx, row in grp.iterrows():
            record = {"product_id": row["product_id"]}
            for r_km, col in zip(RADII_KM, col_names):
                record[col] = poi_density(row["lat"], row["lon"], poi_coords, r_km)
            results.append(record)

    df_feat = pd.DataFrame(results)

    # Normalise
    for col in col_names:
        max_val = df_feat[col].max()
        df_feat[f"{col}_norm"] = (df_feat[col] / max(max_val, 1)).astype(np.float32)

    df_feat.to_parquet(OUT, index=False)
    print(f"\nSaved {OUT}  shape={df_feat.shape}")
    print(df_feat.describe())

    # Summary for paper
    print("\n=== PAPER NUMBERS ===")
    for col in col_names:
        print(f"  {col}: mean={df_feat[col].mean():.1f}  max={df_feat[col].max()}")


if __name__ == "__main__":
    main()
