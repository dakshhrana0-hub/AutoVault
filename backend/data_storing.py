"""
AutoVault — Upload Cars to Supabase
====================================
Reads merged_luxe_dataset.csv and merged_ordinary_dataset.csv
from  data/merged_datasets/  and uploads them to the `cars` table.

SETUP:
  pip install supabase pandas

USAGE:
  1. Fill in SUPABASE_URL and SUPABASE_KEY below (use your service_role key,
     NOT the anon key — service_role bypasses RLS for bulk inserts).
  2. Run:  python upload_cars.py
  3. After uploading, switch back to your anon key in config.js.

SQL to create the table (run in Supabase → SQL Editor first):
------------------------------------------------------------
create table if not exists public.cars (
  id             uuid primary key default gen_random_uuid(),
  title          text,
  link           text,
  location       text,
  price          numeric,
  price_display  text,
  image          text,
  fuel_type      text,
  transmission   text,
  year           integer,
  kms_covered    numeric,
  brand          text,
  tier           text not null default 'standard',  -- 'luxury' or 'standard'
  exterior_color text,
  interior_color text,
  registration   text,
  ownership      integer,
  region_info    text,
  created_at     timestamptz default now()
);

-- Public read policy
alter table public.cars enable row level security;
create policy "Public can read cars"
  on public.cars for select to public using (true);

-- Optional: index for fast filtering
create index if not exists cars_tier_idx  on public.cars(tier);
create index if not exists cars_brand_idx on public.cars(brand);
create index if not exists cars_year_idx  on public.cars(year);
create index if not exists cars_price_idx on public.cars(price);
------------------------------------------------------------
"""

import os
import math
import pandas as pd
from supabase import create_client, Client

# ── CONFIG ──────────────────────────────────────────────────────────────────
SUPABASE_URL = 'https://xxx.supabase.co'        # e.g. https://xxxx.supabase.co
SUPABASE_KEY = 'api_key'   # Settings → API → service_role key

# Path to your CSV files (relative to this script)
LUXE_CSV       = os.path.join( "data", "merged_datasets", "merged_luxe_dataset.csv")
ORDINARY_CSV   = os.path.join( "data", "merged_datasets", "merged_ordinary_dataset.csv")

BATCH_SIZE = 500   # rows per insert call — stay under Supabase's 1 MB payload limit
# ────────────────────────────────────────────────────────────────────────────


def clean_value(v):
    """Convert NaN / NaT to None so JSON serialisation works."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def load_luxury(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop the unnamed index column
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    # Rename to match DB schema
    df = df.rename(columns={
        "Title":          "title",
        "Link":           "link",
        "Location":       "location",
        "Price":          "price",
        "Image":          "image",
        "Fuel Type":      "fuel_type",
        "Type":           "transmission",
        "Year":           "year",
        "Kms Covered":    "kms_covered",
        "Brand":          "brand",
        "Exterior Color": "exterior_color",
        "Interior Color": "interior_color",
        "Registration":   "registration",
        "Ownership":      "ownership",
        "price_display":  "price_display",
    })

    df["tier"] = "luxury"

    # Year is float in this file — cast to nullable int
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Ownership is 1/2/3 float — cast to nullable int
    df["ownership"] = pd.to_numeric(df["ownership"], errors="coerce").astype("Int64")

    # Luxury CSV has no region_info column
    df["region_info"] = None

    return df


def load_ordinary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    df = df.rename(columns={
        "Title":        "title",
        "Link":         "link",
        "Location":     "location",
        "Price":        "price",
        "Image":        "image",
        "Fuel Type":    "fuel_type",
        "Type":         "transmission",
        "Year":         "year",
        "Kms Covered":  "kms_covered",
        "Brand":        "brand",
        "Region Info":  "region_info",
        "price_display":"price_display",
    })

    df["tier"] = "standard"

    # Ordinary CSV has no luxury-only columns
    for col in ["exterior_color", "interior_color", "registration", "ownership"]:
        df[col] = None

    return df


def build_record(row: pd.Series) -> dict:
    """Convert a DataFrame row to a clean dict for Supabase insert."""
    record = {}
    for col in [
        "title", "link", "location", "price", "price_display",
        "image", "fuel_type", "transmission", "year", "kms_covered",
        "brand", "tier", "exterior_color", "interior_color",
        "registration", "ownership", "region_info",
    ]:
        val = row.get(col, None)
        # pandas Int64 NA → None
        try:
            if pd.isna(val):
                val = None
        except (TypeError, ValueError):
            pass
        record[col] = val
    return record


def upload_batch(sb: Client, records: list, label: str):
    """Insert a batch and print result."""
    try:
        res = sb.table("cars").insert(records).execute()
        print(f"  ✓ Inserted {len(records)} rows ({label})")
    except Exception as e:
        print(f"  ✗ Error inserting batch ({label}): {e}")
        raise


def upload_df(sb: Client, df: pd.DataFrame, tier: str):
    total = len(df)
    print(f"\nUploading {total} {tier} cars in batches of {BATCH_SIZE}…")

    records = [build_record(row) for _, row in df.iterrows()]

    for i in range(0, total, BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        label = f"{tier} rows {i+1}–{min(i+BATCH_SIZE, total)}"
        upload_batch(sb, batch, label)

    print(f"  Done — {total} {tier} rows uploaded.")


def main():
    # ── Validate config ──
    if "YOUR_" in SUPABASE_URL or "YOUR_" in SUPABASE_KEY:
        print("✗ Please fill in SUPABASE_URL and SUPABASE_KEY in the script.")
        return

    # ── Init client ──
    sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Connected to Supabase")

    # ── Load CSVs ──
    print(f"\nLoading luxury CSV: {LUXE_CSV}")
    luxe_df = load_luxury(LUXE_CSV)
    print(f"  → {len(luxe_df)} rows, columns: {luxe_df.columns.tolist()}")

    print(f"\nLoading ordinary CSV: {ORDINARY_CSV}")
    ordinary_df = load_ordinary(ORDINARY_CSV)
    print(f"  → {len(ordinary_df)} rows, columns: {ordinary_df.columns.tolist()}")

    # ── Upload ──
    upload_df(sb, luxe_df, "luxury")
    upload_df(sb, ordinary_df, "standard")

    print(f"\n✓ All done! Total rows uploaded: {len(luxe_df) + len(ordinary_df)}")
    print("  Go to Supabase → Table Editor → cars to verify.")


if __name__ == "__main__":
    main()