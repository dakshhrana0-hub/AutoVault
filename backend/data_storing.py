"""
AutoVault — Upload Cars to Supabase (with deduplication)
==========================================================
Reads merged_luxe_dataset.csv and merged_ordinary_dataset.csv,
merges them, deduplicates, then uploads to the `cars` table.

Deduplication steps (applied after merge):
  1. Exact duplicate links           — same listing scraped twice
  2. Fusioncars spam                 — 1 car posted with hundreds of fake links,
                                       keep 1 per title+price+kms+year
  3. Same car, same platform         — identical title+price+kms+year+fuel+trans
                                       on same domain, only location differs
  4. Slightly different title, same  — same brand+price+kms+year+fuel+trans
     car on same platform              on same domain (e.g. "BMW X1" vs "BMW X1 sDrive")

SETUP:
  pip install supabase pandas

USAGE:
  1. Fill in SUPABASE_URL and SUPABASE_KEY below (use your service_role key,
     NOT the anon key — service_role bypasses RLS for bulk inserts).
  2. Run:  python upload_cars.py
  3. After uploading, switch back to your anon key in config.js.

BEFORE RUNNING — truncate the table first to avoid duplicates:
  Supabase → SQL Editor → run:
    TRUNCATE TABLE public.cars RESTART IDENTITY;

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
  tier           text not null default 'standard',
  exterior_color text,
  interior_color text,
  registration   text,
  ownership      integer,
  region_info    text,
  created_at     timestamptz default now()
);

alter table public.cars enable row level security;
create policy "Public can read cars"
  on public.cars for select to public using (true);

create index if not exists cars_tier_idx  on public.cars(tier);
create index if not exists cars_brand_idx on public.cars(brand);
create index if not exists cars_year_idx  on public.cars(year);
create index if not exists cars_price_idx on public.cars(price);
------------------------------------------------------------
"""

import os
import math
import urllib.parse
import pandas as pd
from supabase import create_client, Client

# ── CONFIG ──────────────────────────────────────────────────────────────────
SUPABASE_URL = 'https://xxx.supabase.co'
SUPABASE_KEY = 'api_key'   # Settings → API → service_role (secret)

LUXE_CSV     = os.path.join("data", "merged_datasets", "merged_luxe_dataset.csv")
ORDINARY_CSV = os.path.join("data", "merged_datasets", "merged_ordinary_dataset.csv")

BATCH_SIZE   = 500
# ────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD CSVs
# ═══════════════════════════════════════════════════════════════════════════

def load_luxury(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
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
    df["tier"]        = "luxury"
    df["region_info"] = None
    df["year"]        = pd.to_numeric(df["year"],      errors="coerce").astype("Int64")
    df["ownership"]   = pd.to_numeric(df["ownership"], errors="coerce").astype("Int64")
    return df


def load_ordinary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.rename(columns={
        "Title":         "title",
        "Link":          "link",
        "Location":      "location",
        "Price":         "price",
        "Image":         "image",
        "Fuel Type":     "fuel_type",
        "Type":          "transmission",
        "Year":          "year",
        "Kms Covered":   "kms_covered",
        "Brand":         "brand",
        "Region Info":   "region_info",
        "price_display": "price_display",
    })
    df["tier"] = "standard"
    for col in ["exterior_color", "interior_color", "registration", "ownership"]:
        if col not in df.columns:
            df[col] = None
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. DEDUPLICATE
# ═══════════════════════════════════════════════════════════════════════════

def get_domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(str(url)).netloc
    except Exception:
        return ""


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate car listings using 4 progressive passes.

    Pass 1 — Exact duplicate links
            Same URL scraped more than once → keep first occurrence.

    Pass 2 — Fusioncars spam
            A single car is posted with hundreds of unrelated fake links
            (e.g. "2024/26 MERCEDES BENZ CLE 300 CABRIOLET" appears 477 times
            each pointing to a completely different car's URL).
            Within fusioncars.in, keep only 1 row per title+price+kms+year.

    Pass 3 — Same car, same platform, different location
            Identical title+price+kms_covered+year+fuel_type+transmission on
            the same domain — the seller re-listed in multiple areas.
            Keep the first occurrence (usually the most complete record).

    Pass 4 — Slightly different title, same car, same platform
            e.g. "BMW X1" vs "BMW X1 sDrive20d" at the exact same
            price+kms+year+fuel+trans on the same domain.
            Keep the first occurrence (longer title usually comes first).
    """
    original = len(df)
    print(f"\n{'='*55}")
    print(f"  DEDUPLICATION  ({original:,} rows in)")
    print(f"{'='*55}")

    # ── Pass 1: exact duplicate links ─────────────────────────────────────
    df = df.drop_duplicates(subset=["link"])
    print(f"  Pass 1 — duplicate links removed        : {original - len(df):>5,}  →  {len(df):,} remain")

    # ── Pass 2: fusioncars spam ────────────────────────────────────────────
    p2_before = len(df)
    fus_mask  = df["link"].str.contains("fusioncars", na=False)
    df = pd.concat([
        df[~fus_mask],
        df[fus_mask].drop_duplicates(subset=["title", "price", "kms_covered", "year"])
    ], ignore_index=True)
    print(f"  Pass 2 — fusioncars spam removed         : {p2_before - len(df):>5,}  →  {len(df):,} remain")

    # ── Pass 3: same car, same platform, different location ───────────────
    p3_before   = len(df)
    df["_domain"] = df["link"].apply(get_domain)
    df = df.drop_duplicates(
        subset=["_domain", "title", "price", "kms_covered", "year", "fuel_type", "transmission"]
    )
    print(f"  Pass 3 — same car / same platform        : {p3_before - len(df):>5,}  →  {len(df):,} remain")

    # ── Pass 4: slightly different title, same car, same platform ─────────
    p4_before = len(df)
    df = df.drop_duplicates(
        subset=["_domain", "brand", "price", "kms_covered", "year", "fuel_type", "transmission"]
    )
    print(f"  Pass 4 — fuzzy title / same platform     : {p4_before - len(df):>5,}  →  {len(df):,} remain")

    # Drop helper column
    df = df.drop(columns=["_domain"])

    total_removed = original - len(df)
    print(f"\n  Total removed : {total_removed:,}  ({total_removed / original * 100:.1f}%)")
    print(f"  Final count   : {len(df):,}")
    print(f"\n  Tier breakdown after dedup:")
    for tier, count in df["tier"].value_counts().items():
        print(f"    {tier:<12} : {count:,}")
    print(f"{'='*55}")

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. UPLOAD HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def build_record(row: pd.Series) -> dict:
    cols = [
        "title", "link", "location", "price", "price_display",
        "image", "fuel_type", "transmission", "year", "kms_covered",
        "brand", "tier", "exterior_color", "interior_color",
        "registration", "ownership", "region_info",
    ]
    record = {}
    for col in cols:
        val = row.get(col, None)
        try:
            if pd.isna(val):
                val = None
        except (TypeError, ValueError):
            pass
        record[col] = val
    return record


def upload_batch(sb: Client, records: list, label: str):
    try:
        sb.table("cars").insert(records).execute()
        print(f"  ✓  {label}")
    except Exception as e:
        print(f"  ✗  {label} — ERROR: {e}")
        raise


def upload_df(sb: Client, df: pd.DataFrame):
    total   = len(df)
    records = [build_record(row) for _, row in df.iterrows()]

    print(f"\n🚀 Uploading {total:,} rows in batches of {BATCH_SIZE}…\n")
    for i in range(0, total, BATCH_SIZE):
        batch  = records[i: i + BATCH_SIZE]
        lo, hi = i + 1, min(i + BATCH_SIZE, total)
        upload_batch(sb, batch, f"rows {lo:,} – {hi:,}  ({len(batch)} records)")

    print(f"\n✓ Upload complete — {total:,} rows inserted.")


# ═══════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if "YOUR_" in SUPABASE_KEY or "xxxx" in SUPABASE_URL:
        print("✗ Please set SUPABASE_URL and SUPABASE_KEY in the script.")
        print("  Use the service_role key (Settings → API → service_role secret).")
        return

    # ── Connect ──────────────────────────────────────────────────────────
    sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Connected to Supabase")

    # ── Load ─────────────────────────────────────────────────────────────
    print(f"\nLoading luxury CSV   : {LUXE_CSV}")
    luxe_df = load_luxury(LUXE_CSV)
    print(f"  → {len(luxe_df):,} rows")

    print(f"Loading ordinary CSV : {ORDINARY_CSV}")
    ordinary_df = load_ordinary(ORDINARY_CSV)
    print(f"  → {len(ordinary_df):,} rows")

    # ── Merge ─────────────────────────────────────────────────────────────
    df = pd.concat([luxe_df, ordinary_df], ignore_index=True)
    print(f"\nMerged total         : {len(df):,} rows")

    # ── Deduplicate ───────────────────────────────────────────────────────
    df = deduplicate(df)

    # ── Upload ────────────────────────────────────────────────────────────
    upload_df(sb, df)

    print("\n  Go to Supabase → Table Editor → cars to verify.")


if __name__ == "__main__":
    main()