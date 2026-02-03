"""Concise YouTube query generator (cities x templates)

Goal:
- Keep lists concise.
- Anything repeated becomes a template.
- Generates BOTH:
  1) city-expanded queries (CITY + TEMPLATE)
  2) template-only queries (no city)

NEW:
- Adds city walking / street tour templates (good street view + doors).

Outputs:
- out/queries.txt : unique queries, one per line
- out/queries.csv : structured rows (mode, city, template, query)

Run:
  python gen_queries.py
"""

from __future__ import annotations

import csv
import os
from typing import Iterable, List, Tuple

# -------------------------
# 1) Concise city list
# -------------------------
CITIES: List[str] = [
    # Global
    "New York City", "London", "Singapore", "Hong Kong", "Dubai", "Tokyo", "Sydney",
    "Toronto", "San Francisco", "Los Angeles", "Shanghai", "Shenzhen", "Seoul", "Beijing",
    # India
    "Delhi NCR", "Mumbai", "Bengaluru", "Hyderabad", "Pune", "Chennai",
]

# -------------------------
# 2) Concise templates
# -------------------------
# Use {CITY} as a prefix token. Template-only versions are generated automatically.

OFFICE_TEMPLATES: List[str] = [
    "{CITY} office space for rent walkthrough",
    "{CITY} office space for lease walkthrough",
    "{CITY} office space for sale walkthrough",
    "{CITY} coworking space tour",
    "{CITY} office interior walkthrough glass doors",
]

# Delivery: keep it tight, make the option a template token.
# {APP} and {VEHICLE} cover most variants.
DELIVERY_TEMPLATES: List[str] = [
    "{CITY} {APP} delivery {VEHICLE} POV",
    "{CITY} {APP} delivery {VEHICLE} street POV",
    "{CITY} {APP} delivery {VEHICLE} ride along",
    "{CITY} food delivery {VEHICLE} POV",
    "{CITY} delivery driver POV",
    "{CITY} delivery rider bodycam",
    "{CITY} food delivery apartment drop off POV",
    "{CITY} food delivery building lobby POV",
    "{CITY} food delivery door drop off POV",
]

# City walk / street tour: concise but high-yield for street footage + entrances/doors.
WALK_TEMPLATES: List[str] = [
    "{CITY} walking tour",
    "{CITY} walking tour 4K",
    "{CITY} street walk",
    "{CITY} city walk POV",
    "{CITY} night walk",
    "{CITY} rain walk",
    "{CITY} street tour",
]

# Keep options concise (edit as needed)
APPS: List[str] = [
    "DoorDash",
    "Uber Eats",
    "Deliveroo",
]

VEHICLES: List[str] = [
    "ebike",      # main ask
    "bike",
    "scooter",
]

TEMPLATES: List[str] = OFFICE_TEMPLATES + DELIVERY_TEMPLATES + WALK_TEMPLATES


# -------------------------
# 3) Rendering + expansion
# -------------------------

def render(template: str, city: str = "") -> str:
    """Render a template. If city is empty, produces a template-only query."""
    q = template.replace("{CITY}", city).strip()
    return " ".join(q.split())  # tidy spacing


def expand_city_and_nocity() -> Tuple[List[Tuple[str, str, str, str]], List[str]]:
    """Return (rows, queries) for city-expanded and no-city queries."""
    rows: List[Tuple[str, str, str, str]] = []  # mode, city, template, query
    queries: List[str] = []

    # A) City-expanded queries
    for city in CITIES:
        for tmpl in TEMPLATES:
            if "{APP}" in tmpl or "{VEHICLE}" in tmpl:
                for app in (APPS if "{APP}" in tmpl else [""]):
                    for veh in (VEHICLES if "{VEHICLE}" in tmpl else [""]):
                        q = tmpl.replace("{APP}", app).replace("{VEHICLE}", veh)
                        q = render(q, city)
                        rows.append(("with_city", city, tmpl, q))
                        queries.append(q)
            else:
                q = render(tmpl, city)
                rows.append(("with_city", city, tmpl, q))
                queries.append(q)

    # B) Template-only queries
    for tmpl in TEMPLATES:
        if "{APP}" in tmpl or "{VEHICLE}" in tmpl:
            for app in (APPS if "{APP}" in tmpl else [""]):
                for veh in (VEHICLES if "{VEHICLE}" in tmpl else [""]):
                    q = tmpl.replace("{APP}", app).replace("{VEHICLE}", veh)
                    q = render(q, "")
                    if q:
                        rows.append(("no_city", "", tmpl, q))
                        queries.append(q)
        else:
            q = render(tmpl, "")
            if q:
                rows.append(("no_city", "", tmpl, q))
                queries.append(q)

    return rows, queries


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


# -------------------------
# 4) Output
# -------------------------

def main() -> None:
    outdir = "output"
    os.makedirs(outdir, exist_ok=True)

    rows, queries = expand_city_and_nocity()
    unique_queries = dedupe_preserve_order(queries)

    # queries.txt
    with open(os.path.join(outdir, "queries.txt"), "w", encoding="utf-8") as f:
        for q in unique_queries:
            f.write(q + "\n")

    # queries.csv
    with open(os.path.join(outdir, "queries.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "city", "template", "query"])
        for r in rows:
            w.writerow(list(r))

    print(f"Cities: {len(CITIES)}")
    print(f"Office templates: {len(OFFICE_TEMPLATES)}")
    print(f"Delivery templates: {len(DELIVERY_TEMPLATES)}")
    print(f"Walk templates: {len(WALK_TEMPLATES)}")
    print(f"Apps: {len(APPS)} | Vehicles: {len(VEHICLES)}")
    print(f"Unique queries generated: {len(unique_queries)}")
    print(f"Saved to: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
