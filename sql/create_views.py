"""
SQL LAYER — VIEW CREATOR
Creates all 4 analytical views in PostgreSQL.
Run: python sql/create_views.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
engine = create_engine(DB_URL)

# ── Step 1: Drop all views first in dependency order ──────────
print("Dropping existing views...")
drop_statements = [
    "DROP VIEW IF EXISTS vw_segment_risk_decomposition CASCADE",
    "DROP VIEW IF EXISTS vw_anomaly_detection CASCADE",
    "DROP VIEW IF EXISTS vw_portfolio_cashflow_forecast CASCADE",
    "DROP VIEW IF EXISTS vw_customer_risk_scorecard CASCADE",
]

with engine.connect() as conn:
    for stmt in drop_statements:
        conn.execute(text(stmt))
    conn.commit()
print("  All existing views dropped.")

# ── Step 2: Read SQL file ─────────────────────────────────────
sql_path = os.path.join(os.path.dirname(__file__), "create_views.sql")
with open(sql_path, "r", encoding="utf-8") as f:
    sql = f.read()

# ── Step 3: Extract CREATE VIEW blocks only ───────────────────
# Split on CREATE VIEW keyword to isolate each view definition
import re

# Remove DROP statements from file — we already dropped above
# Remove COMMENT statements — they cause parsing issues
lines = sql.split("\n")
filtered_lines = []
skip = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("DROP VIEW"):
        continue
    if stripped.startswith("COMMENT ON"):
        skip = True
    if skip:
        if stripped.endswith(";"):
            skip = False
        continue
    # Skip the final verification SELECT
    if stripped.startswith("SELECT") and "UNION ALL" in sql[sql.find(stripped):sql.find(stripped)+500]:
        skip = True
        continue
    filtered_lines.append(line)

filtered_sql = "\n".join(filtered_lines)

# Split into individual CREATE VIEW statements
# Each ends with a semicolon on its own line or after ORDER BY clause
view_blocks = re.split(r'(?=CREATE VIEW)', filtered_sql)
view_blocks = [b.strip() for b in view_blocks if "CREATE VIEW" in b]

print(f"\nFound {len(view_blocks)} view definitions to create.")

# ── Step 4: Create each view in its own transaction ───────────
view_names = []
for block in view_blocks:
    # Extract view name for logging
    match = re.search(r'CREATE VIEW (\w+)', block)
    view_name = match.group(1) if match else "unknown"
    view_names.append(view_name)

    # Remove trailing semicolon if present
    block = block.rstrip().rstrip(";").strip()

    print(f"\n  Creating {view_name}...")
    try:
        with engine.connect() as conn:
            conn.execute(text(block))
            conn.commit()
        print(f"  ✓ {view_name} created successfully")
    except Exception as e:
        print(f"  ✗ {view_name} FAILED: {e}")

# ── Step 5: Verify all views ──────────────────────────────────
print("\n" + "=" * 55)
print("VIEW VERIFICATION")
print("=" * 55)
print(f"\n  {'View':<40} {'Rows':>12}")
print(f"  {'─'*39} {'─'*12}")

with engine.connect() as conn:
    for v in view_names:
        try:
            r = conn.execute(text(f"SELECT COUNT(*) FROM {v}"))
            count = r.fetchone()[0]
            status = "✓" if count > 0 else "⚠ zero rows"
            print(f"  {v:<40} {count:>12,}  {status}")
        except Exception as e:
            print(f"  {v:<40} {'ERROR':>12}  {e}")

print("\nDone. Next: connect Power BI to PostgreSQL")