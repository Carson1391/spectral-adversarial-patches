import csv

rows = list(csv.DictReader(open('outputs_clothing/forward_analysis/patch_pipeline/pipeline_results.csv')))

# Group by pattern + patch_size, find entries with per-layer L2 data
from collections import defaultdict

# Check what columns we have
print("Columns:", list(rows[0].keys()))
print()

# Filter to xlarge_16pct at amp=0.02 (Profile A operating point)
for r in rows:
    if r['patch_size'] == 'xlarge_16pct' and abs(float(r['amplitude']) - 0.02) < 0.001:
        pat = r['pattern']
        wl2 = float(r['avg_wearer_l2'])
        wcos = float(r['avg_wearer_cos'])
        bl2 = float(r['avg_bystander_l2'])
        persons = r['total_persons']
        # Check for per-layer columns
        l81 = r.get('L81_wearer_l2', r.get('wearer_l2_L81', ''))
        l93 = r.get('L93_wearer_l2', r.get('wearer_l2_L93', ''))
        l105 = r.get('L105_wearer_l2', r.get('wearer_l2_L105', ''))
        print(f"{pat:15s} amp=0.02  W_L2={wl2:6.2f}  W_cos={wcos:.4f}  B_L2={bl2:5.2f}  P={persons}  L81={l81}  L93={l93}  L105={l105}")
