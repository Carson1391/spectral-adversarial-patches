import csv
rows = list(csv.DictReader(open('outputs_clothing/forward_analysis/patch_pipeline/pipeline_results.csv')))
for r in rows:
    if r['pattern'] in ('digits_196','k12_stripes') and r['patch_size']=='xlarge_16pct':
        print(f"{r['patch_size']:12s} {r['pattern']:15s} amp={float(r['amplitude']):.3f} W_L2={float(r['avg_wearer_l2']):6.2f} B_L2={float(r['avg_bystander_l2']):5.2f} W_cos={float(r['avg_wearer_cos']):.4f} P={r['total_persons']}")
