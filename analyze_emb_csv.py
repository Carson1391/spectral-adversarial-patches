"""
analyze_emb_csv.py
Parses webcam embedding CSV logs and compares capture states.
Shows per-scale L2 norms, cosine shifts, top-channel deltas between phases.

Usage:
  python analyze_emb_csv.py [path_to_csv]
  (defaults to most recent CSV in outputs_clothing/webcam_emb_logs/)
"""

import os
import sys
import csv
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "outputs_clothing", "webcam_emb_logs")

def find_latest_csv():
    csvs = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.csv') and f.startswith('emb_')])
    if not csvs:
        return None
    return os.path.join(LOG_DIR, csvs[-1])

def parse_csv(path):
    """Parse CSV, handling the header-not-first-row bug from earlier runs."""
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Find header row — it starts with 'timestamp'
    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0] == 'timestamp':
            header_idx = i
            break

    if header_idx is None:
        print("ERROR: No header row found in CSV")
        return None, []

    header = rows[header_idx]
    data_rows = rows[header_idx + 1:]

    # Parse data rows — skip any stray header rows
    parsed = []
    for row in data_rows:
        if not row or len(row) < 12:
            continue
        if row[0] == 'timestamp':
            continue  # stray header
        entry = {
            'timestamp': float(row[0]),
            'type': row[1],
            'track_id': row[2],
            'bbox': (row[3], row[4], row[5], row[6]),
            'conf': float(row[7]) if row[7] else 0.0,
            'cos_overall': float(row[8]) if row[8] else 1.0,
            'shift_overall': float(row[9]) if row[9] else 0.0,
            'n_persons': int(row[10]) if row[10] else 0,
            'n_switches': int(row[11]) if row[11] else 0,
        }
        # Per-scale cos and L2
        n_scales = 3  # YOLOv3 has 3 detection scales
        entry['cos_scales'] = []
        entry['l2_scales'] = []
        for s in range(n_scales):
            cos_col = 12 + s
            l2_col = 12 + n_scales + s
            if cos_col < len(row) and row[cos_col]:
                entry['cos_scales'].append(float(row[cos_col]))
            if l2_col < len(row) and row[l2_col]:
                entry['l2_scales'].append(float(row[l2_col]))

        # Embedding dims start after cos+l2 columns
        emb_start = 12 + 2 * n_scales
        emb_vals = []
        for v in row[emb_start:]:
            if v and v != '' and not v.startswith('emb_'):
                try:
                    emb_vals.append(float(v))
                except ValueError:
                    break
        entry['embedding'] = np.array(emb_vals)

        # crop_path is last column
        if row[-1] and ('crop' in row[-1] or row[-1].endswith('.jpg')):
            entry['crop_path'] = row[-1]
        else:
            entry['crop_path'] = ''

        parsed.append(entry)

    return header, parsed

def detect_phases(rows):
    """Detect phases based on n_persons and track_id changes.
    Phase 1: empty frame (WHOLE only, n_persons=0)
    Phase 2: person without patch (PERSON rows appear, baseline cos ~1.0)
    Phase 3: person with patch (PERSON rows, cos drops / shift increases)
    """
    phases = []
    current_phase = {'name': 'init', 'rows': [], 'start_ts': 0}

    # Simple approach: split by n_persons transitions and large shift jumps
    prev_n_persons = 0
    prev_shift = 0.0
    phase_idx = 0

    for r in rows:
        n_p = r['n_persons']
        shift = r['shift_overall']

        # Phase transitions
        if n_p == 0 and prev_n_persons > 0:
            # Person left frame
            if current_phase['rows']:
                phases.append(current_phase)
            phase_idx += 1
            current_phase = {'name': f'empty_{phase_idx}', 'rows': [], 'start_ts': r['timestamp']}
        elif n_p > 0 and prev_n_persons == 0:
            # Person entered frame
            if current_phase['rows']:
                phases.append(current_phase)
            phase_idx += 1
            current_phase = {'name': f'person_{phase_idx}', 'rows': [], 'start_ts': r['timestamp']}

        current_phase['rows'].append(r)
        prev_n_persons = n_p
        prev_shift = shift

    if current_phase['rows']:
        phases.append(current_phase)

    return phases

def compute_phase_stats(phase):
    """Compute statistics for a phase."""
    rows = phase['rows']
    whole_rows = [r for r in rows if r['type'] == 'WHOLE']
    person_rows = [r for r in rows if r['type'] == 'PERSON']

    stats = {
        'name': phase['name'],
        'n_whole': len(whole_rows),
        'n_person': len(person_rows),
        'duration': rows[-1]['timestamp'] - rows[0]['timestamp'] if len(rows) > 1 else 0,
    }

    # Whole-frame stats
    if whole_rows:
        stats['whole_cos_mean'] = np.mean([r['cos_overall'] for r in whole_rows])
        stats['whole_cos_std'] = np.std([r['cos_overall'] for r in whole_rows])
        stats['whole_shift_mean'] = np.mean([r['shift_overall'] for r in whole_rows])
        stats['whole_shift_max'] = np.max([r['shift_overall'] for r in whole_rows])
        # Per-scale L2
        for s in range(min(3, len(whole_rows[0]['l2_scales']))):
            l2s = [r['l2_scales'][s] for r in whole_rows if len(r['l2_scales']) > s]
            if l2s:
                stats[f'whole_l2_s{s}_mean'] = np.mean(l2s)
                stats[f'whole_l2_s{s}_std'] = np.std(l2s)
        # Per-scale cos
        for s in range(min(3, len(whole_rows[0]['cos_scales']))):
            coss = [r['cos_scales'][s] for r in whole_rows if len(r['cos_scales']) > s]
            if coss:
                stats[f'whole_cos_s{s}_mean'] = np.mean(coss)

    # Person stats
    if person_rows:
        # Group by track_id
        by_tid = defaultdict(list)
        for r in person_rows:
            tid = r['track_id']
            by_tid[tid].append(r)

        stats['person_tids'] = sorted(by_tid.keys())
        stats['person_conf_mean'] = np.mean([r['conf'] for r in person_rows])
        stats['person_conf_min'] = np.min([r['conf'] for r in person_rows])
        stats['person_cos_mean'] = np.mean([r['cos_overall'] for r in person_rows])
        stats['person_shift_mean'] = np.mean([r['shift_overall'] for r in person_rows])
        stats['person_shift_max'] = np.max([r['shift_overall'] for r in person_rows])

        for s in range(min(3, len(person_rows[0]['l2_scales']))):
            l2s = [r['l2_scales'][s] for r in person_rows if len(r['l2_scales']) > s]
            if l2s:
                stats[f'person_l2_s{s}_mean'] = np.mean(l2s)
        for s in range(min(3, len(person_rows[0]['cos_scales']))):
            coss = [r['cos_scales'][s] for r in person_rows if len(r['cos_scales']) > s]
            if coss:
                stats[f'person_cos_s{s}_mean'] = np.mean(coss)

        # Track switches
        switches = [r['n_switches'] for r in person_rows]
        stats['n_switches'] = switches[-1] if switches else 0

    return stats

def compare_embeddings(rows):
    """Find the embedding vectors for each phase and compute per-channel deltas."""
    whole_rows = [r for r in rows if r['type'] == 'WHOLE']
    person_rows = [r for r in rows if r['type'] == 'PERSON']

    # Average embeddings per phase
    phases = detect_phases(rows)
    phase_embs = {}
    for ph in phases:
        wr = [r for r in ph['rows'] if r['type'] == 'WHOLE']
        pr = [r for r in ph['rows'] if r['type'] == 'PERSON']
        if wr:
            embs = [r['embedding'] for r in wr if len(r['embedding']) > 0]
            if embs:
                phase_embs[ph['name'] + '_whole'] = np.mean(embs, axis=0)
        if pr:
            embs = [r['embedding'] for r in pr if len(r['embedding']) > 0]
            if embs:
                phase_embs[ph['name'] + '_person'] = np.mean(embs, axis=0)

    return phase_embs

def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = find_latest_csv()

    if not csv_path or not os.path.exists(csv_path):
        print("No CSV found.")
        return

    print(f"Analyzing: {csv_path}")
    print(f"File size: {os.path.getsize(csv_path) / 1024:.1f} KB")
    print("=" * 80)

    header, rows = parse_csv(csv_path)
    if not rows:
        print("No data rows found.")
        return

    print(f"Total rows: {len(rows)}")
    whole = [r for r in rows if r['type'] == 'WHOLE']
    person = [r for r in rows if r['type'] == 'PERSON']
    print(f"  WHOLE rows: {len(whole)}")
    print(f"  PERSON rows: {len(person)}")

    if whole:
        print(f"  Time span: {whole[0]['timestamp']:.1f} -> {whole[-1]['timestamp']:.1f} ({whole[-1]['timestamp'] - whole[0]['timestamp']:.1f}s)")
    if person:
        tids = set(r['track_id'] for r in person)
        print(f"  Track IDs seen: {sorted(tids)}")
        confs = [r['conf'] for r in person]
        print(f"  Confidence: min={min(confs):.3f} max={max(confs):.3f} mean={np.mean(confs):.3f}")

    # Detect scale dims
    if whole and whole[0]['embedding'].shape[0] > 0:
        total_dims = whole[0]['embedding'].shape[0]
        # YOLOv3: 3 scales, channels vary
        print(f"  Embedding dims per row: {total_dims}")

    print("\n" + "=" * 80)
    print("PHASE DETECTION")
    print("=" * 80)

    phases = detect_phases(rows)
    for ph in phases:
        print(f"\n--- Phase: {ph['name']} ({len(ph['rows'])} rows, {ph['rows'][0]['timestamp']:.1f}s) ---")
        stats = compute_phase_stats(ph)
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k:30s} = {v:.6f}")
            else:
                print(f"  {k:30s} = {v}")

    print("\n" + "=" * 80)
    print("EMBEDDING COMPARISON (per-channel deltas)")
    print("=" * 80)

    phase_embs = compare_embeddings(rows)
    names = list(phase_embs.keys())
    print(f"\nPhase embedding keys: {names}")

    if len(names) >= 2:
        # Compare each pair
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                e1 = phase_embs[names[i]]
                e2 = phase_embs[names[j]]
                if len(e1) != len(e2):
                    print(f"\n{names[i]} vs {names[j]}: dim mismatch ({len(e1)} vs {len(e2)}), skipping")
                    continue
                diff = e2 - e1
                cos = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
                l2_dist = np.linalg.norm(diff)
                # Top 20 channels by absolute delta
                top_idx = np.argsort(np.abs(diff))[-20:][::-1]
                print(f"\n{names[i]} vs {names[j]}:")
                print(f"  cosine = {cos:.6f}")
                print(f"  L2 distance = {l2_dist:.4f}")
                print(f"  Top 20 channel deltas:")
                for idx in top_idx:
                    # Determine which scale this channel belongs to
                    # YOLOv3: s0=128, s1=256, s2=512 (typical)
                    print(f"    d{idx:4d}: {e1[idx]:+.6f} -> {e2[idx]:+.6f}  delta={diff[idx]:+.6f}")

    # Save summary to CSV
    summary_path = csv_path.replace('.csv', '_analysis.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['phase', 'metric', 'value'])
        for ph in phases:
            stats = compute_phase_stats(ph)
            for k, v in stats.items():
                if k == 'name':
                    continue
                w.writerow([ph['name'], k, f"{v:.6f}" if isinstance(v, float) else v])
    print(f"\nSummary saved to: {summary_path}")

    # Save per-channel deltas if we have enough phases
    if len(names) >= 2:
        delta_path = csv_path.replace('.csv', '_channel_deltas.csv')
        with open(delta_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['channel_idx'] + [f"{n1}_vs_{n2}" for i, n1 in enumerate(names) for n2 in names[i+1:]])
            max_dim = max(len(e) for e in phase_embs.values())
            for idx in range(max_dim):
                row = [idx]
                for i, n1 in enumerate(names):
                    for j, n2 in enumerate(names):
                        if j <= i:
                            continue
                        e1 = phase_embs[n1]
                        e2 = phase_embs[n2]
                        if idx < len(e1) and idx < len(e2):
                            row.append(f"{e2[idx] - e1[idx]:.8f}")
                        else:
                            row.append('')
                w.writerow(row)
        print(f"Channel deltas saved to: {delta_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

if __name__ == "__main__":
    main()
