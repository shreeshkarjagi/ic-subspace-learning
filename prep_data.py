#load patient data, 30s segmentation
import os
import sys
import json
import numpy as np
from forward_model import stream_to_amp
from load_raw import load_patient


#a patient dir is any subdir with percept jsons somewhere under it
def _discover_patients(data_dir):
    pats = []
    for name in sorted(os.listdir(data_dir)):
        pdir = os.path.join(data_dir, name)
        if not os.path.isdir(pdir): continue
        for root, _, files in os.walk(pdir):
            if any(f.endswith('.json') for f in files):
                pats.append(name)
                break
    return pats


Q = 0.11
SEG_DUR = 30.0  #seconds, matches percept snapshot window


def load_and_segment(data_dir):
    all_hd = {}
    summary = {}

    for pid in _discover_patients(data_dir):
        pdir = os.path.join(data_dir, pid)
        if not os.path.isdir(pdir):
            print(f'  {pid}: not found, skipping')
            continue

        pdata = load_patient(pid, pdir, event_type=None)
        anon = pid

        for hemi in ('left', 'right'):
            #indefinite streaming only, and long enough for at least one segment
            streams = [s for s in pdata['streaming']
                       if s['hemisphere'] == hemi
                       and s['duration_s'] >= SEG_DUR
                       and s.get('source') == 'indefinite']
            if not streams:
                continue

            amps_list = []
            recording_ids = []
            freqs_ref = None
            rec_idx = 0
            n_recordings_used = 0

            for st in streams:
                ts = st['timeseries']
                fs = st['fs']
                seg_samples = int(SEG_DUR * fs)
                n_seg = len(ts) // seg_samples

                if n_seg == 0:
                    rec_idx += 1
                    continue

                n_recordings_used += 1
                for seg_i in range(n_seg):
                    chunk = ts[seg_i * seg_samples:(seg_i + 1) * seg_samples]
                    f, amp = stream_to_amp(chunk, fs)
                    if len(amp) < 20:
                        continue
                    if freqs_ref is None:
                        freqs_ref = f
                    F = min(len(amp), len(freqs_ref))
                    amps_list.append(amp[:F])
                    recording_ids.append(rec_idx)

                rec_idx += 1

            if not amps_list:
                continue

            #fs can differ between recordings, so cut everything to the shortest spectrum
            ml = min(len(a) for a in amps_list)
            amps = np.array([a[:ml] for a in amps_list])
            freqs = freqs_ref[:ml]
            rec_ids = np.array(recording_ids)

            key = f'{anon}_{hemi}'
            all_hd[key] = {
                'freqs': freqs,
                'amps': amps,
                'recording_ids': rec_ids,
            }

            n_seg_total = len(amps_list)
            n_recs = len(np.unique(rec_ids))
            summary[key] = {
                'n_segments': n_seg_total,
                'n_recordings': n_recs,
                'n_bins': ml,
            }
            print(f'  {key}: {n_recs} recordings -> {n_seg_total} segments '
                  f'({ml} bins)')

    return all_hd, summary


def main():
    if len(sys.argv) < 2:
        print('Usage: python prep_data.py /path/to/RawData')
        return

    data_dir = sys.argv[1]
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'data')
    os.makedirs(outdir, exist_ok=True)

    print(f'Loading with {SEG_DUR}s segmentation...')
    all_hd, summary = load_and_segment(data_dir)

    #one npz per hemisphere
    for key, hd in all_hd.items():
        path = os.path.join(outdir, f'{key}.npz')
        np.savez_compressed(path,
                            freqs=hd['freqs'],
                            amps=hd['amps'],
                            recording_ids=hd['recording_ids'])
        print(f'  saved {path}')

    #cohort totals go in the summary json
    total_seg = sum(s['n_segments'] for s in summary.values())
    total_rec = sum(s['n_recordings'] for s in summary.values())
    summary['_cohort'] = {
        'n_hemispheres': len(summary),
        'n_segments_total': total_seg,
        'n_recordings_total': total_rec,
        'q': Q,
        'seg_dur': SEG_DUR,
    }
    with open(os.path.join(outdir, 'data_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nCohort: {len(all_hd)} hemispheres, {total_rec} recordings, '
          f'{total_seg} segments')
    print(f'Saved to {outdir}/')


if __name__ == '__main__':
    main()
