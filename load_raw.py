#read raw percept json files

import os
import json
import datetime
import numpy as np


#percept writes datetimes in a handful of formats depending on export version
def _parse_dt(dt_str):
    if not dt_str:
        return None
    for fmt in ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S%z'):
        try:
            return datetime.datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    return None


def _hemi_from_key(key):
    k = key.upper()
    if 'LEFT' in k:
        return 'left'
    if 'RIGHT' in k:
        return 'right'
    return None


def _hemi_from_channel(ch):
    ch = ch.upper()
    if 'LEFT' in ch:
        return 'left'
    if 'RIGHT' in ch:
        return 'right'
    return 'unknown'


#layout is PATIENT_ID/0_LFP/SESSION_FOLDER/*.json but the nesting is not
#reliable, so walk the whole tree
def find_jsons(lfp_dir):
    paths = []
    for root, _dirs, files in os.walk(lfp_dir):
        for fname in sorted(files):
            if fname.lower().endswith('.json'):
                fp = os.path.join(root, fname)
                if os.path.getsize(fp) > 10:
                    paths.append(fp)
    return sorted(paths)


#yields (hemisphere, fftbin, freq) out of whatever key layout this export used
def _extract_hemisphere_data(block):
    if block is None or not isinstance(block, dict):
        return

    #flat structure, rare
    if 'FFTBinData' in block and 'Frequency' in block:
        fft = block['FFTBinData']
        freq = block['Frequency']
        if fft and freq and len(fft) == len(freq):
            yield 'unknown', np.array(fft, dtype=np.float64), \
                  np.array(freq, dtype=np.float64)
        return

    #usual case: sub-keys carry the hemisphere, either
    #'HemisphereLocationDef.Left' or just something containing Left/Right
    for key, hdata in block.items():
        if not isinstance(hdata, dict):
            continue

        hemi = _hemi_from_key(key)
        if hemi is None:
            continue

        fft_raw = hdata.get('FFTBinData', [])
        freq_raw = hdata.get('Frequency', [])

        if not fft_raw or not freq_raw:
            continue
        if len(fft_raw) != len(freq_raw):
            continue

        yield hemi, np.array(fft_raw, dtype=np.float64), \
              np.array(freq_raw, dtype=np.float64)


#event_type=None pulls every event type (Morning, Evening, ...), otherwise it
#matches case-insensitively. fftbin comes back as raw FFTBinData, untouched
def load_snapshots(json_path, event_type=None):
    with open(json_path, 'r') as f:
        data = json.load(f)

    diag = data.get('DiagnosticData') or {}
    events = diag.get('LfpFrequencySnapshotEvents', [])
    results = []

    for event in events:
        ename = event.get('EventName', '').strip()

        if event_type is not None:
            if ename.lower() != event_type.lower():
                continue

        dt = _parse_dt(event.get('DateTime', ''))

        #two keys can hold the hemisphere block. the nested
        #'LfpFrequencySnapshotEvents' one is what actually works across all 7
        #patients; 'LFP' also shows up in the event dict but with a different
        #internal structure. try each, keep whichever yields data
        found = False
        for snap_key in ('LfpFrequencySnapshotEvents', 'LFP'):
            snap_block = event.get(snap_key)
            if snap_block is None:
                continue
            for hemi, fftbin, freq in _extract_hemisphere_data(snap_block):
                results.append({
                    'hemisphere': hemi,
                    'fftbin': fftbin,
                    'frequency': freq,
                    'datetime': dt,
                    'event_name': ename,
                    'json_path': json_path,
                })
                found = True
            if found:
                break  #don't double-count from both keys

    return results


def load_streaming(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = []

    #stim off, all 6 channels
    for stream in data.get('IndefiniteStreaming', []):
        channel = stream.get('Channel', 'Unknown')
        ts = np.array(stream.get('TimeDomainData', []), dtype=np.float64)
        fs = float(stream.get('SampleRateInHz', 250.0))
        dt = _parse_dt(stream.get('FirstPacketDateTime', ''))

        if len(ts) < int(fs):  #less than a second
            continue

        results.append({
            'hemisphere': _hemi_from_channel(channel),
            'timeseries': ts,
            'fs': fs,
            'datetime': dt,
            'channel': channel,
            'duration_s': len(ts) / fs,
            'source': 'indefinite',
        })

    #stim on, 1-2 channels
    for stream in data.get('BrainSenseTimeDomain', []):
        channel = stream.get('Channel', 'Unknown')
        ts = np.array(stream.get('TimeDomainData', []), dtype=np.float64)
        fs = float(stream.get('SampleRateInHz', 250.0))
        dt = _parse_dt(stream.get('FirstPacketDateTime', ''))

        if len(ts) < int(fs):
            continue

        results.append({
            'hemisphere': _hemi_from_channel(channel),
            'timeseries': ts,
            'fs': fs,
            'datetime': dt,
            'channel': channel,
            'duration_s': len(ts) / fs,
            'source': 'brainsense',
        })

    return results


def load_patient(patient_id, lfp_dir, event_type=None):
    json_paths = find_jsons(lfp_dir)

    snaps = {'left': [], 'right': []}
    streams = []

    #dedup on (datetime, event_name, hemisphere) rather than date alone, so
    #morning and evening on the same day stay separate while the same snapshot
    #exported into several jsons only counts once
    seen_snap = set()
    seen_stream = set()
    _snap_counter = 0  #fallback id when there is no datetime
    _stream_counter = 0
    _snap_warned = False
    _stream_warned = False

    for path in json_paths:
        try:
            snap_results = load_snapshots(path, event_type)
            for s in snap_results:
                h = s['hemisphere']
                if h not in snaps:
                    snaps[h] = []

                dt = s['datetime']
                ename = s['event_name']
                if dt is not None:
                    key = (dt, ename, h)
                else:
                    key = ('_none_', _snap_counter, h)
                    _snap_counter += 1

                if key not in seen_snap:
                    seen_snap.add(key)
                    if h in ('left', 'right'):
                        snaps[h].append(s)
                    else:
                        snaps.setdefault('unknown', []).append(s)
        except Exception as e:
            if not _snap_warned:
                print(f'  WARNING: snapshot load failed for '
                      f'{os.path.basename(path)}: {e}')
                _snap_warned = True

        try:
            for r in load_streaming(path):
                h = r['hemisphere']
                dt = r['datetime']
                ch = r['channel']
                if dt is not None:
                    key = (dt, ch)
                else:
                    key = ('_none_', _stream_counter, ch)
                    _stream_counter += 1

                if key not in seen_stream:
                    seen_stream.add(key)
                    streams.append(r)
        except Exception as e:
            if not _stream_warned:
                print(f'  WARNING: streaming load failed for '
                      f'{os.path.basename(path)}: {e}')
                _stream_warned = True

    for h in list(snaps.keys()):
        snaps[h].sort(key=lambda x: x['datetime'] or datetime.datetime.min)

    n_left = len(snaps.get('left', []))
    n_right = len(snaps.get('right', []))
    n_unknown = len(snaps.get('unknown', []))
    n_indef = sum(1 for s in streams if s.get('source') == 'indefinite')
    n_bs = sum(1 for s in streams if s.get('source') == 'brainsense')
    print(f'  {patient_id}: {len(json_paths)} JSONs | '
          f'snaps L={n_left} R={n_right}'
          f'{f" unk={n_unknown}" if n_unknown else ""} | '
          f'streaming: {n_indef} indef + {n_bs} brainsense')

    return {
        'patient_id': patient_id,
        'snapshots': snaps,
        'streaming': streams,
        'json_paths': json_paths,
    }


#call this when load_snapshots comes back empty and you need to see the schema
def inspect_json_structure(json_path, max_events=2):
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f'\n--- {os.path.basename(json_path)} ---')
    print(f'Top-level keys: {list(data.keys())}')

    diag = data.get('DiagnosticData', {})
    if diag:
        print(f'DiagnosticData keys: {list(diag.keys())}')

    events = diag.get('LfpFrequencySnapshotEvents', [])
    print(f'LfpFrequencySnapshotEvents: {len(events)} events')

    for i, ev in enumerate(events[:max_events]):
        print(f'\n  Event {i}:')
        print(f'    Top keys: {list(ev.keys())}')
        print(f'    EventName: {ev.get("EventName")}')
        print(f'    DateTime:  {ev.get("DateTime")}')

        lfp = ev.get('LFP')
        if lfp is not None:
            print(f'    LFP type: {type(lfp).__name__}')
            if isinstance(lfp, dict):
                print(f'    LFP keys: {list(lfp.keys())}')
                for k, v in lfp.items():
                    if isinstance(v, dict):
                        print(f'      {k}: keys={list(v.keys())}')
                        fft = v.get('FFTBinData', [])
                        freq = v.get('Frequency', [])
                        print(f'        FFTBinData: {len(fft)} values'
                              f' (first 3: {fft[:3]})')
                        print(f'        Frequency:  {len(freq)} values'
                              f' (first 3: {freq[:3]})')
        else:
            #older exports
            old = ev.get('LfpFrequencySnapshotEvents')
            if old is not None:
                print(f'    LfpFrequencySnapshotEvents (nested): '
                      f'keys={list(old.keys()) if isinstance(old, dict) else type(old).__name__}')

    streams = data.get('IndefiniteStreaming', [])
    print(f'\nIndefiniteStreaming: {len(streams)} entries')
    if streams:
        s0 = streams[0]
        print(f'  First entry keys: {list(s0.keys())}')
        print(f'  Channel: {s0.get("Channel")}')
        td = s0.get('TimeDomainData', [])
        print(f'  TimeDomainData: {len(td)} samples')
