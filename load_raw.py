"""
Read raw Percept JSON files
"""

import os
import json
import datetime
import numpy as np


def _parse_dt(dt_str):
    """Try several Percept datetime formats. Return datetime or None."""
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
    """Extract hemisphere from a JSON key string."""
    k = key.upper()
    if 'LEFT' in k:
        return 'left'
    if 'RIGHT' in k:
        return 'right'
    return None


def _hemi_from_channel(ch):
    """Extract hemisphere from a streaming channel string."""
    ch = ch.upper()
    if 'LEFT' in ch:
        return 'left'
    if 'RIGHT' in ch:
        return 'right'
    return 'unknown'


# ── JSON discovery ───────────────────────────────────────────────────────

def find_jsons(lfp_dir):
    """
    Find all JSON files under lfp_dir at ANY depth.

    Data layout: PATIENT_ID/0_LFP/SESSION_FOLDER/*.json (depth 3).
    Uses os.walk so it works regardless of nesting.
    """
    paths = []
    for root, _dirs, files in os.walk(lfp_dir):
        for fname in sorted(files):
            if fname.lower().endswith('.json'):
                fp = os.path.join(root, fname)
                if os.path.getsize(fp) > 10:
                    paths.append(fp)
    return sorted(paths)


# ── Snapshot extraction ──────────────────────────────────────────────────

def _extract_hemisphere_data(block):
    """
    Given a dict that should contain hemisphere sub-dicts, yield
    (hemisphere, fftbin_array, freq_array) tuples.

    Handles multiple possible key formats:
      - 'HemisphereLocationDef.Left' / 'HemisphereLocationDef.Right'
      - Keys containing 'Left' / 'Right'
      - Direct 'FFTBinData' at this level (single-hemisphere export)
    """
    if block is None or not isinstance(block, dict):
        return

    # Case 1: block itself has FFTBinData (flat structure, rare).
    if 'FFTBinData' in block and 'Frequency' in block:
        fft = block['FFTBinData']
        freq = block['Frequency']
        if fft and freq and len(fft) == len(freq):
            yield 'unknown', np.array(fft, dtype=np.float64), \
                  np.array(freq, dtype=np.float64)
        return

    # Case 2: sub-keys contain hemisphere identifiers.
    for key, hdata in block.items():
        if not isinstance(hdata, dict):
            continue

        hemi = _hemi_from_key(key)
        if hemi is None:
            # Key doesn't look like a hemisphere — skip.
            continue

        fft_raw = hdata.get('FFTBinData', [])
        freq_raw = hdata.get('Frequency', [])

        if not fft_raw or not freq_raw:
            continue
        if len(fft_raw) != len(freq_raw):
            continue

        yield hemi, np.array(fft_raw, dtype=np.float64), \
              np.array(freq_raw, dtype=np.float64)


def load_snapshots(json_path, event_type=None):
    """
    Extract snapshot data from one JSON file.

    Parameters
    ----------
    json_path  : str
    event_type : str or None
        If None, load ALL event types (Morning, Evening, etc.).
        If a string, only load events matching that name (case-insensitive).

    Returns list of dicts:
      hemisphere    : 'left' | 'right' | 'unknown'
      fftbin        : (F,) float64 — raw FFTBinData values, untouched
      frequency     : (F,) float64 — frequency axis in Hz
      datetime      : datetime | None
      event_name    : str
      json_path     : str
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    diag = data.get('DiagnosticData') or {}
    events = diag.get('LfpFrequencySnapshotEvents', [])
    results = []

    for event in events:
        ename = event.get('EventName', '').strip()

        # Filter by event type if requested.
        if event_type is not None:
            if ename.lower() != event_type.lower():
                continue

        dt = _parse_dt(event.get('DateTime', ''))

        # Try BOTH possible keys for hemisphere data.
        # Key 1: 'LfpFrequencySnapshotEvents' (nested) — confirmed working
        #         in raw_fftbin_analysis.py across all 7 patients.
        # Key 2: 'LFP' — exists in the event dict per diagnose.py but may
        #         have a different internal structure.
        # Strategy: try each, use whichever yields hemisphere data.
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
                break  # Don't double-count from both keys.

    return results


# ── Streaming extraction ─────────────────────────────────────────────────

def load_streaming(json_path):
    """
    Extract streaming timeseries from one JSON file.

    Checks BOTH streaming modalities:
      - IndefiniteStreaming: all 6 channels, stim OFF
      - BrainSenseTimeDomain: 1-2 channels, stim ON

    Returns list of dicts:
      hemisphere : 'left' | 'right' | 'unknown'
      timeseries : (T,) float64 — raw time-domain voltage
      fs         : float — sampling rate (Hz)
      datetime   : datetime | None
      channel    : str
      duration_s : float
      source     : 'indefinite' | 'brainsense'
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = []

    # Source 1: IndefiniteStreaming (stim OFF, all channels).
    for stream in data.get('IndefiniteStreaming', []):
        channel = stream.get('Channel', 'Unknown')
        ts = np.array(stream.get('TimeDomainData', []), dtype=np.float64)
        fs = float(stream.get('SampleRateInHz', 250.0))
        dt = _parse_dt(stream.get('FirstPacketDateTime', ''))

        if len(ts) < int(fs):  # less than 1 second
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

    # Source 2: BrainSenseTimeDomain (stim ON, 1-2 channels).
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


# ── Per-patient loader ───────────────────────────────────────────────────

def load_patient(patient_id, lfp_dir, event_type=None):
    """
    Load all snapshots and streaming for one patient.

    Parameters
    ----------
    patient_id : str
    lfp_dir    : str — path to patient directory (contains 0_LFP/, etc.)
    event_type : str or None — None = load all event types

    Returns dict:
      patient_id : str
      snapshots  : {'left': [...], 'right': [...]}
      streaming  : [stream_dicts]
      json_paths : [str]
    """
    json_paths = find_jsons(lfp_dir)

    snaps = {'left': [], 'right': []}
    streams = []

    # Dedup by (datetime, event_name, hemisphere) — not just date.
    # This keeps Morning + Evening on the same day as separate entries,
    # while still deduplicating the same snapshot exported in multiple JSONs.
    seen_snap = set()
    seen_stream = set()
    _snap_counter = 0  # fallback ID for snapshots without datetime
    _stream_counter = 0
    _snap_warned = False
    _stream_warned = False

    for path in json_paths:
        # Snapshots.
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

        # Streaming.
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

    # Sort chronologically.
    for h in list(snaps.keys()):
        snaps[h].sort(key=lambda x: x['datetime'] or datetime.datetime.min)

    # Summary.
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


# ── Diagnostic utility ───────────────────────────────────────────────────

def inspect_json_structure(json_path, max_events=2):
    """
    Print the key structure of a Percept JSON for debugging.
    Call this when load_snapshots returns nothing to diagnose the schema.
    """
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

        # Show LFP structure.
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
            # Try old key.
            old = ev.get('LfpFrequencySnapshotEvents')
            if old is not None:
                print(f'    LfpFrequencySnapshotEvents (nested): '
                      f'keys={list(old.keys()) if isinstance(old, dict) else type(old).__name__}')

    # Streaming.
    streams = data.get('IndefiniteStreaming', [])
    print(f'\nIndefiniteStreaming: {len(streams)} entries')
    if streams:
        s0 = streams[0]
        print(f'  First entry keys: {list(s0.keys())}')
        print(f'  Channel: {s0.get("Channel")}')
        td = s0.get('TimeDomainData', [])
        print(f'  TimeDomainData: {len(td)} samples')