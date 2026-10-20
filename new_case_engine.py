"""Generate a new SSI input graph from edited matrix parameters; no dynamics/training.

Only the checked-in original OpenSees source files are executed. The Google
Drive service account downloads earthquake and soil DATA, never executable code.
"""
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
from openpyxl import Workbook

FOLDER_MIME = 'application/vnd.google-apps.folder'
MAX_BYTES = 80_000_000
REQUIRED_COLS = [
    'Simulation_No', 'Soil Profile (based on SPT-N)', 'Column height H (m)',
    'Column/pile diameter D (m)', 'Longitudinal steel ratio pl (%)',
    'Transverse steel ratio pt (%)', 'Concrete strength fc (MPa)',
    'Steel yield strength fy (MPa)', 'Number of spans (-)', 'Span length (m)',
    'Number of piles (-)', 'Pile spacing (m)', 'Number of bents (-)',
    'Pile embedment length Lp (m)', 'Peak ground acceleration PGA (g)',
    'Ground motion time step (s)', 'EQ', 'Angle of incidence (deg)',
]


def _session(credentials_json):
    credentials = service_account.Credentials.from_service_account_info(
        json.loads(credentials_json), scopes=['https://www.googleapis.com/auth/drive.readonly'])
    return AuthorizedSession(credentials)


def _children(session, parent):
    out = []
    token = None
    while True:
        params = {'q': "'%s' in parents and trashed = false" % parent,
                  'fields': 'nextPageToken,files(id,name,mimeType,size)', 'pageSize': 1000}
        if token:
            params['pageToken'] = token
        response = session.get('https://www.googleapis.com/drive/v3/files',
                               params=params, timeout=120)
        response.raise_for_status()
        data = response.json()
        out.extend(data.get('files', []))
        token = data.get('nextPageToken')
        if not token:
            return out


def _find(items, name):
    return next((item for item in items if item['name'] == name), None)


def _download(session, metadata, path):
    if metadata is None or metadata.get('mimeType') == FOLDER_MIME:
        raise FileNotFoundError('Input data file is missing from Google Drive')
    if int(metadata.get('size') or 0) > MAX_BYTES:
        raise ValueError('Input data file is larger than the size limit')
    response = session.get('https://www.googleapis.com/drive/v3/files/' + metadata['id'],
                           params={'alt': 'media'}, stream=True, timeout=120)
    response.raise_for_status()
    total = 0
    try:
        with open(path, 'wb') as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError('Input data download exceeded size limit')
                target.write(chunk)
    finally:
        response.close()


def _identifier(value, endings):
    text = str(value).strip()
    for ending in endings:
        if text.lower().endswith(ending):
            text = text[:-len(ending)]
            break
    if text in ('', '.', '..') or not re.fullmatch(r'[A-Za-z0-9_(). -]+', text):
        raise ValueError('Invalid earthquake / soil identifier: %r' % value)
    return text


def _write_matrix(path, edits, source_number):
    values = dict(edits)
    values['Simulation_No'] = int(source_number)
    values['Number of spans (-)'] = int(round(float(values['Number of spans (-)'])))
    values['Number of piles (-)'] = int(round(float(values['Number of piles (-)'])))
    values['Number of bents (-)'] = values['Number of spans (-)'] + 1
    values['EQ'] = _identifier(values['EQ'], ('.out',))
    values['Soil Profile (based on SPT-N)'] = _identifier(
        values['Soil Profile (based on SPT-N)'], ('.txt', '.csv'))
    for key in REQUIRED_COLS:
        if key not in values:
            raise ValueError('Missing parameter: ' + key)
    for key in REQUIRED_COLS:
        if key not in ('Simulation_No', 'EQ', 'Soil Profile (based on SPT-N)'):
            number = float(values[key])
            if not np.isfinite(number):
                raise ValueError('Non-finite input: ' + key)
    if values['Number of piles (-)'] < 2 or values['Number of spans (-)'] < 1:
        raise ValueError('Require at least two piles and one span')
    if not 0 <= float(values['Angle of incidence (deg)']) <= 360:
        raise ValueError('Incidence angle must be between 0 and 360 degrees')
    if (float(values['Pile embedment length Lp (m)']) <=
            float(values['Column height H (m)'])):
        raise ValueError('Pile embedment must exceed column height')
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'Simulation_Matrix'
    sheet.append(REQUIRED_COLS)
    sheet.append([values[col] for col in REQUIRED_COLS])
    workbook.save(path)
    return values


def build_new_case(edited_inputs, source_number, drive_folder_id, credentials_json,
                   project_root):
    """Return (HDF5 filename, case id, temp owner, graph statistics).

    Caller must keep temp owner alive until all checkpoint inference completes.
    """
    owner = tempfile.TemporaryDirectory(prefix='ssi_new_ml_case_')
    try:
        work = Path(owner.name)
        eqdir = work / 'EQs'
        soildir = work / 'soil_profiles'
        eqdir.mkdir()
        soildir.mkdir()
        inputs = _write_matrix(work / 'edited_matrix.xlsx', edited_inputs, source_number)
        session = _session(credentials_json)
        children = _children(session, drive_folder_id)
        eq_folder = _find(children, 'EQs')
        soil_folder = _find(children, 'soil_profiles')
        if not eq_folder or not soil_folder:
            raise FileNotFoundError('EQs or soil_profiles is not present in the shared Drive folder')
        eqid = inputs['EQ']
        soilid = inputs['Soil Profile (based on SPT-N)']
        eqitems = _children(session, eq_folder['id'])
        soilitems = _children(session, soil_folder['id'])
        _download(session, _find(eqitems, eqid + '.out'), eqdir / (eqid + '.out'))
        source_soil = _find(soilitems, soilid + '.csv') or _find(soilitems, soilid + '.txt')
        if source_soil is None:
            raise FileNotFoundError('Soil profile %s was not found' % soilid)
        _download(session, source_soil, soildir / source_soil['name'])
        project = Path(project_root).resolve()
        script = project / 'extract_ml_features_v5_external_dirs.py'
        if not script.is_file():
            raise FileNotFoundError('Feature extractor is missing from deployed GitHub files')
        output = work / 'new_input_graph.h5'
        argv = [sys.executable, str(script), '--matrix-file', str(work / 'edited_matrix.xlsx'),
                '--project-dir', str(project), '--eqs-dir', str(eqdir),
                '--soil-profiles-dir', str(soildir), '--output', str(output),
                '--simulations', str(source_number), '--prediction-only']
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        env['PYTHONPATH'] = str(project) + os.pathsep + env.get('PYTHONPATH', '')
        try:
            proc = subprocess.run(argv, cwd=work, env=env, timeout=300,
                                  text=True, capture_output=True)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Feature-only model generation exceeded five minutes') from exc
        if proc.returncode != 0:
            log = '\n'.join((proc.stdout + '\n' + proc.stderr).splitlines()[-24:])
            raise RuntimeError('Feature-only OpenSees generation failed:\n' + log)
        if not output.is_file():
            raise RuntimeError('No input-only graph was generated')
        from ssi_ml_surrogate_v1.hdf5_data import read_graph_case
        case_id = 'simulation_' + str(int(source_number))
        graph = read_graph_case(str(output), case_id)
        dims = (len(graph.global_features), graph.node_features.shape[1],
                graph.edge_features.shape[1])
        if dims != (19, 6, 9):
            raise ValueError('Generated features do not match trained dimensions: %r' % (dims,))
        xyz = graph.node_features[:, 2]
        return str(output), case_id, owner, {
            'node_z_mean_m': float(xyz.mean()),
            'node_z_min_m': float(xyz.min()),
            'node_z_max_m': float(xyz.max()),
            'node_count': len(xyz),
            'edge_count': graph.edge_index.shape[1],
        }
    except BaseException:
        owner.cleanup()
        raise
