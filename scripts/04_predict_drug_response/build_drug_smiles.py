"""Build ``resources/drug_smiles.csv`` from PubChem for McFarland + SciPlex drug names."""

from __future__ import annotations

import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, setup_project

setup_project()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

RESOURCES = Path(config.RESOURCES_DIR)
OUT_PATH = RESOURCES / 'drug_smiles.csv'

# Names that PubChem cannot resolve as small molecules (CRISPR, vehicle, etc.)
NON_SMALL_MOLECULE = {
    'sgGPX4',
    'sgOR2J2',
    'Vehicle',
    'ctrl',
    'NA',
}


def _pubchem_smiles(name: str, timeout: float = 15.0) -> str:
    """Return canonical SMILES from PubChem name lookup, or empty string."""
    query = urllib.parse.quote(name.strip())
    url = (
        'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/'
        f'{query}/property/CanonicalSMILES/JSON'
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        props = payload['PropertyTable']['Properties'][0]
        return str(props.get('CanonicalSMILES', '') or props.get('ConnectivitySMILES', '')).strip()
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError, TimeoutError) as exc:
        logger.debug('PubChem miss for %r: %s', name, exc)
        return ''


def _alias_lookup(name: str) -> str:
    """Try alternate names for common aliases."""
    aliases = {
        'JQ1': '(+)-JQ1',
        'Tanespimycin (17-AAG)': '17-AAG',
        'Glesatinib?(MGCD265)': 'MGCD265',
    }
    alt = aliases.get(name.strip())
    if alt:
        smi = _pubchem_smiles(alt)
        if smi:
            return smi
    return ''


def collect_drug_names() -> list[str]:
    mcf = pd.read_csv(RESOURCES / 'mcfarland_drug_to_perturbation.csv')
    sci = pd.read_csv(RESOURCES / 'sciplex_drug_to_perturbation.csv', index_col=0)
    names = set(mcf['drug'].astype(str).str.strip())
    names.update(sci['product_name'].astype(str).str.strip())
    names -= {'', 'nan'}
    return sorted(names)


def build_table(sleep_s: float = 0.25) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for name in collect_drug_names():
        if name in NON_SMALL_MOLECULE:
            rows.append({'drug_key': name, 'smiles': '', 'source': 'non_small_molecule'})
            continue
        smi = _pubchem_smiles(name)
        if not smi:
            smi = _alias_lookup(name)
        source = 'pubchem' if smi else 'missing'
        rows.append({'drug_key': name, 'smiles': smi, 'source': source})
        if sleep_s:
            time.sleep(sleep_s)
        logger.info('%s -> %s (%s)', name, smi[:40] + ('...' if len(smi) > 40 else ''), source)

    df = pd.DataFrame(rows)
    n_hit = int((df['smiles'] != '').sum())
    logger.info('Resolved SMILES for %d / %d drugs', n_hit, len(df))
    return df


def write_drug_smiles_table(
    out_path: Path | None = None,
    sleep_s: float = 0.25,
) -> Path:
    """Fetch SMILES from PubChem and write ``resources/drug_smiles.csv``."""
    target = out_path or OUT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    df = build_table(sleep_s=sleep_s)
    df.to_csv(target, index=False)
    logger.info('Wrote %s (%d drugs, %d with SMILES)', target, len(df), int((df['smiles'] != '').sum()))
    return target


def main() -> None:
    smiles_path = write_drug_smiles_table()
    from drug_fingerprints import write_ecfp_cache

    write_ecfp_cache(smiles_table=pd.read_csv(smiles_path))


if __name__ == '__main__':
    main()
