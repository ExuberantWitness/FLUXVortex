#!/usr/bin/env python3
"""Reproduce the second-pass public dynamic-stall asset audit.

The audit downloads ZIP central directories and small metadata payloads only.
It never downloads either multi-gigabyte archive as a whole.  ``--deep``
additionally downloads one 8.5 MB compressed pressure MAT record to confirm
that it is an acquisition time series rather than a disguised PIV field.

This script audits evidence identity.  It does not fit, calibrate, or execute
the FLUXV aerodynamic model.
"""
from __future__ import annotations

import argparse
import binascii
from io import BytesIO
import json
from pathlib import Path
import struct
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import zlib

import numpy as np
from scipy.io import loadmat
import yaml


PLATFORM = Path(__file__).resolve().parent
INVENTORY = (
    PLATFORM
    / "docs"
    / "diag"
    / "high_fidelity_field_candidate_inventory_20260728.yaml"
)
OUTPUT = (
    PLATFORM
    / "docs"
    / "diag"
    / "high_fidelity_field_candidate_audit.json"
)
USER_AGENT = "FLUXV-public-data-audit/1.0"
EOCD_SIGNATURE = b"PK\x05\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"


def _request(
    url: str,
    *,
    method: str = "GET",
    byte_range: tuple[int, int] | None = None,
) -> tuple[bytes, dict[str, str], int]:
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = Request(url, headers=headers, method=method)
    with urlopen(request, timeout=60) as response:
        payload = b"" if method == "HEAD" else response.read()
        return payload, dict(response.headers.items()), response.status


def _json_get(url: str) -> Any:
    payload, _, status = _request(url)
    if status != 200:
        raise RuntimeError(f"JSON request returned HTTP {status}: {url}")
    return json.loads(payload)


def _zip_index(url: str, expected_size: int) -> dict[str, Any]:
    tail_start = max(0, expected_size - 65536)
    tail, headers, status = _request(
        url, byte_range=(tail_start, expected_size - 1)
    )
    if status != 206:
        raise RuntimeError(f"server ignored ZIP tail Range request: {status}")
    eocd_offset = tail.rfind(EOCD_SIGNATURE)
    if eocd_offset < 0:
        raise RuntimeError("ZIP EOCD not found in final 65536 bytes")
    (
        _,
        disk_number,
        central_disk,
        entries_on_disk,
        total_entries,
        central_size,
        central_offset,
        comment_length,
    ) = struct.unpack_from("<4s4H2LH", tail, eocd_offset)
    if disk_number or central_disk or entries_on_disk != total_entries:
        raise RuntimeError("multi-disk ZIP archives are not supported")
    central, _, central_status = _request(
        url,
        byte_range=(
            central_offset,
            central_offset + central_size - 1,
        ),
    )
    if central_status != 206 or len(central) != central_size:
        raise RuntimeError("incomplete ZIP central-directory Range response")

    records: dict[str, dict[str, Any]] = {}
    offset = 0
    while offset < len(central):
        if central[offset : offset + 4] != CENTRAL_SIGNATURE:
            raise RuntimeError(f"invalid central-directory record at {offset}")
        values = struct.unpack_from("<4s6H3L5H2L", central, offset)
        name_length, extra_length, comment_len = values[10:13]
        name = central[
            offset + 46 : offset + 46 + name_length
        ].decode("utf-8", errors="replace")
        records[name] = {
            "compression_method": values[4],
            "crc32": values[7],
            "compressed_size": values[8],
            "uncompressed_size": values[9],
            "local_header_offset": values[-1],
        }
        offset += 46 + name_length + extra_length + comment_len
    if offset != central_size or len(records) != total_entries:
        raise RuntimeError("ZIP central-directory count/size mismatch")
    return {
        "status": status,
        "content_range": headers.get("Content-Range"),
        "size_bytes": expected_size,
        "entry_count": total_entries,
        "central_directory_size_bytes": central_size,
        "central_directory_offset": central_offset,
        "comment_length": comment_length,
        "records": records,
    }


def _zip_member(url: str, record: dict[str, Any]) -> bytes:
    offset = int(record["local_header_offset"])
    compressed_size = int(record["compressed_size"])
    # File names in the audited archives are shorter than this probe.
    header_probe, _, status = _request(
        url, byte_range=(offset, offset + 4095)
    )
    if status != 206 or header_probe[:4] != LOCAL_SIGNATURE:
        raise RuntimeError("invalid local ZIP header probe")
    values = struct.unpack_from("<4s5H3L2H", header_probe, 0)
    name_length, extra_length = values[-2:]
    data_start = offset + 30 + name_length + extra_length
    compressed, _, data_status = _request(
        url,
        byte_range=(data_start, data_start + compressed_size - 1),
    )
    if data_status != 206 or len(compressed) != compressed_size:
        raise RuntimeError("incomplete compressed ZIP member")
    method = int(record["compression_method"])
    if method == 0:
        payload = compressed
    elif method == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"unsupported ZIP method: {method}")
    if len(payload) != int(record["uncompressed_size"]):
        raise RuntimeError("uncompressed ZIP member size mismatch")
    if binascii.crc32(payload) & 0xFFFFFFFF != int(record["crc32"]):
        raise RuntimeError("ZIP member CRC mismatch")
    return payload


def _archive_audit(asset: dict[str, Any]) -> tuple[dict[str, Any], dict]:
    archive = asset["archive"]
    index = _zip_index(archive["url"], int(archive["size_bytes"]))
    names = tuple(index["records"])
    lowered = tuple(name.lower() for name in names)
    field_tokens = ("piv", "velocity", "flowfield", "flow_field")
    field_named = sorted(
        name
        for name, lower in zip(names, lowered)
        if any(token in lower for token in field_tokens)
    )
    result = {
        key: value for key, value in index.items() if key != "records"
    }
    result["field_named_entries"] = field_named
    result["pressure_named_entry_count"] = sum(
        "pressure" in lower or "/cp/" in lower for lower in lowered
    )
    result["force_named_entry_count"] = sum(
        "force" in lower for lower in lowered
    )
    result["entry_suffixes"] = sorted(
        {
            Path(name).suffix.lower()
            for name in names
            if name and not name.endswith("/")
        }
    )
    return result, index["records"]


def _registry_relations(dois: list[str]) -> list[dict[str, Any]]:
    records = []
    for doi in dois:
        crossref = _json_get(
            "https://api.crossref.org/works/" + quote(doi, safe="")
        )["message"]
        query = urlencode(
            {
                "query": f"relatedIdentifiers.relatedIdentifier:{doi}",
                "page[size]": 0,
            }
        )
        datacite = _json_get("https://api.datacite.org/dois?" + query)
        records.append(
            {
                "doi": doi,
                "title": crossref["title"][0],
                "crossref_relation": crossref.get("relation", {}),
                "datacite_related_total": datacite["meta"]["total"],
            }
        )
    return records


def _supplement_payloads() -> dict[str, Any]:
    aip_ids = (32187777, 32187783, 32187789, 32187795)
    aip = []
    for article_id in aip_ids:
        article = _json_get(
            f"https://api.figshare.com/v2/articles/{article_id}"
        )
        aip.extend(
            {
                "name": item["name"],
                "size": item["size"],
                "md5": item["computed_md5"],
            }
            for item in article["files"]
        )
    scielo = _json_get("https://api.figshare.com/v2/articles/10438160")
    scielo_files = [
        {
            "name": item["name"],
            "size": item["size"],
            "md5": item["computed_md5"],
        }
        for item in scielo["files"]
    ]
    return {
        "aip_2026": aip,
        "scielo_2019": scielo_files,
    }


def _deep_pressure_mat(
    url: str, records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    target = next(
        name
        for name in records
        if name.endswith(
            "RawPre_new_meanaoa170_fre2_amp10_150cycles.mat"
        )
    )
    payload = _zip_member(url, records[target])
    mat = loadmat(BytesIO(payload), simplify_cells=True)
    channel_names = np.asarray(mat["ChanNames"])
    measured = mat["ConvertedData"]["Data"]["MeasuredData"]
    sample_lengths = [
        int(np.asarray(item["Data"]).size)
        for item in measured
        if np.asarray(item["Data"]).size
    ]
    return {
        "name": target,
        "variables": sorted(
            key for key in mat if not key.startswith("__")
        ),
        "channel_count": int(channel_names.size),
        "nonempty_measured_channel_count": len(sample_lengths),
        "unique_samples_per_nonempty_channel": sorted(set(sample_lengths)),
        "coordinate_or_velocity_variables": sorted(
            key
            for key in mat
            if key.lower()
            in {"x", "y", "z", "u", "v", "w", "velocity"}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deep",
        action="store_true",
        help="also inspect one compressed 8.5 MB raw pressure MAT",
    )
    args = parser.parse_args()
    inventory = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))
    assets = inventory["assets"]

    reverse_asset = assets["four_tu_reverse_flow_naca643418_v2"]
    reverse, reverse_records = _archive_audit(reverse_asset)
    reverse_readme_name = next(
        name for name in reverse_records if name.endswith("/readme.txt")
    )
    reverse_readme = _zip_member(
        reverse_asset["archive"]["url"],
        reverse_records[reverse_readme_name],
    ).decode("utf-8", errors="replace")
    reverse["readme_primary_content_is_pressure"] = (
        "process_unsteady_pressure.m" in reverse_readme
        and "folder Pressure/unsteady" in reverse_readme
    )
    reverse["readme_states_150_cycles"] = (
        "A total of 150 cycles" in reverse_readme
    )
    reverse["deep_pressure_mat"] = (
        _deep_pressure_mat(
            reverse_asset["archive"]["url"], reverse_records
        )
        if args.deep
        else {"executed": False}
    )

    vg_asset = assets["four_tu_du97w300_vortex_generator_v2"]
    vg, _ = _archive_audit(vg_asset)
    relations = _registry_relations(
        inventory["discovery_snapshot"][
            "high_fidelity_paper_relation_audit"
        ]["dois"]
    )
    supplements = _supplement_payloads()

    reverse_expected = reverse_asset["archive"]
    vg_expected = vg_asset["archive"]
    archive_checks = {
        "reverse_archive_identity": (
            reverse["entry_count"] == reverse_expected["zip_entries"]
            and reverse["central_directory_size_bytes"]
            == reverse_expected["central_directory_size_bytes"]
            and not reverse["field_named_entries"]
            and reverse["readme_primary_content_is_pressure"]
            and reverse["readme_states_150_cycles"]
        ),
        "vg_archive_identity": (
            vg["entry_count"] == vg_expected["zip_entries"]
            and vg["central_directory_size_bytes"]
            == vg_expected["central_directory_size_bytes"]
            and not vg["field_named_entries"]
            and vg["pressure_named_entry_count"] > 0
            and vg["force_named_entry_count"] > 0
        ),
        "high_fidelity_registry_snapshot": all(
            not item["crossref_relation"]
            and item["datacite_related_total"] == 0
            for item in relations
        ),
        "aip_supplements_are_movies": (
            len(supplements["aip_2026"]) == 4
            and all(
                item["name"].lower().endswith(".mp4")
                for item in supplements["aip_2026"]
            )
        ),
        "scielo_payload_is_figures_and_tables": all(
            Path(item["name"]).suffix.lower() in {".jpg", ".xls"}
            for item in supplements["scielo_2019"]
        ),
    }
    if args.deep:
        deep = reverse["deep_pressure_mat"]
        archive_checks["reverse_raw_mat_is_pressure_acquisition"] = (
            deep["variables"]
            == ["ChanNames", "ConvertVer", "ConvertedData"]
            and deep["channel_count"] == 253
            and deep["unique_samples_per_nonempty_channel"] == [24000]
            and not deep["coordinate_or_velocity_variables"]
        )

    result = {
        "version": 1,
        "as_of": inventory["as_of"],
        "evaluated_against": inventory["contract"],
        "deep": args.deep,
        "archives": {
            "four_tu_reverse_flow_naca643418_v2": reverse,
            "four_tu_du97w300_vortex_generator_v2": vg,
        },
        "high_fidelity_paper_relations": relations,
        "supplement_payloads": supplements,
        "checks": archive_checks,
        "production_field_eligible_assets": [],
        "independent_pressure_output_assets": inventory["decision"][
            "independent_pressure_output_assets"
        ],
        "decision": {
            "spatial_state_gate": "NO-GO",
            "independent_pressure_output_gate": "GO",
            "physical_promotion": False,
        },
        "passed": all(archive_checks.values()),
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
