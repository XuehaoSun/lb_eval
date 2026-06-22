#!/usr/bin/env python3
"""Reusable failure analysis tool for lb_eval pipeline.

Correlates status/ entries with results/ directories, extracts error information
from logs, categorizes failures, and generates a markdown report.

Usage:
    python analyze_failures.py [--repo-dir /path/to/lb_eval] [--output report.md]
    python analyze_failures.py --json  # Output JSON instead of markdown

Examples:
    # Analyze from lb_eval repo root
    cd /root/new_commit/lb_eval && python auto_quant/tools/analyze_failures.py

    # Specify paths explicitly
    python analyze_failures.py --repo-dir /root/new_commit/lb_eval --output /tmp/report.md
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_status_entries(status_dir: Path) -> list[dict]:
    """Load all status JSON files."""
    entries = []
    for root, _, files in os.walk(status_dir):
        for f in files:
            if not f.endswith('.json'):
                continue
            p = Path(root) / f
            try:
                d = json.loads(p.read_text())
                entries.append({
                    'path': str(p),
                    'filename': f,
                    'org': p.parent.name,
                    'status': d.get('status', ''),
                    'model': d.get('model', ''),
                    'scheme': d.get('quant_type', d.get('scheme', '')),
                    'method': d.get('method', 'RTN'),
                    'submitted_time': d.get('submitted_time', ''),
                    'data': d,
                })
            except Exception:
                pass
    return entries


def load_results_map(results_dir: Path) -> dict[tuple, list[dict]]:
    """Load results directories indexed by (org_lower, model_short_lower)."""
    results_map = {}
    if not results_dir.exists():
        return results_map

    for org_dir in results_dir.iterdir():
        if not org_dir.is_dir():
            continue
        for artifact_dir in org_dir.iterdir():
            if not artifact_dir.is_dir():
                continue
            result_jsons = sorted(artifact_dir.glob('results_*.json'), reverse=True)
            for rj in result_jsons:
                try:
                    rd = json.loads(rj.read_text())
                    model_id = rd.get('model_id', '')
                    model_short = model_id.split('/')[-1] if '/' in model_id else model_id
                    key = (org_dir.name.lower(), model_short.lower())
                    results_map.setdefault(key, []).append({
                        'artifact_dir': str(artifact_dir),
                        'artifact_name': artifact_dir.name,
                        'result_json': str(rj),
                        'status': rd.get('status', ''),
                        'model_id': model_id,
                        'generated_at': rd.get('generated_at', ''),
                    })
                except Exception:
                    pass
    return results_map


# ═══════════════════════════════════════════════════════════════════════════════
# Matching Logic
# ═══════════════════════════════════════════════════════════════════════════════

def extract_scheme_from_filename(filename: str) -> str:
    """Extract quantization scheme from status filename."""
    m = re.search(r'_(W4A16|MXFP4|NVFP4|W8A16|MXFP8)', filename)
    return m.group(1) if m else ''


def match_status_to_results(
    failed_entries: list[dict],
    results_map: dict[tuple, list[dict]],
) -> tuple[list[dict], list[dict]]:
    """Match failed status entries to their results directories.

    Matching criteria:
    1. (org, model_short) must match exactly (case-insensitive)
    2. Scheme from filename must appear in artifact directory name
    3. Method (RTN vs TUNING) must match the -RTN/-Tuning suffix
    """
    matched = []
    unmatched = []

    for fe in failed_entries:
        model_full = fe['model']
        model_short = model_full.split('/')[-1] if '/' in model_full else model_full
        org = model_full.split('/')[0] if '/' in model_full else fe['org']
        scheme = extract_scheme_from_filename(fe['filename'])
        method = fe['method'] or 'RTN'

        key = (org.lower(), model_short.lower())
        candidates = results_map.get(key, [])

        best_match = None
        for c in candidates:
            aname = c['artifact_name']
            if scheme and scheme not in aname:
                continue
            if method == 'TUNING' and 'Tuning' not in aname:
                continue
            if method == 'RTN' and 'Tuning' in aname:
                continue
            best_match = c
            break

        if best_match:
            matched.append({'status_entry': fe, 'result': best_match, 'scheme': scheme})
        else:
            unmatched.append({'status_entry': fe, 'scheme': scheme})

    return matched, unmatched


# ═══════════════════════════════════════════════════════════════════════════════
# Error Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_errors_from_run(artifact_dir: str) -> dict:
    """Extract error details from the most recent run directory."""
    ad = Path(artifact_dir)
    run_dirs = sorted(ad.glob('run_*'), reverse=True)
    if not run_dirs:
        return {'phase': None, 'error': 'No run directory found', 'category': 'No Logs'}

    latest_run = run_dirs[0]
    errors = []
    phase_failed = None
    killed = False

    # 1. auto.log (most informative)
    auto_log = latest_run / 'logs' / 'auto.log'
    if auto_log.exists():
        log_text = auto_log.read_text()[-8000:]
        if 'Killed' in log_text:
            killed = True
        err_lines = re.findall(
            r'.*(?:\[ERROR\]|Traceback|OSError|RuntimeError|CUDA|Killed).*',
            log_text,
        )
        if err_lines:
            errors.append(('auto.log', '\n'.join(err_lines[-5:])[:500]))
        phase_m = re.search(r'Pipeline failed at:\s*(\w+)', log_text)
        if phase_m:
            phase_failed = phase_m.group(1)

    # 2. quantize.log
    quant_log = latest_run / 'logs' / 'quantize.log'
    if quant_log.exists():
        log_text = quant_log.read_text()[-5000:]
        if 'Killed' in log_text:
            killed = True
        err_lines = re.findall(
            r'.*(?:\[ERROR\]|Traceback|OSError|RuntimeError|KeyError|ValueError|TypeError|ImportError).*',
            log_text,
        )
        if err_lines:
            errors.append(('quantize.log', '\n'.join(err_lines[-5:])[:500]))

    # 3. setup_env.log
    setup_log = latest_run / 'logs' / 'setup_env.log'
    if setup_log.exists() and not errors:
        log_text = setup_log.read_text()[-3000:]
        err_lines = re.findall(
            r'.*(?:ERROR|failed|ModuleNotFoundError|ImportError).*',
            log_text,
            re.IGNORECASE,
        )
        if err_lines:
            errors.append(('setup_env.log', '\n'.join(err_lines[-3:])[:300]))

    # 4. Agent fix retry logs (last attempt)
    agent_dir = latest_run / 'logs' / 'agent_fixes'
    if agent_dir.exists():
        for phase_dir in sorted(agent_dir.iterdir()):
            if not phase_dir.is_dir():
                continue
            retry_logs = sorted(phase_dir.glob('retry_*.log'))
            if retry_logs:
                last_retry = retry_logs[-1].read_text()[-2000:]
                err_lines = re.findall(
                    r'.*(?:\[ERROR\]|Traceback|OSError|RuntimeError|Killed).*',
                    last_retry,
                )
                if err_lines:
                    errors.append((f'retry({phase_dir.name})', '\n'.join(err_lines[-3:])[:300]))

    # 5. run_report.md (fallback for phase info)
    report = latest_run / 'run_report.md'
    if report.exists():
        text = report.read_text()
        phase_m = re.search(r'Failed Phase.*?:\s*`?(\w+)`?', text)
        if phase_m and not phase_failed:
            phase_failed = phase_m.group(1)

    error_text = errors[0][1] if errors else 'Unknown error'
    category = categorize_error(errors, killed)

    return {
        'phase': phase_failed,
        'error': error_text[:500],
        'category': category,
        'killed': killed,
        'run_dir': str(latest_run),
    }


def categorize_error(errors: list[tuple[str, str]], killed: bool) -> str:
    """Categorize the error based on combined error text patterns."""
    combined = ' '.join(e[1] for e in errors).lower()

    if killed or 'killed' in combined or 'exit=137' in combined or 'signal 9' in combined:
        return 'OOM Killed'
    if 'gguf' in combined or ('safetensors' in combined and 'pytorch_model' in combined):
        return 'GGUF/No Weights'
    if 'cuda out of memory' in combined or 'outofmemoryerror' in combined:
        return 'GPU OOM'
    if 'meta device' in combined or 'more gpus' in combined:
        return 'Needs More GPUs'
    if 'image processor' in combined or 'processor should not be none' in combined:
        return 'Multimodal Model (Unsupported)'
    if 'expected' in combined and ('column' in combined or 'line 1' in combined):
        return 'Corrupt Config JSON'
    if 'import' in combined or 'module' in combined or 'package' in combined:
        return 'Missing Dependencies'
    if 'tokenizer' in combined and ('class' in combined or 'not found' in combined):
        return 'Tokenizer Error'
    if 'attn_mask' in combined or ('dtype' in combined and 'mismatch' in combined):
        return 'Dtype/Attention Mismatch'
    if 'unrecognized configuration' in combined:
        return 'Unknown Architecture'
    if 'not a local folder' in combined or ('not' in combined and 'valid' in combined and 'repo' in combined):
        return 'Model 404 / Deleted'
    if '404' in combined and ('repository' in combined or 'not found' in combined):
        return 'Model 404 / Deleted'
    if 'ledger' in combined or ('git' in combined and 'fatal' in combined):
        return 'Infrastructure Error'
    if 'list index' in combined or 'key error' in combined:
        return 'AutoRound Bug'
    if 'drift' in combined or 'same error repeat' in combined:
        return 'Unresolvable (Drift)'
    if 'connection' in combined or 'network' in combined or 'timeout' in combined:
        return 'Network Error'
    return 'Other'


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_markdown_report(results: list[dict]) -> str:
    """Generate a markdown failure analysis report."""
    lines = []
    lines.append("# lb_eval Failure Analysis Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Failed/Partial entries:** {len(results)}\n")

    # Category summary
    categories = {}
    for r in results:
        categories.setdefault(r['category'], []).append(r)

    lines.append("## Summary by Error Category\n")
    lines.append("| Category | Count | % |")
    lines.append("|----------|-------|---|")
    for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        pct = len(items) * 100 / max(len(results), 1)
        lines.append(f"| {cat} | {len(items)} | {pct:.0f}% |")

    # Phase distribution
    phases = {}
    for r in results:
        p = r.get('phase_failed') or 'unknown/pre-upload'
        phases[p] = phases.get(p, 0) + 1

    lines.append("\n## Failed Phase Distribution\n")
    lines.append("| Phase | Count |")
    lines.append("|-------|-------|")
    for p, c in sorted(phases.items(), key=lambda x: -x[1]):
        lines.append(f"| {p} | {c} |")

    # Detailed per-category
    lines.append("\n---\n## Detailed Failures\n")
    for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        lines.append(f"### {cat} ({len(items)} cases)\n")
        for item in items:
            model = item['model']
            scheme = item['scheme']
            method = item['method']
            phase = item.get('phase_failed') or '?'
            lines.append(f"**`{model}`** — {scheme}/{method} — phase: `{phase}`\n")
            lines.append(f"- Status: `{item['status_file']}`")
            if item.get('artifact_dir'):
                lines.append(f"- Results: `{item['artifact_dir']}`")
            err = item.get('error', '').strip()[:400].replace('\n', '\n  ')
            lines.append(f"- Error:\n  ```\n  {err}\n  ```\n")
        lines.append("")

    # Methodology
    lines.append("---\n## Correlation Methodology\n")
    lines.append("""This script matches `status/` entries to `results/` directories via:

1. **Model ID:** `status.model` → parse org + model_short → find `results/{org}/{artifact}`
2. **Scheme:** Extracted from status filename (W4A16, MXFP4, etc.) → matched in artifact dir name
3. **Method:** `status.method` (RTN/TUNING) → matched against `-RTN`/`-Tuning` dir suffix
4. **Error extraction:** Searches `auto.log` → `quantize.log` → `setup_env.log` → agent retry logs
5. **Categorization:** Pattern-matching on combined error text (OOM, 404, import, dtype, etc.)

Unmatched entries = pipeline crashed before upload phase could execute.
""")
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze lb_eval pipeline failures")
    parser.add_argument('--repo-dir', default='.', help='Path to lb_eval repo root')
    parser.add_argument('--output', default='', help='Output file path (default: stdout)')
    parser.add_argument('--json', action='store_true', help='Output as JSON instead of markdown')
    parser.add_argument('--status-filter', default='Failed,Partial,Eval Failed',
                        help='Comma-separated status values to include')
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    status_dir = repo_dir / 'status'
    results_dir = repo_dir / 'results'

    if not status_dir.exists():
        print(f"ERROR: status directory not found: {status_dir}", file=sys.stderr)
        return 1

    # Load data
    all_entries = load_status_entries(status_dir)
    status_filters = [s.strip() for s in args.status_filter.split(',')]
    failed_entries = [e for e in all_entries if any(sf in e['status'] for sf in status_filters)]
    results_map = load_results_map(results_dir)

    print(f"[info] Total status entries: {len(all_entries)}", file=sys.stderr)
    print(f"[info] Failed/filtered: {len(failed_entries)}", file=sys.stderr)
    print(f"[info] Results map keys: {len(results_map)}", file=sys.stderr)

    # Match
    matched, unmatched = match_status_to_results(failed_entries, results_map)
    print(f"[info] Matched: {len(matched)}, Unmatched: {len(unmatched)}", file=sys.stderr)

    # Analyze
    results = []
    for m in matched:
        info = extract_errors_from_run(m['result']['artifact_dir'])
        se = m['status_entry']
        results.append({
            'model': se['model'],
            'scheme': m['scheme'],
            'method': se['method'] or 'RTN',
            'status_file': se['path'],
            'artifact_dir': m['result']['artifact_dir'],
            'run_dir': info.get('run_dir', ''),
            'phase_failed': info['phase'],
            'error': info['error'],
            'category': info['category'],
            'killed': info.get('killed', False),
        })

    for u in unmatched:
        se = u['status_entry']
        results.append({
            'model': se['model'],
            'scheme': u['scheme'],
            'method': se['method'] or 'RTN',
            'status_file': se['path'],
            'artifact_dir': None,
            'run_dir': None,
            'phase_failed': None,
            'error': 'No results uploaded (pipeline crashed before upload)',
            'category': 'No Results Uploaded',
            'killed': False,
        })

    # Output
    if args.json:
        output = json.dumps(results, indent=2, default=str)
    else:
        output = generate_markdown_report(results)

    if args.output:
        Path(args.output).write_text(output)
        print(f"[info] Report written to: {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
