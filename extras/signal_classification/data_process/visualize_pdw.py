"""Serve an interactive Plotly PDW viewer for recursively discovered CSV files.

Run ``python data_process/visualize_pdw.py --input_dir ./dataset/cut_data``.
The browser groups files by their relative directory, links the three
pulse-order plots on the X axis, and fixes their Y axes. CSV feature mapping is
shared with the training and analysis loaders; Plotly.js is served from
``plotly-2.35.2.min.js`` beside this script for offline use.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

try:
    from data_process.csv_features import (
        FEATURE_COLS,
        normalize_column_name,
        read_feature_csv,
    )
except ModuleNotFoundError:
    from csv_features import FEATURE_COLS, normalize_column_name, read_feature_csv


FEATURE_LABELS = {"CF_MHz": "Carrier frequency (MHz)", "PW_us": "Pulse width (us)", "PRI_us": "PRI (us)"}


def _read_csv(path: Path, no_header_order: str | None = None) -> dict[str, Any]:
    """Read numeric PDW rows, skipping rows with invalid values."""
    table = read_feature_csv(path, no_header_order=no_header_order)
    values = {
        feature: [float(value) for value in table.features[feature].tolist()]
        for feature in FEATURE_COLS
    }
    # Keep optional time/index columns only when no rows were removed during
    # numeric validation, so their positions remain aligned with the features.
    time_values: dict[str, list[float]] = {}
    if table.has_header and len(table.features) == len(table.frame):
        for column in table.frame.columns:
            name = normalize_column_name(column)
            if name not in {"time", "timestamp", "sample", "index", "pulseindex"}:
                continue
            series = table.frame[column].map(pd.to_numeric, errors="coerce")
            if series.notna().all():
                time_values[str(column)] = [float(value) for value in series.tolist()]
    return {
        "features": values,
        "columns": [str(column) for column in table.source_columns],
        "time_columns": list(time_values),
        "time_values": time_values,
        "row_count": len(values["CF_MHz"]),
    }


def _safe_csv_path(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / unquote(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path is outside the configured input directory") from exc
    if candidate.suffix.lower() != ".csv" or not candidate.is_file():
        raise FileNotFoundError(relative_path)
    return candidate


def _safe_directory_path(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / unquote(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path is outside the configured input directory") from exc
    if not candidate.is_dir():
        raise FileNotFoundError(relative_path or str(root))
    return candidate


def _list_directory(root: Path, relative_path: str = "") -> dict[str, Any]:
    """Return one directory level; recursive contents are loaded on demand."""
    directory = _safe_directory_path(root, relative_path)
    directories: list[dict[str, str]] = []
    files: list[dict[str, str]] = []
    with os.scandir(directory) as entries:
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    child_path = f"{relative_path.strip('/')}/{entry.name}".strip("/")
                    directories.append({"name": entry.name, "path": child_path})
                elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".csv"):
                    file_path = f"{relative_path.strip('/')}/{entry.name}".strip("/")
                    files.append({"name": entry.name, "path": file_path})
            except OSError:
                continue
    directories.sort(key=lambda item: item["name"].lower())
    files.sort(key=lambda item: item["name"].lower())
    return {"path": relative_path.strip("/"), "directories": directories, "files": files}


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


HTML_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PDW Plotly Viewer</title><script src="/plotly-2.35.2.min.js"></script>
<style>
:root{color-scheme:light;--ink:#18212b;--muted:#667382;--line:#dce3ea;--panel:#fff;--bg:#f4f7fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:22px 28px 18px;background:var(--panel);border-bottom:1px solid var(--line)}h1{margin:0 0 5px;font-size:23px;font-weight:600}header p{margin:0;color:var(--muted)}
.layout{display:grid;grid-template-columns:minmax(220px,300px) minmax(0,1fr);min-height:calc(100vh - 90px)}aside{padding:18px;background:var(--panel);border-right:1px solid var(--line)}
.root{margin:0 0 14px;padding:10px;border:1px solid var(--line);background:#f8fafc;border-radius:6px;word-break:break-all;color:var(--muted);font-size:12px}.file-list{max-height:calc(100vh - 190px);overflow:auto}.directory{margin:0 0 8px;border:1px solid var(--line);border-radius:5px;background:#fbfcfd}.directory summary{padding:9px 10px;cursor:pointer;font-weight:600;color:#334455}.directory summary::marker{color:#667382}.directory-files{padding:0 5px 5px}.directory .directory{margin:7px 0 0;border-color:#e8edf2}.directory .directory summary{padding:7px 8px;font-weight:500}
.file{width:100%;display:block;padding:8px 7px;text-align:left;border:0;border-radius:5px;background:transparent;color:var(--ink);cursor:pointer}.file:hover,.file.selected{background:#e8f1f9;color:#0d4d7b}.file small{display:block;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
main{min-width:0;padding:20px}.toolbar{display:flex;gap:14px;align-items:end;flex-wrap:wrap;margin-bottom:16px}.toolbar label{display:flex;flex-direction:column;gap:5px;color:var(--muted);font-size:12px}select{min-width:150px;padding:8px 9px;border:1px solid var(--line);border-radius:5px;background:var(--panel);color:var(--ink);font:inherit}
.selected-name{margin:0 0 16px;font-size:17px;font-weight:500;word-break:break-all}.plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.plot{min-width:0;min-height:360px;padding:8px;background:var(--panel);border:1px solid var(--line);border-radius:6px}.plot.wide{grid-column:1/-1}.time-plot{min-height:980px}.status{color:var(--muted);padding:28px 4px}.error{color:#a32626;padding:14px;border:1px solid #f0c9c9;background:#fff5f5;border-radius:5px}
@media(max-width:900px){.layout{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid var(--line)}.file-list{max-height:220px}.plots{grid-template-columns:1fr}.plot.wide{grid-column:auto}}
</style></head><body>
<header><h1>PDW Plotly Viewer</h1><p>Choose a CSV to inspect pulse width, carrier frequency, and PRI distributions.</p></header>
<div class="layout"><aside><div class="root" id="root-path"></div><div id="file-list" class="file-list" aria-live="polite"></div></aside>
<main><div id="status" class="status">Loading CSV files...</div><section id="viewer" hidden><div class="toolbar">
<label>Pulse-order X axis<select id="time-select"></select></label><label>2D X axis<select id="x-select"></select></label><label>2D Y axis<select id="y-select"></select></label></div>
<h2 id="selected-name" class="selected-name"></h2><div class="plots"><div id="plot-time" class="plot wide time-plot"></div><div id="plot-2d" class="plot wide"></div></div></section></main></div>
<script>
const FEATURES={CF_MHz:'Carrier frequency (MHz)',PW_us:'Pulse width (us)',PRI_us:'PRI (us)'},state={data:null,syncing:false};const $=id=>document.getElementById(id);
const plotConfig={responsive:true,displaylogo:false,scrollZoom:true,dragmode:'zoom',modeBarButtonsToRemove:['lasso2d','select2d']};const baseLayout={margin:{l:62,r:18,t:42,b:52},hovermode:'closest',paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{family:'system-ui,sans-serif',color:'#18212b'},xaxis:{showgrid:true,gridcolor:'#e8edf2',fixedrange:false},yaxis:{showgrid:true,gridcolor:'#e8edf2',fixedrange:true}};
function option(label,value){const el=document.createElement('option');el.value=value;el.textContent=label;return el}function setStatus(message,error=false){$('status').hidden=false;$('status').className=error?'error':'status';$('status').textContent=message}
function axisValues(){const selected=$('time-select').value;return selected==='__index__'?state.data.features.CF_MHz.map((_,i)=>i):state.data.time_values[selected]}
function renderPlots(){if(!state.data)return;const x=axisValues(),common={mode:'markers',type:'scatter',x,marker:{size:6,opacity:.72}};
const timeFeatures=[['CF_MHz','#1769aa','x','y',0.69,1],['PW_us','#d95f02','x2','y2',0.35,0.65],['PRI_us','#238b45','x3','y3',0,0.31]];
const timeTraces=timeFeatures.map(([feature,color,xaxis,yaxis])=>({type:'scatter',mode:'markers',x,y:state.data.features[feature],xaxis,yaxis,marker:{size:6,opacity:.72,color},name:FEATURES[feature]}));
const timeLayout={...baseLayout,height:980,hovermode:'x unified',showlegend:false,annotations:timeFeatures.map(([feature,,, ,low,high])=>({xref:'paper',yref:'paper',x:0.5,y:high+0.015,text:FEATURES[feature],showarrow:false,font:{size:15,color:'#18212b'}})),xaxis:{...baseLayout.xaxis,domain:[0,1],anchor:'y',title:''},xaxis2:{...baseLayout.xaxis,domain:[0,1],anchor:'y2',matches:'x',title:''},xaxis3:{...baseLayout.xaxis,domain:[0,1],anchor:'y3',matches:'x',title:$('time-select').selectedOptions[0].text},yaxis:{...baseLayout.yaxis,domain:[0.69,1],anchor:'x',fixedrange:true,title:FEATURES.CF_MHz},yaxis2:{...baseLayout.yaxis,domain:[0.35,0.65],anchor:'x2',fixedrange:true,title:FEATURES.PW_us},yaxis3:{...baseLayout.yaxis,domain:[0,0.31],anchor:'x3',fixedrange:true,title:FEATURES.PRI_us}};
Plotly.react('plot-time',timeTraces,timeLayout,plotConfig);const fx=$('x-select').value,fy=$('y-select').value;
const twoDLayout={...baseLayout,height:520,hovermode:'closest',title:{text:`${FEATURES[fy]} vs ${FEATURES[fx]}`,font:{size:15}},xaxis:{...baseLayout.xaxis,title:FEATURES[fx]},yaxis:{...baseLayout.yaxis,title:FEATURES[fy],fixedrange:false}};
Plotly.react('plot-2d',[{type:'scatter',mode:'markers',x:state.data.features[fx],y:state.data.features[fy],marker:{size:7,opacity:.72,color:'#7b3f98'},text:state.data.features[fx].map((_,i)=>`Pulse ${i}`),hovertemplate:`${FEATURES[fx]}: %{x}<br>${FEATURES[fy]}: %{y}<br>%{text}<extra></extra>`}],twoDLayout,plotConfig)}
async function loadFile(path){document.querySelectorAll('.file').forEach(el=>el.classList.toggle('selected',el.dataset.path===path));setStatus('Loading '+path+'...');try{const response=await fetch('/api/data?path='+encodeURIComponent(path)),payload=await response.json();if(!response.ok)throw new Error(payload.error||'Unable to read CSV');state.data=payload;$('time-select').replaceChildren(option('Pulse index (row order)','__index__'));payload.time_columns.forEach(column=>$('time-select').append(option(column,column)));$('selected-name').textContent=path;$('status').hidden=true;$('viewer').hidden=false;renderPlots()}catch(error){$('viewer').hidden=true;setStatus(error.message,true)}}
function appendFile(container,file){const button=document.createElement('button');button.className='file';button.dataset.path=file.path;button.textContent=file.name;button.addEventListener('click',()=>loadFile(file.path));container.append(button)}
async function loadDirectory(details,path){if(details.dataset.loaded==='true')return;const contents=details.querySelector('.directory-files');contents.replaceChildren();const loading=document.createElement('div');loading.className='status';loading.textContent='Loading folder...';contents.append(loading);try{const response=await fetch('/api/tree?path='+encodeURIComponent(path));const payload=await response.json();if(!response.ok)throw new Error(payload.error||'Unable to read folder');contents.replaceChildren();payload.directories.forEach(directory=>contents.append(makeDirectory(directory)));payload.files.forEach(file=>appendFile(contents,file));details.dataset.loaded='true';if(!payload.directories.length&&!payload.files.length){const empty=document.createElement('div');empty.className='status';empty.textContent='No CSV files in this folder.';contents.append(empty)}}catch(error){contents.replaceChildren();const message=document.createElement('div');message.className='error';message.textContent=error.message;contents.append(message)}}
function makeDirectory(directory){const details=document.createElement('details');details.className='directory';details.dataset.path=directory.path;const summary=document.createElement('summary');summary.textContent=directory.name;details.append(summary);const contents=document.createElement('div');contents.className='directory-files';details.append(contents);details.addEventListener('toggle',()=>{if(details.open)loadDirectory(details,directory.path)});return details}
async function init(){try{const response=await fetch('/api/tree?path=');const payload=await response.json();if(!response.ok)throw new Error(payload.error||'Unable to list CSV folders');$('root-path').textContent=payload.root||$('root-path').textContent;const list=$('file-list');list.replaceChildren();payload.directories.forEach(directory=>list.append(makeDirectory(directory)));payload.files.forEach(file=>appendFile(list,file));if(!payload.directories.length&&!payload.files.length)setStatus('No CSV files found below the configured directory.',true);else setStatus('Expand a folder and choose a CSV file.')}catch(error){setStatus(error.message,true)}}
['time-select','x-select','y-select'].forEach(id=>$(id).addEventListener('change',renderPlots));for(const[key,label]of Object.entries(FEATURES)){$('x-select').append(option(label,key));$('y-select').append(option(label,key))}$('x-select').value='PW_us';$('y-select').value='CF_MHz';$('time-select').append(option('Pulse index (row order)','__index__'));init();
</script></body></html>'''


class ViewerHandler(BaseHTTPRequestHandler):
    root: Path
    no_header_order: str | None = None

    def _send(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(payload))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(payload)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(_json_bytes(payload), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:  # noqa: N802
        request = urlparse(self.path)
        try:
            if request.path in {"/", "/index.html"}: self._send(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif request.path == "/plotly-2.35.2.min.js":
                self._send(Path(__file__).with_name("plotly-2.35.2.min.js").read_bytes(), "application/javascript; charset=utf-8")
            elif request.path in {"/api/files", "/api/tree"}:
                relative = parse_qs(request.query).get("path", [""])[0]
                payload = _list_directory(self.root, relative)
                payload["root"] = str(self.root)
                self._send_json(payload)
            elif request.path == "/api/data":
                relative = parse_qs(request.query).get("path", [""])[0]
                if not relative: raise ValueError("missing path query parameter")
                self._send_json(
                    _read_csv(_safe_csv_path(self.root, relative), self.no_header_order)
                )
            else: self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except FileNotFoundError as exc: self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except (OSError, ValueError, UnicodeError) as exc: self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        if self.path.startswith("/api/"): super().log_message(format, *args)


def create_server(
    input_dir: str | Path,
    host: str,
    port: int,
    no_header_order: str | None = None,
) -> ThreadingHTTPServer:
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir(): raise NotADirectoryError(f"input directory does not exist: {root}")
    handler = type(
        "ConfiguredViewerHandler",
        (ViewerHandler,),
        {"root": root, "no_header_order": no_header_order},
    )
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a Plotly PDW CSV viewer")
    parser.add_argument("--input_dir", "--input-dir", "-i", default=r'G:\competition\2026第五届天智杯\赛前准备\雷达型号识别\test_competition_lengyan-main\test_competition_lengyan-main\datasets\synthetic_radar_overlap', help="Directory to scan recursively for CSV files")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8775, help="HTTP port (default: 8775; use 0 for an auto port)")
    parser.add_argument("--no_browser", "--no-browser", action="store_true", help="Do not open a browser automatically")
    parser.add_argument(
        "--no-header-order",
        default=None,
        help="无表头CSV的物理列顺序，例如 CF_MHz,PW_us,PRI_us；有表头CSV自动按列名识别",
    )
    args = parser.parse_args()
    server = create_server(args.input_dir, args.host, args.port, args.no_header_order)
    address = f"http://{args.host}:{server.server_port}/"
    print(f"Scanning CSV files below: {Path(args.input_dir).expanduser().resolve()}"); print(f"PDW viewer is available at: {address}")
    if not args.no_browser: threading.Timer(0.3, lambda: webbrowser.open(address)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nStopping PDW viewer.")
    finally: server.server_close()


if __name__ == "__main__": main()
