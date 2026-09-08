#!/usr/bin/env python3
"""
Dash dashboard for browsing transcripts, running the speech analysis script,
and viewing results.

Usage:
    source .venv/bin/activate
    export ANTHROPIC_API_KEY=sk-ant-...
    python site/dash_app.py

Then open http://127.0.0.1:8050. The workspace root is shown as a collapsed
folder tree; click a folder name to expand/collapse it and select it as the
upload/new-folder target. Click "Analyze" next to any .txt/.md file to run
scripts/analyze_speech.py — re-analyzing the same file replaces its row.
"""
import base64
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dash import Dash, callback_context, dash_table, dcc, html
from dash.dependencies import ALL, Input, Output, State
from dash.exceptions import PreventUpdate

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = WORKSPACE_ROOT / "analysis.csv"
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "analyze_speech.py"

EXCLUDE_DIRS = {".venv", "site", "scripts", ".git", "__pycache__", "node_modules"}
TRANSCRIPT_SUFFIXES = {".txt", ".md"}

WRAP_COLUMNS = {"priority_1", "priority_2", "priority_3", "overall_impression"}

app = Dash(__name__, title="Speech Analysis Dashboard")
server = app.server


def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CSV_PATH)


def build_file_tree(dir_path: Path, expanded: set, selected: str) -> html.Ul:
    df = load_data()
    analyzed_files = set(df["source_file"]) if not df.empty and "source_file" in df.columns else set()
    entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    items = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        rel = str(entry.relative_to(WORKSPACE_ROOT))
        if entry.is_dir():
            if entry.name in EXCLUDE_DIRS:
                continue
            is_expanded = rel in expanded
            is_selected = rel == selected
            arrow = "\u25BC" if is_expanded else "\u25B6"
            children = build_file_tree(entry, expanded, selected) if is_expanded else None
            items.append(
                html.Li(
                    [
                        html.Span(
                            f"{arrow} \U0001F4C1 {entry.name}/",
                            id={"type": "folder-toggle", "index": rel},
                            n_clicks=0,
                            style={
                                "fontWeight": "bold",
                                "cursor": "pointer",
                                "background": "#dbeafe" if is_selected else "transparent",
                                "padding": "0.1rem 0.3rem",
                                "borderRadius": "4px",
                            },
                        ),
                        children,
                    ]
                )
            )
        elif entry.suffix.lower() in TRANSCRIPT_SUFFIXES:
            is_analyzed = entry.name in analyzed_files
            items.append(
                html.Li(
                    [
                        html.Span(f"\U0001F4C4 {entry.name}", style={"marginRight": "0.75rem"}),
                        html.Button(
                            "Analyzed" if is_analyzed else "Analyze",
                            id={"type": "analyze-btn", "index": rel},
                            n_clicks=0,
                            disabled=is_analyzed,
                            style={
                                "fontSize": "0.75rem",
                                "padding": "0.15rem 0.6rem",
                                "cursor": "not-allowed" if is_analyzed else "pointer",
                                "border": "1px solid #d0d7de",
                                "borderRadius": "4px",
                                "background": "#2563eb" if is_analyzed else "#f6f8fa",
                                "color": "white" if is_analyzed else "black",
                            },
                        ),
                        html.Span(
                            id={"type": "analyze-status", "index": rel},
                            style={"marginLeft": "0.5rem", "fontSize": "0.8rem", "color": "#57606a"},
                        ),
                    ],
                    style={"padding": "0.15rem 0"},
                )
            )
    return html.Ul(items, style={"listStyle": "none", "paddingLeft": "1.25rem"})


MODAL_STYLE_HIDDEN = {"display": "none"}
MODAL_STYLE_VISIBLE = {
    "display": "flex",
    "position": "fixed",
    "top": 0,
    "left": 0,
    "width": "100%",
    "height": "100%",
    "background": "rgba(0,0,0,0.4)",
    "alignItems": "center",
    "justifyContent": "center",
    "zIndex": 1000,
}

app.layout = html.Div(
    style={"fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", "margin": "2rem"},
    children=[
        html.H1("Speech Analysis Dashboard"),
        html.H2("Transcripts", style={"marginTop": "1.5rem"}),
        dcc.Store(id="expanded-folders", data=[]),
        dcc.Store(id="selected-folder", data=""),
        dcc.Store(id="tree-version", data=0),
        dcc.Store(id="api-key-store", storage_type="session", data=""),
        dcc.Store(id="pending-analyze-file", data=""),
        html.Div(
            [
                html.Span(
                    "(workspace root)",
                    id="select-root",
                    n_clicks=0,
                    style={"cursor": "pointer", "fontWeight": "bold", "marginRight": "1rem"},
                ),
                html.Span(id="selected-folder-label", style={"color": "#57606a", "fontSize": "0.85rem"}),
            ],
            style={"marginBottom": "0.5rem"},
        ),
        html.Div(
            [
                dcc.Input(id="new-folder-name", type="text", placeholder="New folder name", style={"marginRight": "0.5rem"}),
                html.Button("Create Folder", id="create-folder-btn", n_clicks=0, style={"cursor": "pointer"}),
                html.Span(id="folder-status", style={"marginLeft": "0.75rem", "fontSize": "0.85rem", "color": "#57606a"}),
            ],
            style={"marginBottom": "0.75rem"},
        ),
        dcc.Upload(
            id="upload-transcript",
            children=html.Div(["Drag and drop or ", html.A("select a .txt file")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "6px",
                "textAlign": "center",
                "marginBottom": "0.75rem",
            },
            disabled=True,
            accept=".txt,.md",
        ),
        html.Div(id="upload-status", style={"marginBottom": "1rem", "fontSize": "0.85rem"}),
        dcc.Loading(type="circle", children=[html.Div(id="file-tree")]),
        html.H2("Results", style={"marginTop": "2rem"}),
        html.P(
            f"Data from {CSV_PATH.name} — click a column header to sort, use the search boxes to filter, "
            "click a cell to view its full text.",
            style={"color": "#57606a"},
        ),
        dcc.Interval(id="refresh-interval", interval=5000),  # re-check the CSV every 5s
        dash_table.DataTable(
            id="results-table",
            editable=True,
            sort_action="native",
            filter_action="native",
            page_size=15,
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "padding": "0.5rem 0.75rem",
                "fontSize": "0.85rem",
                "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                "whiteSpace": "nowrap",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "maxWidth": "300px",
            },
            style_cell_conditional=[
                {"if": {"column_id": col}, "cursor": "pointer"} for col in WRAP_COLUMNS
            ] + [{"if": {"column_id": "title"}, "cursor": "text", "backgroundColor": "#fffbea"}],
            style_header={"backgroundColor": "#f6f8fa", "fontWeight": "bold"},
            style_data={"border": "1px solid #d0d7de"},
        ),
        html.Div(
            id="cell-modal",
            style=MODAL_STYLE_HIDDEN,
            children=[
                html.Div(
                    style={
                        "background": "white",
                        "borderRadius": "8px",
                        "padding": "1.5rem",
                        "maxWidth": "600px",
                        "maxHeight": "80vh",
                        "overflowY": "auto",
                        "boxShadow": "0 4px 20px rgba(0,0,0,0.2)",
                    },
                    children=[
                        html.Button(
                            "Close",
                            id="modal-close",
                            n_clicks=0,
                            style={"float": "right", "cursor": "pointer"},
                        ),
                        html.Div(id="modal-body", style={"whiteSpace": "pre-wrap", "clear": "both"}),
                    ],
                )
            ],
        ),
        html.Div(
            id="api-key-modal",
            style=MODAL_STYLE_HIDDEN,
            children=[
                dcc.Loading(
                    type="circle",
                    children=[
                        html.Div(
                            style={
                                "background": "white",
                                "borderRadius": "8px",
                                "padding": "1.5rem",
                                "maxWidth": "400px",
                                "boxShadow": "0 4px 20px rgba(0,0,0,0.2)",
                            },
                            children=[
                                html.H3("Enter Anthropic API Key", style={"marginTop": 0}),
                                html.P(
                                    "Needed to run the coaching analysis. Stored only in your browser session.",
                                    style={"fontSize": "0.85rem", "color": "#57606a"},
                                ),
                                dcc.Input(
                                    id="api-key-input",
                                    type="password",
                                    placeholder="sk-ant-...",
                                    style={"width": "100%", "marginBottom": "0.75rem", "boxSizing": "border-box"},
                                ),
                                html.P(
                                    "Analysis can take 30\u201360 seconds \u2014 please wait for the spinner to finish.",
                                    style={"fontSize": "0.75rem", "color": "#57606a", "fontStyle": "italic"},
                                ),
                                html.Div(
                                    [
                                        html.Button("Cancel", id="api-key-cancel", n_clicks=0, style={"cursor": "pointer", "marginRight": "0.5rem"}),
                                        html.Button(
                                            "Submit & Analyze",
                                            id="api-key-submit",
                                            n_clicks=0,
                                    style={"cursor": "pointer", "background": "#2563eb", "color": "white", "border": "none", "padding": "0.4rem 0.8rem", "borderRadius": "4px"},
                                ),
                            ]
                        ),
                    ],
                )
            ],
        ),
    ],
),
    ],
)


@app.callback(
    Output("results-table", "data"),
    Output("results-table", "columns"),
    Input("refresh-interval", "n_intervals"),
)
def refresh_data(_):
    df = load_data()
    if df.empty:
        return [], []
    columns = [{"name": col, "id": col, "editable": col == "title"} for col in df.columns]
    return df.to_dict("records"), columns


@app.callback(
    Output("modal-body", "children"),
    Output("cell-modal", "style"),
    Input("results-table", "active_cell"),
    Input("modal-close", "n_clicks"),
    State("results-table", "data"),
    prevent_initial_call=True,
)
def toggle_modal(active_cell, _close_clicks, data):
    triggered_id = callback_context.triggered_id
    if triggered_id == "modal-close":
        return "", MODAL_STYLE_HIDDEN
    if active_cell and active_cell["column_id"] in WRAP_COLUMNS:
        row = data[active_cell["row"]]
        text = row.get(active_cell["column_id"], "")
        return str(text), MODAL_STYLE_VISIBLE
    raise PreventUpdate


@app.callback(
    Output("expanded-folders", "data"),
    Output("selected-folder", "data"),
    Input({"type": "folder-toggle", "index": ALL}, "n_clicks"),
    Input("select-root", "n_clicks"),
    State("expanded-folders", "data"),
    prevent_initial_call=True,
)
def toggle_folder(_folder_clicks, _root_clicks, expanded):
    triggered_id = callback_context.triggered_id
    expanded = set(expanded or [])
    if triggered_id == "select-root":
        return list(expanded), ""
    if not triggered_id:
        raise PreventUpdate
    rel = triggered_id["index"]
    if rel in expanded:
        expanded.discard(rel)
    else:
        expanded.add(rel)
    return list(expanded), rel


@app.callback(
    Output("file-tree", "children"),
    Input("expanded-folders", "data"),
    Input("selected-folder", "data"),
    Input("tree-version", "data"),
)
def rebuild_tree(expanded, selected, _version):
    return build_file_tree(WORKSPACE_ROOT, set(expanded or []), selected or "")


@app.callback(
    Output("selected-folder-label", "children"),
    Output("upload-transcript", "disabled"),
    Input("selected-folder", "data"),
)
def update_selected_folder_label(selected):
    if not selected:
        return "No folder selected — click a folder above to enable upload.", True
    return f"Selected: {selected}/ (uploads and new folders go here)", False


@app.callback(
    Output("upload-status", "children"),
    Output("tree-version", "data", allow_duplicate=True),
    Input("upload-transcript", "contents"),
    State("upload-transcript", "filename"),
    State("selected-folder", "data"),
    State("tree-version", "data"),
    prevent_initial_call=True,
)
def save_upload(contents, filename, selected_folder, version):
    if not contents or not filename or not selected_folder:
        raise PreventUpdate
    target_dir = WORKSPACE_ROOT / selected_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    _header, encoded = contents.split(",", 1)
    data = base64.b64decode(encoded)
    dest = target_dir / filename
    dest.write_bytes(data)
    status = f"Uploaded \u2192 {selected_folder}/{filename}"
    return status, (version or 0) + 1


@app.callback(
    Output("folder-status", "children"),
    Output("tree-version", "data", allow_duplicate=True),
    Input("create-folder-btn", "n_clicks"),
    State("new-folder-name", "value"),
    State("selected-folder", "data"),
    State("tree-version", "data"),
    prevent_initial_call=True,
)
def create_folder(n_clicks, name, selected_folder, version):
    if not n_clicks or not name:
        raise PreventUpdate
    safe_name = name.strip()
    if not safe_name or "/" in safe_name or safe_name.startswith("."):
        return "Invalid folder name.", (version or 0)
    parent = WORKSPACE_ROOT / selected_folder if selected_folder else WORKSPACE_ROOT
    new_dir = parent / safe_name
    if new_dir.exists():
        return f"'{safe_name}' already exists.", (version or 0)
    new_dir.mkdir(parents=True)
    return f"Created {new_dir.relative_to(WORKSPACE_ROOT)}/", (version or 0) + 1


def run_analyze_subprocess(rel_path: str, api_key: str) -> tuple[str, bool]:
    transcript_path = WORKSPACE_ROOT / rel_path
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = api_key
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(transcript_path), "--csv", str(CSV_PATH)],
            capture_output=True,
            text=True,
            cwd=WORKSPACE_ROOT,
            timeout=180,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        return f"Error: {exc}", False

    if result.returncode != 0:
        return f"Failed: {result.stderr.strip()[-300:]}", False
    return "\u2705 Analyzed", True


@app.callback(
    Output({"type": "analyze-status", "index": ALL}, "children"),
    Output("tree-version", "data", allow_duplicate=True),
    Output("api-key-modal", "style"),
    Output("pending-analyze-file", "data"),
    Input({"type": "analyze-btn", "index": ALL}, "n_clicks"),
    State({"type": "analyze-btn", "index": ALL}, "id"),
    State({"type": "analyze-status", "index": ALL}, "children"),
    State("tree-version", "data"),
    State("api-key-store", "data"),
    prevent_initial_call=True,
)
def request_analysis(n_clicks_list, btn_ids, current_statuses, version, api_key):
    triggered = callback_context.triggered_id
    if not triggered:
        raise PreventUpdate

    triggered_index = next((i for i, b in enumerate(btn_ids) if b == triggered), None)
    if triggered_index is None or not n_clicks_list[triggered_index]:
        raise PreventUpdate

    rel_path = triggered["index"]

    if not api_key:
        # No key on hand yet — open the modal and remember which file to analyze.
        return current_statuses, (version or 0), MODAL_STYLE_VISIBLE, rel_path

    outputs = list(current_statuses)
    outputs[triggered_index] = "Analyzing..."
    status_text, success = run_analyze_subprocess(rel_path, api_key)
    outputs[triggered_index] = status_text
    new_version = (version or 0) + 1 if success else (version or 0)
    return outputs, new_version, MODAL_STYLE_HIDDEN, ""


@app.callback(
    Output({"type": "analyze-status", "index": ALL}, "children", allow_duplicate=True),
    Output("tree-version", "data", allow_duplicate=True),
    Output("api-key-modal", "style", allow_duplicate=True),
    Output("api-key-store", "data"),
    Output("pending-analyze-file", "data", allow_duplicate=True),
    Input("api-key-submit", "n_clicks"),
    State("api-key-input", "value"),
    State("pending-analyze-file", "data"),
    State({"type": "analyze-status", "index": ALL}, "children"),
    State({"type": "analyze-btn", "index": ALL}, "id"),
    State("tree-version", "data"),
    prevent_initial_call=True,
)
def submit_api_key(n_clicks, api_key, pending_file, current_statuses, btn_ids, version):
    if not n_clicks or not pending_file:
        raise PreventUpdate
    if not api_key:
        raise PreventUpdate

    outputs = list(current_statuses)
    target_index = next((i for i, b in enumerate(btn_ids) if b["index"] == pending_file), None)
    if target_index is not None:
        outputs[target_index] = "Analyzing..."
    status_text, success = run_analyze_subprocess(pending_file, api_key)
    if target_index is not None:
        outputs[target_index] = status_text
    new_version = (version or 0) + 1 if success else (version or 0)
    return outputs, new_version, MODAL_STYLE_HIDDEN, api_key, ""


@app.callback(
    Output("api-key-modal", "style", allow_duplicate=True),
    Output("pending-analyze-file", "data", allow_duplicate=True),
    Input("api-key-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_api_key_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return MODAL_STYLE_HIDDEN, ""


@app.callback(
    Output("upload-status", "children", allow_duplicate=True),
    Input("results-table", "data"),
    State("results-table", "data_previous"),
    prevent_initial_call=True,
)
def save_title_edits(data, data_previous):
    if not data_previous or not data:
        raise PreventUpdate
    changed = any(
        new_row.get("title") != old_row.get("title")
        for new_row, old_row in zip(data, data_previous)
    )
    if not changed:
        raise PreventUpdate
    pd.DataFrame(data).to_csv(CSV_PATH, index=False)
    return "Title updated \u2705"


if __name__ == "__main__":
    app.run(debug=True, port=8051)
