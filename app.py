import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input, State, dash_table, ALL
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import io
import base64
import json
from utils.panelapp_api import (
	fetch_panels,
	fetch_panel_genes,
	PANELAPP_UK_BASE,
	PANELAPP_AU_BASE
)

panels_uk_df = fetch_panels(PANELAPP_UK_BASE)
panels_au_df = fetch_panels(PANELAPP_AU_BASE)

internal_df = pd.read_csv("data/internal_panels.csv")
internal_panels = internal_df[["panel_id", "panel_name"]].drop_duplicates()

def panel_options(df):
	return [{"label": f"{row['name']} (ID {row['id']})", "value": row["id"]} for _, row in df.iterrows()]

def internal_options(df):
	return [{"label": f"{row['panel_name']} (ID {row['panel_id']})", "value": row["panel_id"]} for _, row in df.iterrows()]

# Function to generate a pie chart for a single panel

def generate_panel_pie_chart(panel_df, panel_name):
    # Filter out confidence level 0 (manual genes) before generating the chart
    panel_df = panel_df[panel_df['confidence_level'] != 0]
    
    conf_counts = panel_df.groupby('confidence_level').size().reset_index(name='count')
    conf_counts = conf_counts.sort_values('confidence_level', ascending=False)
    
    # Use only colors for levels 1-3 (remove the blue for level 0)
    colors = ['#d4edda', '#fff3cd', '#f8d7da']  # Green, Yellow, Red for 3,2,1
    
    labels = [f"Confidence {level} ({count})" for level, count in 
              zip(conf_counts['confidence_level'], conf_counts['count'])]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(conf_counts['count'], labels=labels, colors=colors, autopct='%1.1f%%', 
           startangle=90, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    return html.Div([
        html.H4(f"Gene Distribution - {panel_name}", className="text-center mb-3"),
        html.Img(src=f"data:image/png;base64,{data}", 
                style={"maxWidth": "100%", "height": "auto", "display": "block", "margin": "auto"})
    ], style={"border": "1px solid #999", "padding": "15px", "borderRadius": "8px", 
             "maxWidth": "650px", "margin": "0 auto"})

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

app.layout = dbc.Container([
	html.H1("🧬Panel Builder"),
	html.Div(html.Hr(), id="hr-hidden", style={"display": "none"}),
	dbc.Row([
		dbc.Col([html.Label("PanelApp UK"), dcc.Dropdown(id="dropdown-uk", options=panel_options(panels_uk_df), placeholder="Select a UK panel", multi=True)]),
		dbc.Col([html.Label("PanelApp Australia"), dcc.Dropdown(id="dropdown-au", options=panel_options(panels_au_df), placeholder="Select a AUS panel", multi=True)]),
		dbc.Col([html.Label("Internal Panel"), dcc.Dropdown(id="dropdown-internal", options=internal_options(internal_panels), placeholder="Select an internal panel", multi=True)])
	]),
	dbc.Row([
		dbc.Col([html.Label("Add custom gene(s) manually:"), dcc.Textarea(id="manual-genes", placeholder="Enter gene symbols, one per line", style={"width": "100%", "height": "100px"})]),
		dbc.Col([html.Label("Filter Genes by confidence level:"), dcc.Checklist(id="confidence-filter", options=[{"label": " Green (3)", "value": 3}, {"label": " Amber (2)", "value": 2}, {"label": " Red (1)", "value": 1}], value=[3, 2], inline=False)]),
		dbc.Col([
			html.Label("Import code:"),
			dcc.Textarea(id="panel-code-input", placeholder="Paste a previously generated code here...", style={"width": "100%", "height": "60px"}),
			html.Div(dbc.Button("Import Previous Panel", id="import-panel-btn", color="info", className="mt-2"), className="d-flex justify-content-center")
		])
	]),
	html.Hr(),
	dbc.Row(
		dbc.Col(html.Div([
			dbc.Button("Reset", id="reset-btn", color="danger", className="me-2"),
			dbc.Button("Build Panel", id="load-genes-btn", color="primary")
		], className="d-flex justify-content-center"), width=12),
		className="mb-3"
	),
	html.Div(html.Hr(), id="hr-venn", style={"display": "none"}),

	# Loading wrapper
	dcc.Loading(
		children=[
			# Added pie chart container here
			html.Div(id="pie-chart-container", style={"marginBottom": "20px", "display": "none"}),
			html.Div(id="venn-container", style={"marginBottom": "20px"}),
			html.Div(html.Hr(), id="hr-summary", style={"display": "none"}),
			html.Div(id="summary-table-output"),
			html.Div(html.Hr(), id="hr-table", style={"display": "none"}),
			html.Div(id="gene-table-output"),
		],
		type="circle",
		color="#007BFF",
		fullscreen=True
	),

	dcc.Store(id="gene-list-store"),
	html.Hr(),
	html.Div(
		id="generate-code-section",
		style={"display": "none", "width": "100%"},
		children=[
			html.Div(dbc.Button("Generate Unique Code", id="generate-code-btn", color="primary"), style={"textAlign": "center", "marginBottom": "10px"}),
			html.Div(dcc.Textarea(id="generated-code-output", style={"width": "80%", "maxWidth": "900px", "height": "60px", "margin": "0 auto", "display": "block"}, readOnly=True), id="generated-code-container-text")
		]
	),
	html.Hr(),
], fluid=True)

@app.callback(
	Output("generate-code-section", "style"),
	Input("load-genes-btn", "n_clicks"),
	Input("reset-btn", "n_clicks"),
	prevent_initial_call=True
)
def toggle_code_visibility(n_build, n_reset):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "load-genes-btn":
        return {"display": "block"}
    elif triggered_id == "reset-btn":
        return {"display": "none"}
    return dash.no_update

@app.callback(
	Output("generated-code-output", "value", allow_duplicate=True),
	Input("generate-code-btn", "n_clicks"),
	State("dropdown-uk", "value"),
	State("dropdown-au", "value"),
	State("dropdown-internal", "value"),
	State("confidence-filter", "value"),
	State("manual-genes", "value"),
	prevent_initial_call=True
)
def generate_unique_code(n_clicks, uk_ids, au_ids, internal_ids, confs, manual):
	manual_list = [g.strip() for g in manual.strip().splitlines() if g.strip()] if manual else []
	config = {
		"uk": uk_ids or [],
		"au": au_ids or [],
		"internal": internal_ids or [],
		"conf": confs or [],
		"manual": manual_list
	}
	encoded = base64.urlsafe_b64encode(json.dumps(config).encode()).decode()
	return encoded

@app.callback(
	Output("dropdown-uk", "value"),
	Output("dropdown-au", "value"),
	Output("dropdown-internal", "value"),
	Output("confidence-filter", "value"),
	Output("manual-genes", "value"),
	Output("panel-code-input", "value"),
	Output("summary-table-output", "children"),
	Output("gene-table-output", "children"),
	Output("venn-container", "children"),
	Output("pie-chart-container", "children"),  # Added pie chart output
	Output("pie-chart-container", "style"),     # Added pie chart style output
	Output("gene-list-store", "data"),
	Output("generated-code-output", "value"),
	Input("reset-btn", "n_clicks"),
	Input("import-panel-btn", "n_clicks"),
	State("panel-code-input", "value"),
	prevent_initial_call=True
)
def handle_reset_or_import(n_reset, n_import, code):
	ctx = dash.callback_context
	if not ctx.triggered:
		raise dash.exceptions.PreventUpdate

	triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

	if triggered_id == "reset-btn":
		return None, None, None, [3, 2], "", "", "", "", "", "", {"display": "none"}, [], ""

	if triggered_id == "import-panel-btn" and code:
		try:
			decoded = base64.urlsafe_b64decode(code).decode()
			config = json.loads(decoded)
			return (
				config.get("uk", []),
				config.get("au", []),
				config.get("internal", []),
				config.get("conf", []),
				"\n".join(config.get("manual", [])),
				code,
				"", "", "", "", {"display": "none"}, [], ""
			)
		except Exception:
			return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

	raise dash.exceptions.PreventUpdate

# Merged the two duplicate display_panel_genes functions into one
@app.callback(
	Output("summary-table-output", "children", allow_duplicate=True),
	Output("gene-table-output", "children", allow_duplicate=True),
	Output("venn-container", "children", allow_duplicate=True),
	Output("pie-chart-container", "children", allow_duplicate=True),  # Added pie chart output
	Output("pie-chart-container", "style", allow_duplicate=True),     # Added pie chart style output
	Output("gene-list-store", "data", allow_duplicate=True),
	Input("load-genes-btn", "n_clicks"),
	State("dropdown-uk", "value"),
	State("dropdown-au", "value"),
	State("dropdown-internal", "value"),
	State("confidence-filter", "value"),
	State("manual-genes", "value"),
	prevent_initial_call=True
)
def display_panel_genes(n_clicks, selected_uk_ids, selected_au_ids, selected_internal_ids, selected_confidences, manual_genes):
	if not n_clicks:
		return "", "", "", "", {"display": "none"}, []

	genes_combined = []
	gene_sets = {}
	manual_genes_list = []
	panel_dataframes = {}  # Store dataframes for each panel for the pie chart
	panel_names = {}       # Store panel names for identification

	if selected_uk_ids:
		for pid in selected_uk_ids:
			df, panel_info = fetch_panel_genes(PANELAPP_UK_BASE, pid)
			df["confidence_level"] = df["confidence_level"].astype(int)
			
			# Store complete dataframe for pie chart (all confidence levels)
			panel_dataframes[f"UK_{pid}"] = df.copy()
			
			# Filter for combined gene list based on selected confidence levels
			df_filtered = df[df["confidence_level"].isin(selected_confidences)]
			genes_combined.append(df_filtered[["gene_symbol", "confidence_level"]])
			gene_sets[f"UK_{pid}"] = set(df_filtered["gene_symbol"])
			
			panel_name = f"UK Panel {pid}"
			if panel_info and 'name' in panel_info:
				panel_name = panel_info['name']
			panel_names[f"UK_{pid}"] = panel_name

	if selected_au_ids:
		for pid in selected_au_ids:
			df, panel_info = fetch_panel_genes(PANELAPP_AU_BASE, pid)
			df["confidence_level"] = df["confidence_level"].astype(int)
			
			# Store complete dataframe for pie chart (all confidence levels)
			panel_dataframes[f"AUS_{pid}"] = df.copy()
			
			# Filter for combined gene list based on selected confidence levels
			df_filtered = df[df["confidence_level"].isin(selected_confidences)]
			genes_combined.append(df_filtered[["gene_symbol", "confidence_level"]])
			gene_sets[f"AUS_{pid}"] = set(df_filtered["gene_symbol"])
			
			panel_name = f"AUS Panel {pid}"
			if panel_info and 'name' in panel_info:
				panel_name = panel_info['name']
			panel_names[f"AUS_{pid}"] = panel_name

	if selected_internal_ids:
		for pid in selected_internal_ids:
			panel_df = internal_df[internal_df["panel_id"] == pid]
			panel_df["confidence_level"] = panel_df["confidence_level"].astype(int)
			
			# Store complete dataframe for pie chart (all confidence levels)
			panel_dataframes[f"INT-{pid}"] = panel_df.copy()
			
			# Filter for combined gene list based on selected confidence levels
			panel_df_filtered = panel_df[panel_df["confidence_level"].isin(selected_confidences)].copy()
			genes_combined.append(panel_df_filtered[["gene_symbol", "confidence_level"]])
			gene_sets[f"INT-{pid}"] = set(panel_df_filtered["gene_symbol"])
			
			panel_name = next((row['panel_name'] for _, row in internal_panels.iterrows() if row['panel_id'] == pid), f"Internal Panel {pid}")
			panel_names[f"INT-{pid}"] = panel_name

	# Properly handle manual genes
	if manual_genes:
		# Strip and filter empty lines
		manual_genes_list = [g.strip() for g in manual_genes.strip().splitlines() if g.strip()]
		if manual_genes_list:  # Only add if there are actual gene symbols
			manual_df = pd.DataFrame({"gene_symbol": manual_genes_list, "confidence_level": [0] * len(manual_genes_list)})
			genes_combined.append(manual_df)
			gene_sets["Manual"] = set(manual_genes_list)
			panel_dataframes["Manual"] = manual_df
			panel_names["Manual"] = "Manual Gene List"

	if not genes_combined:
		return "No gene found.", "", "", "", {"display": "none"}, []

	df_all = pd.concat(genes_combined)
	df_all["confidence_level"] = df_all["confidence_level"].astype(int)
	df_conf_max = df_all.groupby("gene_symbol", as_index=False)["confidence_level"].max()
	df_merged = df_all.merge(df_conf_max, on=["gene_symbol", "confidence_level"], how="inner")
	df_unique = df_merged.drop_duplicates(subset="gene_symbol", keep="first")
	df_unique = df_unique.sort_values(by=["confidence_level", "gene_symbol"], ascending=[False, True])
	df_unique = df_unique.rename(columns={"gene_symbol": "Gene symbol", "confidence_level": "Confidence level"})

	total_genes = pd.DataFrame({"Number of genes in panel": [df_unique.shape[0]]})
	summary = df_unique.groupby("Confidence level").size().reset_index(name="Number of genes")
	summary_table = dbc.Row([
		dbc.Col(dash_table.DataTable(columns=[{"name": col, "id": col} for col in total_genes.columns], data=total_genes.to_dict("records"), style_cell={"textAlign": "left"}, style_table={"marginBottom": "20px", "width": "100%"}), width=4),
		dbc.Col(dash_table.DataTable(columns=[{"name": col, "id": col} for col in ["Confidence level", "Number of genes"]], data=summary.to_dict("records"), style_cell={"textAlign": "left"}, style_table={"width": "100%"}), width=8)
	])

	# Decide whether to show Venn diagram or pie chart based on number of panels
	show_venn = True
	show_pie = False
	single_panel_id = None
	venn_component = html.Div()
	pie_component = html.Div()

	# Count active panels (excluding Manual)
	active_panels = len([k for k in gene_sets.keys() if k != "Manual"])

	# New improved logic:
	if active_panels == 0 and "Manual" in gene_sets and len(manual_genes_list) > 0:
		# Only manual genes (and they actually exist)
		show_venn = False
		show_pie = True
		single_panel_id = "Manual"
	elif active_panels == 1 and "Manual" not in gene_sets:
		# Only one panel, no manual genes
		show_venn = False
		show_pie = True
		# Get the single non-manual panel
		single_panel_id = next((k for k in gene_sets.keys() if k != "Manual"), None)
	elif active_panels == 1 and "Manual" in gene_sets:
		# One panel + manual genes - show Venn diagram
		show_venn = True
		show_pie = False
	elif active_panels >= 2:
		# Multiple panels - show Venn diagram
		show_venn = True
		show_pie = False

	# Generate Venn diagram if needed
	if show_venn:
		valid_sets = [s for s in gene_sets.values() if len(s) > 0]
		if 2 <= len(valid_sets) <= 3:
			set_items = list(gene_sets.items())[:3]
			labels = [s[0] for s in set_items]
			sets = [s[1] for s in set_items]
			fig, ax = plt.subplots(figsize=(8, 8))
			try:
				if len(sets) == 2:
					venn2(sets, set_labels=labels)#,set_colors = ('#1f77b4', '#2ca02c'))
				elif len(sets) == 3:
					venn3(sets, set_labels=labels)#,set_colors = ('#1f77b4', '#2ca02c', '#9467bd'))
				buf = io.BytesIO()
				plt.tight_layout()
				plt.savefig(buf, format="png")
				plt.close(fig)
				data = base64.b64encode(buf.getbuffer()).decode("ascii")
				venn_component = html.Div([
					html.Img(src=f"data:image/png;base64,{data}", style={"maxWidth": "100%", "height": "auto", "display": "block", "margin": "auto"})
				], style={"border": "1px solid #999", "padding": "5px", "borderRadius": "8px", "maxWidth": "650px", "margin": "0 auto"})
			except Exception as e:
				venn_component = html.Div("Could not generate Venn diagram.", style={"textAlign": "center", "fontStyle": "italic", "color": "#666"})
		else:
			venn_component = html.Div("Venn diagram not available for fewer than 2 or more than 3 groups.", style={"textAlign": "center", "fontStyle": "italic", "color": "#666"})
	
	# Generate pie chart if needed 
	if show_pie and single_panel_id:
		try:
			panel_df = panel_dataframes[single_panel_id]
			panel_name = panel_names[single_panel_id]
			pie_component = generate_panel_pie_chart(panel_df, panel_name)
		except Exception as e:
			pie_component = html.Div(f"Could not generate pie chart: {str(e)}", style={"textAlign": "center", "fontStyle": "italic", "color": "#666"})

	# Settings for displaying pie chart or Venn diagram
	pie_style = {"display": "block", "marginBottom": "20px"} if show_pie else {"display": "none"}

	confidence_levels_present = sorted(df_unique["Confidence level"].unique(), reverse=True)
	buttons = [
		dbc.Button(f"Gene list (confidence {level})", id={"type": "btn-confidence", "level": str(level)}, color="secondary", className="me-2", n_clicks=0)
		for level in confidence_levels_present
	]

	tables_by_level = {
		str(level): dash_table.DataTable(
			columns=[{"name": col, "id": col} for col in ["Gene symbol", "Confidence level"]],
			data=df_unique[df_unique["Confidence level"] == level].to_dict("records"),
			style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"},
			style_cell={"textAlign": "left", "padding": "4px"},
			style_header={"fontWeight": "bold"},
			style_data_conditional=[
				{"if": {"filter_query": "{Confidence level} = 3", "column_id": "Confidence level"}, "backgroundColor": "#d4edda"},
				{"if": {"filter_query": "{Confidence level} = 2", "column_id": "Confidence level"}, "backgroundColor": "#fff3cd"},
				{"if": {"filter_query": "{Confidence level} = 1", "column_id": "Confidence level"}, "backgroundColor": "#f8d7da"},
				{"if": {"filter_query": "{Confidence level} = 0", "column_id": "Confidence level"}, "backgroundColor": "#d1ecf1"},
			],
			page_action="none"
		)
		for level in confidence_levels_present
	}

	if manual_genes_list:
		tables_by_level["Manual"] = dash_table.DataTable(
			columns=[{"name": col, "id": col} for col in ["Gene symbol", "Confidence level"]],
			data=[{"Gene symbol": gene, "Confidence level": 0} for gene in manual_genes_list],
			style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"},
			style_cell={"textAlign": "left", "padding": "4px"},
			style_header={"fontWeight": "bold"},
			style_data_conditional=[{"if": {"filter_query": "{Confidence level} = 0", "column_id": "Confidence level"}, "backgroundColor": "#d1ecf1"}],
			page_action="none"
		)

	table_output = html.Div(id="table-per-confidence")

	summary_layout = html.Div([
		dbc.Row([
			dbc.Col([
				dbc.InputGroup([
					dbc.Input(id="gene-check-input", type="text", placeholder="Search for a gene in custom panel...", className="form-control", debounce=True, n_submit=0),
					dbc.Button("Search", id="gene-check-btn", color="secondary", n_clicks=0)
				])
			], width=6),
			dbc.Col([
				html.Div(id="gene-check-result", className="mt-2", style={"fontStyle": "italic"})
			], width=6)
		]),
		html.Div(summary_table, id="summary-table-content", style={"marginTop": "20px"})
	])

	return summary_layout, html.Div([
		html.Div(buttons, className="mb-3", style={"textAlign": "center"}),
		table_output,
		dcc.Store(id="gene-data-store", data=tables_by_level)
	]), venn_component, pie_component, pie_style, df_unique["Gene symbol"].tolist()

@app.callback(
	Output("table-per-confidence", "children"),
	Input({"type": "btn-confidence", "level": ALL}, "n_clicks"),
	State("gene-data-store", "data")
)
def update_table_by_confidence(btn_clicks, data):
	ctx = dash.callback_context
	if not ctx.triggered:
		return ""
	triggered = ctx.triggered[0]["prop_id"].split(".")[0]
	triggered_dict = json.loads(triggered)
	level = triggered_dict["level"]
	return data.get(level, "")

@app.callback(
	Output("gene-check-result", "children"),
	Output("gene-check-input", "value"),
	Input("gene-check-btn", "n_clicks"),
	Input("gene-check-input", "n_submit"),
	State("gene-check-input", "value"),
	State("gene-list-store", "data")
)
def check_gene_in_panel(n_clicks, n_submit, gene_name, gene_list):
	if not gene_name or not gene_list:
		return "", ""
	if gene_name.upper() in [g.upper() for g in gene_list]:
		return f"✅ Gene '{gene_name}' is present in the custom panel.", ""
	else:
		return f"❌ Gene '{gene_name}' is NOT present in the custom panel.", ""

@app.callback(
	Output("hr-venn", "style"),
	Output("hr-summary", "style"),
	Output("hr-table", "style"),
	Input("load-genes-btn", "n_clicks"),
	Input("reset-btn", "n_clicks"),
	prevent_initial_call=True
)
def toggle_hrs(n_load, n_reset):
	ctx = dash.callback_context
	if not ctx.triggered:
		raise dash.exceptions.PreventUpdate
	triggered = ctx.triggered[0]["prop_id"].split(".")[0]
	if triggered == "load-genes-btn":
		return {"display": "block"}, {"display": "block"}, {"display": "block"}
	elif triggered == "reset-btn":
		return {"display": "none"}, {"display": "none"}, {"display": "none"}
	return dash.no_update, dash.no_update, dash.no_update

if __name__ == "__main__":
	app.run(debug=True)