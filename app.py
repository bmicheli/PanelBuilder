# =============================================================================
# IMPORTS
# =============================================================================

import dash
import os
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
import requests
import re
import time
from datetime import datetime
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import numpy as np
from utils.panelapp_api import (
	fetch_panels,
	fetch_panel_genes,
	PANELAPP_UK_BASE,
	PANELAPP_AU_BASE
)

# =============================================================================
# PANEL PRESETS CONFIGURATION
# =============================================================================

PANEL_PRESETS = {
	"epilepsy": {
		"name": "Epilepsy",
		"icon": "mdi:brain",
		"uk_panels": [402], 
		"au_panels": [202],
		"internal": [],
		"conf": [3,2],
		"manual": [],
		"hpo_terms": ["HP:0200134","HP:0002353","HP:0001250"] 
	},
	"cardiac": {
		"name": "Cardiac Conditions",
		"icon": "mdi:heart",
		"uk_panels": [749],
		"au_panels": [253],
		"internal": [],
		"conf": [3,2],
		"manual": [],
		"hpo_terms": ["HP:0001638","HP:0001637"]  
	},
	"cancer_predisposition": {
		"name": "Colorectal Cancer Predisposition",
		"icon": "mdi:dna",
		"uk_panels": [244],
		"au_panels": [4371],
		"internal": [3,2],
		"conf": [],
		"manual": [],
		"hpo_terms": []
	},
	"neurodevelopmental": {
		"name": "Neurodevelopmental Disorders",
		"icon": "mdi:head-cog",
		"uk_panels": [285],
		"au_panels": [250],
		"internal": [],
		"conf": [3],
		"manual": [],
		"hpo_terms": ["HP:0012758", "HP:0001249"] 
	}
}

# =============================================================================
# UTILITY FUNCTIONS - HPO MANAGEMENT
# =============================================================================

def fetch_panel_disorders(base_url, panel_id):
	try:
		base_url = base_url.rstrip('/')
		url = f"{base_url}/panels/{panel_id}/"
		response = requests.get(url, timeout=10)
		response.raise_for_status()
		data = response.json()
		
		relevant_disorders = data.get('relevant_disorders', [])
		
		if not relevant_disorders:
			return []
		
		hpo_terms = []
		
		for disorder in relevant_disorders:
			if isinstance(disorder, str):
				hpo_matches = re.findall(r'HP:\d{7}', disorder)
				hpo_terms.extend(hpo_matches)
		
		hpo_terms = list(dict.fromkeys(hpo_terms))
		
		return hpo_terms
		
	except requests.exceptions.HTTPError as e:
		if e.response.status_code == 404:
			print(f"Panel {panel_id} non trouvé (404) - URL: {url}")
		else:
			print(f"Erreur HTTP {e.response.status_code} pour le panel {panel_id}: {e}")
		return []
	except Exception as e:
		print(f"Erreur lors de la récupération des disorders pour le panel {panel_id}: {e}")
		return []

def get_hpo_terms_from_panels(uk_ids=None, au_ids=None):
	all_hpo_terms = set() 
	if au_ids:
		for panel_id in au_ids:
			hpo_terms = fetch_panel_disorders(PANELAPP_AU_BASE, panel_id)
			all_hpo_terms.update(hpo_terms)
	
	return list(all_hpo_terms)

def search_hpo_terms(query, limit=100):
	if not query or len(query.strip()) < 2:
		return []
	
	try:
		url = f"https://ontology.jax.org/api/hp/search?q={query}&page=0&limit={limit}"
		response = requests.get(url, timeout=5)
		response.raise_for_status()
		data = response.json()
		
		options = []
		if 'terms' in data:
			for term in data['terms']:
				label = f"{term.get('name', '')} ({term.get('id', '')})"
				value = term.get('id', '')
				options.append({"label": label, "value": value})
		
		return options
	except Exception as e:
		print(f"Error searching HPO terms: {e}")
		return []

def fetch_hpo_term_details(term_id):
	try:
		url = f"https://ontology.jax.org/api/hp/terms/{term_id}"
		response = requests.get(url, timeout=5)
		if response.status_code == 200:
			term_data = response.json()
			return {
				"id": term_id,
				"name": term_data.get("name", term_id),
				"definition": term_data.get("definition", "No definition available")
			}
		else:
			return {
				"id": term_id,
				"name": term_id,
				"definition": "Unable to fetch definition"
			}
	except Exception as e:
		return {
			"id": term_id,
			"name": term_id,
			"definition": "Unable to fetch definition"
		}

# =============================================================================
# UTILITY FUNCTIONS - PANEL OPTIONS
# =============================================================================

def panel_options(df):
	options = []
	for _, row in df.iterrows():
		version_text = f" v{row['version']}" if 'version' in row and pd.notna(row['version']) else ""
		label = f"{row['name']}{version_text} (ID {row['id']})"
		options.append({"label": label, "value": row["id"]})
	return options

def internal_options(df):
	return [{"label": f"{row['panel_name']} (ID {row['panel_id']})", "value": row["panel_id"]} for _, row in df.iterrows()]

# =============================================================================
# UTILITY FUNCTIONS - PANEL SUMMARY GENERATION
# =============================================================================

def generate_panel_summary(uk_ids, au_ids, internal_ids, confs, manual_genes_list, panels_uk_df, panels_au_df, internal_panels):
	"""Generate a formatted summary of panels and genes"""
	summary_parts = []
	
	# Helper function to get confidence notation
	def get_confidence_notation(conf_list):
		if not conf_list:
			return ""
		conf_set = set(conf_list)
		if conf_set == {3}:
			return "_G"
		elif conf_set == {2}:
			return "_O"  
		elif conf_set == {1}:
			return "_R"
		elif conf_set == {3, 2}:
			return "_GO"
		elif conf_set == {3, 1}:
			return "_GR"
		elif conf_set == {2, 1}:
			return "_OR"
		elif conf_set == {3, 2, 1}:
			return "_GOR"
		else:
			return ""
	
	confidence_suffix = get_confidence_notation(confs)
	
	# Process UK panels
	if uk_ids:
		for panel_id in uk_ids:
			panel_row = panels_uk_df[panels_uk_df['id'] == panel_id]
			if not panel_row.empty:
				panel_info = panel_row.iloc[0]
				panel_name = panel_info['name'].replace(' ', '_').replace('/', '_').replace(',', '_')
				version = f"_v{panel_info['version']}" if pd.notna(panel_info.get('version')) else ""
				summary_parts.append(f"PanelApp_UK/{panel_name}{version}{confidence_suffix}")
	
	# Process AU panels
	if au_ids:
		for panel_id in au_ids:
			panel_row = panels_au_df[panels_au_df['id'] == panel_id]
			if not panel_row.empty:
				panel_info = panel_row.iloc[0]
				panel_name = panel_info['name'].replace(' ', '_').replace('/', '_').replace(',', '_')
				version = f"_v{panel_info['version']}" if pd.notna(panel_info.get('version')) else ""
				summary_parts.append(f"PanelApp_AUS/{panel_name}{version}{confidence_suffix}")
	
	# Process Internal panels
	if internal_ids:
		for panel_id in internal_ids:
			panel_row = internal_panels[internal_panels['panel_id'] == panel_id]
			if not panel_row.empty:
				panel_info = panel_row.iloc[0]
				panel_name = panel_info['panel_name'].replace(' ', '_').replace('/', '_').replace(',', '_')
				summary_parts.append(f"Internal/{panel_name}{confidence_suffix}")
	
	# Add manual genes
	if manual_genes_list:
		summary_parts.extend(manual_genes_list)
	
	return ",".join(summary_parts)

# =============================================================================
# UTILITY FUNCTIONS - CHART GENERATION
# =============================================================================

def generate_panel_pie_chart(panel_df, panel_name, version=None):
	panel_df = panel_df[panel_df['confidence_level'] != 0]
	
	conf_counts = panel_df.groupby('confidence_level').size().reset_index(name='count')
	conf_counts = conf_counts.sort_values('confidence_level', ascending=False)
	
	colors = ['#d4edda', '#fff3cd', '#f8d7da']  # Green, Yellow, Red for 3,2,1
	
	labels = [f"{count} genes" for level, count in 
			zip(conf_counts['confidence_level'], conf_counts['count'])]
	
	fig, ax = plt.subplots(figsize=(9, 5))  
	ax.pie(conf_counts['count'], labels=labels, colors=colors, autopct='%1.1f%%', 
		startangle=90, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
	ax.axis('equal') 

	title = f"Gene Distribution - {panel_name}"
	if version:
		title += f" (v{version})"
	
	# Convert plot to base64 image
	buf = io.BytesIO()
	plt.tight_layout()
	plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
	plt.close(fig)
	data = base64.b64encode(buf.getbuffer()).decode("ascii")
	
	return html.Div([
		html.H4(title, className="text-center mb-3", style={"fontSize": "16px"}),
		html.Img(src=f"data:image/png;base64,{data}", 
				style={"maxWidth": "100%", "height": "auto", "display": "block", "margin": "auto"})
	], style={
		"border": "1px solid #999", 
		"padding": "10px", 
		"borderRadius": "8px", 
		"maxWidth": "100%", 
		"margin": "0",
		"height": "580px",  
		"display": "flex",
		"flexDirection": "column",
		"justifyContent": "center"
	})

def create_upset_plot(gene_sets, panel_names):
    """Create an UpSet plot for visualizing intersections of multiple sets"""
    from itertools import combinations, chain
    
    # Prepare data for UpSet plot
    all_genes = set()
    for genes in gene_sets.values():
        all_genes.update(genes)
    
    if not all_genes:
        return None
    
    # For each gene, find which sets it belongs to
    gene_memberships = {}
    sets_list = list(gene_sets.keys())
    
    for gene in all_genes:
        membership = tuple(i for i, (name, genes) in enumerate(gene_sets.items()) if gene in genes)
        if membership not in gene_memberships:
            gene_memberships[membership] = []
        gene_memberships[membership].append(gene)
    
    # Sort intersections: single sets first (panels), then multi-set intersections
    single_sets = []  # Individual panels
    multi_sets = []   # Intersections between panels
    
    for membership, genes in gene_memberships.items():
        if len(membership) == 1:
            # Single panel
            single_sets.append((membership, genes))
        else:
            # Multi-panel intersection
            multi_sets.append((membership, genes))
    
    # Sort single sets by size (largest first)
    single_sets.sort(key=lambda x: len(x[1]), reverse=True)
    
    # Sort multi-set intersections by size (largest first)
    multi_sets.sort(key=lambda x: len(x[1]), reverse=True)
    
    # Combine: panels first, then intersections (limit to max 10 total)
    sorted_intersections = single_sets + multi_sets
    max_intersections = min(15, len(sorted_intersections))
    sorted_intersections = sorted_intersections[:max_intersections]
    
    # Create figure with higher DPI for crispness
    fig, (ax_bars, ax_matrix) = plt.subplots(2, 1, figsize=(10, 5), dpi=120,
                                           gridspec_kw={'height_ratios': [1, 1]})
    
    # Top plot: intersection sizes (VERTICAL bars)
    intersection_sizes = [len(genes) for _, genes in sorted_intersections]
    x_pos = np.arange(len(intersection_sizes))  # Use arange for perfect positioning
    
    # Create vertical bars with different colors for panels vs intersections
    bar_colors = []
    for membership, _ in sorted_intersections:
        if len(membership) == 1:
            bar_colors.append('#3498db')  # Blue for individual panels
        else:
            bar_colors.append('#2c3e50')  # Dark for intersections
    
    bars = ax_bars.bar(x_pos, intersection_sizes, color=bar_colors, alpha=0.8, width=0.6,
                       edgecolor='white', linewidth=0.5)
    ax_bars.set_ylabel('Number of Genes', fontsize=11, fontweight='bold')
    ax_bars.set_title('Gene Panel Intersections', fontsize=13, fontweight='bold', pad=20)
    ax_bars.set_xticks([])
    ax_bars.grid(True, alpha=0.3, axis='y')
    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['right'].set_visible(False)
    
    # Set x-axis limits to match matrix exactly
    ax_bars.set_xlim(-0.5, len(sorted_intersections) - 0.5)
    
    # Add value labels on top of bars
    for i, (bar, size) in enumerate(zip(bars, intersection_sizes)):
        ax_bars.text(i, bar.get_height() + max(intersection_sizes) * 0.01, 
                    str(size), ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Bottom plot: binary matrix with precise alignment
    matrix_data = np.zeros((len(sets_list), len(sorted_intersections)))
    for j, (membership, _) in enumerate(sorted_intersections):
        for i in membership:
            matrix_data[i, j] = 1
    
    # Clear the matrix plot
    ax_matrix.clear()
    
    # Set up the matrix plot with EXACT alignment to bars
    ax_matrix.set_xlim(-0.5, len(sorted_intersections) - 0.5)
    ax_matrix.set_ylim(-0.5, len(sets_list) - 0.5)
    
    # Draw the matrix with perfect circles aligned with bars
    circle_radius = 0.1 # Precise circles
    for i in range(len(sets_list)):
        for j in range(len(sorted_intersections)):
            # Use EXACT same x positioning as bars: j (which matches x_pos[j])
            x_center = float(j)  # Ensure float for perfect alignment
            y_center = float(i)
            
            if matrix_data[i, j] == 1:
                # Draw filled circle (black) - perfectly round
                circle = plt.Circle((x_center, y_center), circle_radius, 
                                  color='black', zorder=2, clip_on=False)
                ax_matrix.add_patch(circle)
            else:
                # Draw empty circle (light gray) - smaller and subtle
                circle = plt.Circle((x_center, y_center), circle_radius, 
                                  fill=False, color='lightgray', 
                                  linewidth=1, alpha=0.4, zorder=2, clip_on=False)
                ax_matrix.add_patch(circle)
    
    # Connect dots vertically for each intersection with crisp lines
    for j in range(len(sorted_intersections)):
        connected = [k for k in range(len(sets_list)) if matrix_data[k, j] == 1]
        if len(connected) > 1:
            min_y, max_y = min(connected), max(connected)
            x_line = float(j)  # Use same x positioning
            ax_matrix.plot([x_line, x_line], [min_y, max_y], 'k-', linewidth=2.5, 
                          alpha=1.0, zorder=2, solid_capstyle='round')
    
    # Extract origin and ID for display names WITH sizes
    display_names = []
    for name in sets_list:
        set_size = len(gene_sets[name])
        
        if name == "Manual":
            display_names.append(f"Manual ({set_size})")
        elif name.startswith("UK_"):
            panel_id = name.replace("UK_", "")
            display_names.append(f"UK_{panel_id} ({set_size})")
        elif name.startswith("AUS_"):
            panel_id = name.replace("AUS_", "")
            display_names.append(f"AUS_{panel_id} ({set_size})")
        elif name.startswith("INT-"):
            panel_id = name.replace("INT-", "")
            display_names.append(f"INT_{panel_id} ({set_size})")
        else:
            display_names.append(f"{name} ({set_size})")
    
    # Set up y-axis with panel names INCLUDING sizes
    ax_matrix.set_yticks(range(len(sets_list)))
    ax_matrix.set_yticklabels(display_names, fontsize=10,)
    ax_matrix.set_xticks([])
    
    # Remove the xlabel
    ax_matrix.set_xlabel('')
    
    # Remove grid and spines for cleaner look
    ax_matrix.grid(False)
    ax_matrix.spines['top'].set_visible(False)
    ax_matrix.spines['right'].set_visible(False)
    ax_matrix.spines['bottom'].set_visible(False)
    ax_matrix.spines['left'].set_visible(False)
    
    # Invert y-axis to match the order of bars above
    ax_matrix.invert_yaxis()
    
    # Add a visual separator and better labels
    separator_pos = -1
    for j, (membership, _) in enumerate(sorted_intersections):
        if len(membership) > 1:
            separator_pos = j - 0.5
            break
    
    # Adjust layout with tighter spacing
    plt.tight_layout(pad=1.5)
    
    # Set clean background
    ax_matrix.set_facecolor('white')
    ax_bars.set_facecolor('white')
    
    return fig

def create_hpo_terms_table(hpo_details):
	if not hpo_details:
		return html.Div()
	
	table_data = []
	for term in hpo_details:
		table_data.append({
			"HPO ID": term["id"],
			"Term Name": term["name"],
			"Definition": term["definition"][:80] + "..." if len(term["definition"]) > 80 else term["definition"]
		})
	
	return html.Div([
		html.H5(f"Selected HPO Terms ({len(hpo_details)})", className="mb-3", style={"textAlign": "center", "fontSize": "16px"}),
		dash_table.DataTable(
			columns=[
				{"name": "HPO ID", "id": "HPO ID"},
				{"name": "Term Name", "id": "Term Name"},
				{"name": "Definition", "id": "Definition"}
			],
			data=table_data,
			style_table={
				"overflowX": "auto",
				"height": "520px",  
				"overflowY": "auto",
				"border": "1px solid #ddd",
				"borderRadius": "8px",
				"width": "100%"
			},
			style_cell={
				"textAlign": "left",
				"padding": "8px",
				"fontFamily": "Arial, sans-serif",
				"fontSize": "11px",
				"whiteSpace": "normal",
				"height": "auto",
				"minWidth": "60px",
				"maxWidth": "120px"
			},
			style_header={
				"fontWeight": "bold",
				"backgroundColor": "#f8f9fa",
				"border": "1px solid #ddd",
				"fontSize": "12px"
			},
			style_data={
				"backgroundColor": "#ffffff",
				"border": "1px solid #eee"
			},
			style_data_conditional=[
				{
					"if": {"row_index": "odd"},
					"backgroundColor": "#f8f9fa"
				}
			],
			page_action="native",  
			page_size=50,  
			virtualization=False,  
			tooltip_data=[
				{
					"Definition": {"value": term["definition"], "type": "text"}
					for column in ["HPO ID", "Term Name", "Definition"]
				} for term in hpo_details
			],
			tooltip_duration=None
		)
	], style={
		"border": "1px solid #999", 
		"padding": "10px", 
		"borderRadius": "8px",
		"backgroundColor": "#f8f9fa",
		"width": "100%",
		"height": "100%",  
		"display": "flex",
		"flexDirection": "column"
	})

# =============================================================================
# SIDEBAR COMPONENT
# =============================================================================

def create_sidebar():
	return dbc.Offcanvas(
		id="sidebar-offcanvas",
		title="Quick Panel Presets",
		is_open=False,
		placement="start",
		backdrop=False,
		style={"width": "350px"},
		children=[
			# Quick Presets Section
			html.Div([
				#html.H5("Quick Presets", className="mb-3"),
				html.Div(id="preset-buttons", children=[
					dbc.Button(
						[
							DashIconify(icon=preset["icon"], width=20, className="me-2"),
							preset["name"]
						],
						id={"type": "preset-btn", "index": key},
						color="light",
						className="mb-2 w-100 text-start",
						n_clicks=0
					)
					for key, preset in PANEL_PRESETS.items()
				])
			], className="mb-4")
		]
	)

# =============================================================================
# DATA INITIALIZATION
# =============================================================================

panels_uk_df = fetch_panels(PANELAPP_UK_BASE)
panels_au_df = fetch_panels(PANELAPP_AU_BASE)

internal_df = pd.read_csv("data/internal_panels.csv")
internal_panels = internal_df[["panel_id", "panel_name"]].drop_duplicates()

# =============================================================================
# DASH APP INITIALIZATION
# =============================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# =============================================================================
# APP LAYOUT
# =============================================================================

app.layout = dbc.Container([
	# Sidebar
	create_sidebar(),
	
	# Header with menu button
	html.Div([
		dbc.Button(
			[DashIconify(icon="mdi:menu", width=24)],
			id="sidebar-toggle",
			color="primary",
			className="me-2",
			style={"position": "absolute", "top": "20px", "right": "20px"}
		),
		html.H1("🧬Panel Builder"),
	], style={"position": "relative"}),
	
	html.Div(html.Hr(), id="hr-hidden", style={"display": "none"}),
	dbc.Row([
		dbc.Col([html.Label("PanelApp UK"), dcc.Dropdown(id="dropdown-uk", options=panel_options(panels_uk_df), placeholder="Select a UK panel", multi=True)]),
		dbc.Col([html.Label("PanelApp Australia"), dcc.Dropdown(id="dropdown-au", options=panel_options(panels_au_df), placeholder="Select a AUS panel", multi=True)]),
		dbc.Col([html.Label("Internal Panel"), dcc.Dropdown(id="dropdown-internal", options=internal_options(internal_panels), placeholder="Select an internal panel", multi=True)])
	]),
	dbc.Row([
		dbc.Col([
			html.Label("Add gene(s) manually:"), 
			dcc.Textarea(id="manual-genes", placeholder="Enter gene symbols, one per line", style={"width": "100%", "height": "100px"})
		]),
		dbc.Col([html.Label("Filter Genes by confidence level:"), dcc.Checklist(id="confidence-filter", options=[{"label": " Green (3)", "value": 3}, {"label": " Amber (2)", "value": 2}, {"label": " Red (1)", "value": 1}], value=[3, 2], inline=False)]),
		dbc.Col([
			html.Label("Search HPO terms:"),
			dcc.Dropdown(
				id="hpo-search-dropdown",
				placeholder="Type to search HPO terms",
				multi=True,
				searchable=True,
				options=[],
				style={"width": "100%", "marginBottom": "5px"}
			),
			html.Div([
				html.Small("ℹ️ HPO terms are auto-generated from Australia panels only (takes a few seconds)", 
						className="text-muted", style={"fontSize": "11px"}),
				dcc.Loading(
					id="hpo-loading",
					type="default",
					children=html.Div(id="hpo-loading-output"),
					style={
						"display": "inline-block", 
						"marginLeft": "40px",
						"marginTop": "2px",
						"transform": "scale(0.6)",  # Make spinner smaller
						"transformOrigin": "center"
					}
				)
			], style={"display": "flex", "alignItems": "center"})
		])
	]),
	html.Hr(),
	
	dbc.Row(
		dbc.Col(html.Div([
			dbc.Button("Reset", id="reset-btn", color="danger", className="me-2"),
			dbc.Button("Build Panel", id="load-genes-btn", color="primary", className="me-2"),
			dbc.Button("Import Panel", id="show-import-btn", color="info")
		], className="d-flex justify-content-center"), width=12),
		className="mb-3"
	),
	
	html.Div(
		id="import-section",
		style={"display": "none"},
		children=[
			dbc.Row([
				dbc.Col([
					html.Label("Import code:", style={"marginBottom": "10px"}),
					dcc.Textarea(
						id="panel-code-input", 
						placeholder="Paste a previously generated code here...", 
						style={"width": "100%", "height": "80px", "marginBottom": "10px"}
					),
					html.Div([
						dbc.Button("Import", id="import-panel-btn", color="success", className="me-2"),
						dbc.Button("Cancel", id="cancel-import-btn", color="secondary")
					], className="d-flex justify-content-center")
				], width={"size": 6, "offset": 3})
			])
		]
	),
	html.Div(html.Hr(), id="hr-venn", style={"display": "none"}),

	dcc.Loading(
		children=[
			html.Div(id="pie-chart-container", style={"marginBottom": "20px", "display": "none"}),
			html.Div([
				dbc.Row([
					dbc.Col(
						html.Div(id="venn-container"), 
						width=7, 
						style={
							"paddingRight": "15px", 
							"display": "flex", 
							"flexDirection": "column",
							"height": "600px"  
						}
					),
					dbc.Col(
						html.Div(id="hpo-terms-table-container"), 
						width=5, 
						style={
							"paddingLeft": "5px",
							"display": "flex", 
							"flexDirection": "column",
							"height": "600px"  
						}
					)
				], className="no-gutters", style={"display": "flex", "flexWrap": "nowrap"})
			], id="venn-hpo-row", style={"marginBottom": "20px", "display": "none"}),
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
			html.Div(dbc.Button("Generate Code", id="generate-code-btn", color="primary"), 
					style={"textAlign": "center", "marginBottom": "10px"}),
			html.Div([
				html.Label("Import Code:", style={"fontWeight": "bold", "marginBottom": "5px"}),
				dcc.Textarea(id="generated-code-output", 
							style={"width": "80%", "maxWidth": "900px", "height": "60px", 
								"margin": "0 auto", "display": "block"}, readOnly=True),
				html.Div(id="copy-notification", 
						style={"textAlign": "center", "marginTop": "5px", "height": "20px"})
			], id="generated-code-container-text"),
			html.Div([
				html.Label("Panel Summary:", style={"fontWeight": "bold", "marginBottom": "5px", "marginTop": "15px"}),
				dcc.Textarea(id="panel-summary-output", 
							style={"width": "80%", "maxWidth": "900px", "height": "60px", 
								"margin": "0 auto", "display": "block"}, readOnly=True),
				html.Div(id="copy-notification-summary", 
						style={"textAlign": "center", "marginTop": "5px", "height": "20px"})
			], id="panel-summary-container-text")
		]
	),
	html.Hr(),
], fluid=True)

# =============================================================================
# CALLBACKS - HPO MANAGEMENT
# =============================================================================

@app.callback(
	Output("hpo-search-dropdown", "value", allow_duplicate=True),
	Output("hpo-search-dropdown", "options", allow_duplicate=True),
	Output("hpo-loading-output", "children", allow_duplicate=True),
	Input("dropdown-au", "value"),  
	State("hpo-search-dropdown", "value"),
	State("hpo-search-dropdown", "options"),
	prevent_initial_call=True
)
def auto_generate_hpo_from_panels_preview(au_ids, current_hpo_values, current_hpo_options):
	if not au_ids:
		return current_hpo_values or [], current_hpo_options or [], html.Div()
	
	panel_hpo_terms = get_hpo_terms_from_panels(uk_ids=None, au_ids=au_ids)
	
	if not panel_hpo_terms:
		return current_hpo_values or [], current_hpo_options or [], html.Div()
	
	new_hpo_options = []
	new_hpo_values = []
	
	for hpo_id in panel_hpo_terms:
		hpo_details = fetch_hpo_term_details(hpo_id)
		
		option = {
			"label": f"{hpo_details['name']} ({hpo_id})",
			"value": hpo_id
		}
		new_hpo_options.append(option)
		new_hpo_values.append(hpo_id)
	
	current_values = current_hpo_values or []
	current_options = current_hpo_options or []
	
	all_values = list(set(current_values + new_hpo_values))
	
	existing_option_values = [opt["value"] for opt in current_options]
	all_options = current_options.copy()
	
	for option in new_hpo_options:
		if option["value"] not in existing_option_values:
			all_options.append(option)
	
	return all_values, all_options, html.Div()

@app.callback(
	Output("hpo-search-dropdown", "options", allow_duplicate=True),
	Input("hpo-search-dropdown", "search_value"),
	State("hpo-search-dropdown", "value"),
	State("hpo-search-dropdown", "options"),
	prevent_initial_call=True
)
def update_hpo_options(search_value, current_values, current_options):
	selected_options = []
	if current_values and current_options:
		selected_options = [opt for opt in current_options if opt["value"] in current_values]
	
	new_options = []
	if search_value and len(search_value.strip()) >= 2:
		new_options = search_hpo_terms(search_value)
	
	all_options = selected_options.copy()
	
	existing_values = [opt["value"] for opt in selected_options]
	for opt in new_options:
		if opt["value"] not in existing_values:
			all_options.append(opt)
	
	return all_options

# =============================================================================
# CALLBACKS - UI MANAGEMENT
# =============================================================================

@app.callback(
	Output("import-section", "style"),
	Input("show-import-btn", "n_clicks"),
	Input("cancel-import-btn", "n_clicks"),
	Input("import-panel-btn", "n_clicks"),
	Input("reset-btn", "n_clicks"),
	prevent_initial_call=True
)
def toggle_import_section(n_show, n_cancel, n_import, n_reset):
	ctx = dash.callback_context
	if not ctx.triggered:
		raise dash.exceptions.PreventUpdate
	
	triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
	
	if triggered_id == "show-import-btn":
		return {"display": "block", "marginBottom": "20px", "padding": "15px", "backgroundColor": "#f8f9fa", "borderRadius": "8px"}
	elif triggered_id in ["cancel-import-btn", "import-panel-btn", "reset-btn"]:
		return {"display": "none"}
	
	return dash.no_update

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

# =============================================================================
# CALLBACKS - NEW FEATURES (SIDEBAR)
# =============================================================================

# Toggle sidebar
@app.callback(
	Output("sidebar-offcanvas", "is_open"),
	Input("sidebar-toggle", "n_clicks"),
	State("sidebar-offcanvas", "is_open"),
	prevent_initial_call=True
)
def toggle_sidebar(n_clicks, is_open):
	if n_clicks:
		return not is_open
	return is_open

# Handle preset selection
@app.callback(
	Output("dropdown-uk", "value", allow_duplicate=True),
	Output("dropdown-au", "value", allow_duplicate=True),
	Output("dropdown-internal", "value", allow_duplicate=True),
	Output("confidence-filter", "value", allow_duplicate=True),
	Output("manual-genes", "value", allow_duplicate=True),
	Output("hpo-search-dropdown", "value", allow_duplicate=True),
	Output("hpo-search-dropdown", "options", allow_duplicate=True),
	Output("sidebar-offcanvas", "is_open", allow_duplicate=True),
	Input({"type": "preset-btn", "index": ALL}, "n_clicks"),
	State("hpo-search-dropdown", "options"),
	prevent_initial_call=True
)
def apply_preset(n_clicks_list, current_hpo_options):
	ctx = dash.callback_context
	if not ctx.triggered or all(n == 0 for n in n_clicks_list):
		raise dash.exceptions.PreventUpdate
	
	# Find which preset was clicked
	prop_id = ctx.triggered[0]["prop_id"]
	preset_key = json.loads(prop_id.split(".")[0])["index"]
	preset = PANEL_PRESETS[preset_key]
	
	# Get all values from preset with defaults
	uk_panels = preset.get("uk_panels", [])
	au_panels = preset.get("au_panels", [])
	internal_panels = preset.get("internal", [])
	conf_levels = preset.get("conf", [3, 2])  # Default to Green and Amber
	manual_genes_list = preset.get("manual", [])
	manual_genes_text = "\n".join(manual_genes_list) if manual_genes_list else ""
	hpo_terms = preset.get("hpo_terms", [])
	
	# Create HPO options for the preset terms
	updated_hpo_options = current_hpo_options or []
	existing_option_values = [opt["value"] for opt in updated_hpo_options]
	
	for hpo_id in hpo_terms:
		if hpo_id not in existing_option_values:
			hpo_details = fetch_hpo_term_details(hpo_id)
			option = {
				"label": f"{hpo_details['name']} ({hpo_id})",
				"value": hpo_id
			}
			updated_hpo_options.append(option)
	
	# Return all values AND close sidebar (False)
	return uk_panels, au_panels, internal_panels, conf_levels, manual_genes_text, hpo_terms, updated_hpo_options, False

# =============================================================================
# CALLBACKS - CODE GENERATION
# =============================================================================

@app.callback(
	Output("generated-code-output", "value", allow_duplicate=True),
	Output("panel-summary-output", "value", allow_duplicate=True),
	Input("generate-code-btn", "n_clicks"),
	State("dropdown-uk", "value"),
	State("dropdown-au", "value"),
	State("dropdown-internal", "value"),
	State("confidence-filter", "value"),
	State("manual-genes", "value"),
	State("hpo-search-dropdown", "value"),  
	prevent_initial_call=True
)
def generate_unique_code_and_summary(n_clicks, uk_ids, au_ids, internal_ids, confs, manual, hpo_terms):
	manual_list = [g.strip() for g in manual.strip().splitlines() if g.strip()] if manual else []
	config = {
		"uk": uk_ids or [],
		"au": au_ids or [],
		"internal": internal_ids or [],
		"conf": confs or [],
		"manual": manual_list,
		"hpo_terms": hpo_terms or []  
	}
	encoded = base64.urlsafe_b64encode(json.dumps(config).encode()).decode()
	
	# Generate panel summary
	summary = generate_panel_summary(
		uk_ids or [], 
		au_ids or [], 
		internal_ids or [], 
		confs or [], 
		manual_list, 
		panels_uk_df, 
		panels_au_df, 
		internal_panels
	)
	
	return encoded, summary

# =============================================================================
# CALLBACKS - RESET AND IMPORT
# =============================================================================

@app.callback(
	Output("dropdown-uk", "value"),
	Output("dropdown-au", "value"),
	Output("dropdown-internal", "value"),
	Output("confidence-filter", "value"),
	Output("manual-genes", "value"),
	Output("hpo-search-dropdown", "value"),  
	Output("hpo-search-dropdown", "options"), 
	Output("panel-code-input", "value"),
	Output("summary-table-output", "children"),
	Output("gene-table-output", "children"),
	Output("venn-container", "children"),
	Output("hpo-terms-table-container", "children"),
	Output("venn-hpo-row", "style"), 
	Output("pie-chart-container", "children"), 
	Output("pie-chart-container", "style"),     
	Output("gene-list-store", "data"),
	Output("generated-code-output", "value"),
	Output("panel-summary-output", "value"),
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
		return None, None, None, [3, 2], "", [], [], "", "", "", "", "", {"display": "none"}, "", {"display": "none"}, [], "", ""

	if triggered_id == "import-panel-btn" and code:
		try:
			decoded = base64.urlsafe_b64decode(code).decode()
			config = json.loads(decoded)
			
			hpo_terms = config.get("hpo_terms", [])
			hpo_options = []
			for hpo_id in hpo_terms:
				hpo_details = fetch_hpo_term_details(hpo_id)
				option = {
					"label": f"{hpo_details['name']} ({hpo_id})",
					"value": hpo_id
				}
				hpo_options.append(option)
			
			return (
				config.get("uk", []),
				config.get("au", []),
				config.get("internal", []),
				config.get("conf", []),
				"\n".join(config.get("manual", [])),
				hpo_terms,  
				hpo_options, 
				code,
				"", "", "", "", {"display": "none"}, "", {"display": "none"}, [], "", ""
			)
		except Exception:
			return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

	raise dash.exceptions.PreventUpdate

# =============================================================================
# CALLBACKS - MAIN PANEL PROCESSING
# =============================================================================

@app.callback(
	Output("summary-table-output", "children", allow_duplicate=True),
	Output("gene-table-output", "children", allow_duplicate=True),
	Output("venn-container", "children", allow_duplicate=True),
	Output("hpo-terms-table-container", "children", allow_duplicate=True),  
	Output("venn-hpo-row", "style", allow_duplicate=True),  
	Output("pie-chart-container", "children", allow_duplicate=True), 
	Output("pie-chart-container", "style", allow_duplicate=True),   
	Output("gene-list-store", "data", allow_duplicate=True),
	Output("hpo-search-dropdown", "value", allow_duplicate=True),  
	Output("hpo-search-dropdown", "options", allow_duplicate=True),
	Output("generated-code-output", "value", allow_duplicate=True),
	Output("panel-summary-output", "value", allow_duplicate=True), 
	Input("load-genes-btn", "n_clicks"),
	State("dropdown-uk", "value"),
	State("dropdown-au", "value"),
	State("dropdown-internal", "value"),
	State("confidence-filter", "value"),
	State("manual-genes", "value"),
	State("hpo-search-dropdown", "value"),  
	State("hpo-search-dropdown", "options"), 
	prevent_initial_call=True
)
def display_panel_genes(n_clicks, selected_uk_ids, selected_au_ids, selected_internal_ids, selected_confidences, manual_genes, selected_hpo_terms, current_hpo_options):
	if not n_clicks:
		return "", "", "", "", {"display": "none"}, "", {"display": "none"}, [], [], [], "", ""

	# Use only the selected HPO terms, don't auto-add more
	all_hpo_terms = selected_hpo_terms or []
	updated_hpo_options = current_hpo_options or []

	genes_combined = []
	gene_sets = {}
	manual_genes_list = []
	panel_dataframes = {} 
	panel_names = {}      
	panel_versions = {}    

	if selected_uk_ids:
		for pid in selected_uk_ids:
			df, panel_info = fetch_panel_genes(PANELAPP_UK_BASE, pid)
			df["confidence_level"] = df["confidence_level"].astype(int)
			
			panel_dataframes[f"UK_{pid}"] = df.copy()
			
			df_filtered = df[df["confidence_level"].isin(selected_confidences)]
			genes_combined.append(df_filtered[["gene_symbol", "confidence_level"]])
			gene_sets[f"UK_{pid}"] = set(df_filtered["gene_symbol"])
			
			panel_name = f"UK Panel {pid}"
			panel_version = None
			if panel_info:
				if 'name' in panel_info:
					panel_name = panel_info['name']
				if 'version' in panel_info:
					panel_version = panel_info['version']
			
			panel_names[f"UK_{pid}"] = panel_name
			panel_versions[f"UK_{pid}"] = panel_version

	if selected_au_ids:
		for pid in selected_au_ids:
			df, panel_info = fetch_panel_genes(PANELAPP_AU_BASE, pid)
			df["confidence_level"] = df["confidence_level"].astype(int)
			
			panel_dataframes[f"AUS_{pid}"] = df.copy()
			
			df_filtered = df[df["confidence_level"].isin(selected_confidences)]
			genes_combined.append(df_filtered[["gene_symbol", "confidence_level"]])
			gene_sets[f"AUS_{pid}"] = set(df_filtered["gene_symbol"])
			
			panel_name = f"AUS Panel {pid}"
			panel_version = None
			if panel_info:
				if 'name' in panel_info:
					panel_name = panel_info['name']
				if 'version' in panel_info:
					panel_version = panel_info['version']
			
			panel_names[f"AUS_{pid}"] = panel_name
			panel_versions[f"AUS_{pid}"] = panel_version

	if selected_internal_ids:
		for pid in selected_internal_ids:
			panel_df = internal_df[internal_df["panel_id"] == pid]
			panel_df["confidence_level"] = panel_df["confidence_level"].astype(int)
			
			panel_dataframes[f"INT-{pid}"] = panel_df.copy()
			
			panel_df_filtered = panel_df[panel_df["confidence_level"].isin(selected_confidences)].copy()
			genes_combined.append(panel_df_filtered[["gene_symbol", "confidence_level"]])
			gene_sets[f"INT-{pid}"] = set(panel_df_filtered["gene_symbol"])
			
			panel_name = next((row['panel_name'] for _, row in internal_panels.iterrows() if row['panel_id'] == pid), f"Internal Panel {pid}")
			panel_names[f"INT-{pid}"] = panel_name
			panel_versions[f"INT-{pid}"] = None 

	if manual_genes:
		manual_genes_list = [g.strip() for g in manual_genes.strip().splitlines() if g.strip()]
		if manual_genes_list:  
			manual_df = pd.DataFrame({"gene_symbol": manual_genes_list, "confidence_level": [0] * len(manual_genes_list)})
			genes_combined.append(manual_df)
			gene_sets["Manual"] = set(manual_genes_list)
			panel_dataframes["Manual"] = manual_df
			panel_names["Manual"] = "Manual Gene List"
			panel_versions["Manual"] = None

	if not genes_combined:
		return "No gene found.", "", "", "", {"display": "none"}, "", {"display": "none"}, [], all_hpo_terms, updated_hpo_options, "", ""

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

	# NEW VISUALIZATION LOGIC WITH UPSET PLOT AND FIXED VENN LABELS
	venn_component = html.Div()
	
	# Remove empty sets and manual genes for counting active panels
	active_panels = {k: v for k, v in gene_sets.items() if k != "Manual" and len(v) > 0}
	manual_genes_present = "Manual" in gene_sets and len(gene_sets["Manual"]) > 0
	
	total_active = len(active_panels)
	
	# Case 1: Only manual genes
	if total_active == 0 and manual_genes_present:
		single_panel_id = "Manual"
		panel_df = panel_dataframes[single_panel_id]
		panel_name = panel_names[single_panel_id]
		panel_version = panel_versions[single_panel_id]
		venn_component = generate_panel_pie_chart(panel_df, panel_name, panel_version)
	
	# Case 2: Only one active panel (with or without manual)
	elif total_active == 1:
		single_panel_id = next(iter(active_panels.keys()))
		panel_df = panel_dataframes[single_panel_id]
		panel_name = panel_names[single_panel_id]
		panel_version = panel_versions[single_panel_id]
		venn_component = generate_panel_pie_chart(panel_df, panel_name, panel_version)
	
	# Case 3: 2-3 panels - use traditional Venn diagram with FIXED LABELS
	elif 2 <= total_active <= 3:
		# Include manual genes if present for Venn
		venn_sets = active_panels.copy()
		if manual_genes_present:
			venn_sets["Manual"] = gene_sets["Manual"]
		
		valid_sets = [s for s in venn_sets.values() if len(s) > 0]
		if 2 <= len(valid_sets) <= 3:
			set_items = list(venn_sets.items())[:3]
			
			# MODIFIED: Use panel IDs instead of full names for labels
			labels = []
			for panel_key, _ in set_items:
				if panel_key == "Manual":
					labels.append("Manual")
				elif panel_key.startswith("UK_"):
					panel_id = panel_key.replace("UK_", "")
					labels.append(f"UK_{panel_id}")
				elif panel_key.startswith("AUS_"):
					panel_id = panel_key.replace("AUS_", "")
					labels.append(f"AUS_{panel_id}")
				elif panel_key.startswith("INT-"):
					panel_id = panel_key.replace("INT-", "")
					labels.append(f"INT_{panel_id}")
				else:
					labels.append(panel_key)
			
			sets = [s[1] for s in set_items]
			
			fig, ax = plt.subplots(figsize=(9, 5))
			try:
				if len(sets) == 2:
					venn2(sets, set_labels=labels)
				elif len(sets) == 3:
					venn3(sets, set_labels=labels)
				
				buf = io.BytesIO()
				plt.tight_layout()
				plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
				plt.close(fig)
				data = base64.b64encode(buf.getbuffer()).decode("ascii")
				
				venn_component = html.Div([
					html.Img(src=f"data:image/png;base64,{data}", 
							style={"maxWidth": "100%", "height": "auto", "display": "block", "margin": "auto"})
				], style={
					"border": "1px solid #999", 
					"padding": "10px", 
					"borderRadius": "8px", 
					"maxWidth": "100%", 
					"margin": "0",
					"height": "580px",  
					"display": "flex",
					"alignItems": "center",
					"justifyContent": "center"
				})
			except Exception as e:
				venn_component = html.Div(f"Could not generate Venn diagram: {str(e)}", style={
					"textAlign": "center", 
					"fontStyle": "italic", 
					"color": "#666",
					"height": "580px",
					"display": "flex",
					"alignItems": "center",
					"justifyContent": "center"
				})
	
	# Case 4: 4+ panels - use UpSet plot
	elif total_active >= 4:
		# Include manual genes if present
		upset_sets = active_panels.copy()
		if manual_genes_present:
			upset_sets["Manual"] = gene_sets["Manual"]
		
		try:
			fig = create_upset_plot(upset_sets, panel_names)
			if fig:
				buf = io.BytesIO()
				plt.tight_layout()
				plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
				plt.close(fig)
				data = base64.b64encode(buf.getbuffer()).decode("ascii")
				
				venn_component = html.Div([
					html.Img(src=f"data:image/png;base64,{data}", 
							style={"maxWidth": "100%", "height": "auto", "display": "block", "margin": "auto"})
				], style={
					"border": "1px solid #999", 
					"padding": "10px", 
					"borderRadius": "8px", 
					"maxWidth": "100%", 
					"margin": "0",
					"height": "580px",  
					"display": "flex",
					"flexDirection": "column",
					"justifyContent": "center"
				})
			else:
				venn_component = html.Div("Could not generate UpSet plot.", style={
					"textAlign": "center", 
					"fontStyle": "italic", 
					"color": "#666",
					"height": "580px",
					"display": "flex",
					"alignItems": "center",
					"justifyContent": "center"
				})
		except Exception as e:
			venn_component = html.Div(f"Error generating UpSet plot: {str(e)}", style={
				"textAlign": "center", 
				"fontStyle": "italic", 
				"color": "#666",
				"height": "580px",
				"display": "flex",
				"alignItems": "center",
				"justifyContent": "center"
			})
	
	else:
		venn_component = html.Div("No panels selected.", style={
			"textAlign": "center", 
			"fontStyle": "italic", 
			"color": "#666",
			"height": "580px",
			"display": "flex",
			"alignItems": "center",
			"justifyContent": "center"
		})

	pie_style = {"display": "none"} 
	venn_hpo_style = {"display": "block", "marginBottom": "20px"}

	hpo_details = []
	if all_hpo_terms:
		for term_id in all_hpo_terms:
			hpo_detail = fetch_hpo_term_details(term_id)
			hpo_details.append(hpo_detail)

	hpo_table_component = html.Div()
	if hpo_details:
		hpo_table_component = create_hpo_terms_table(hpo_details)

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

	return (summary_layout, 
			html.Div([
				html.Div(buttons, className="mb-3", style={"textAlign": "center"}),
				table_output,
				dcc.Store(id="gene-data-store", data=tables_by_level)
			]), 
			venn_component, 
			hpo_table_component,  
			venn_hpo_style,       
			html.Div(), 
			pie_style, 
			df_unique["Gene symbol"].tolist(),
			all_hpo_terms,       
			updated_hpo_options,
			"",  # Clear unique code
			"")  # Clear panel summary

# =============================================================================
# CALLBACKS - TABLE INTERACTION
# =============================================================================

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

# =============================================================================
# CLIENTSIDE CALLBACK - AUTO COPY TO CLIPBOARD WITH NOTIFICATION
# =============================================================================

# Add a clientside callback for the panel summary
app.clientside_callback(
	"""
	function(panel_summary) {
		if (panel_summary && panel_summary.trim() !== '') {
			// Use the modern Clipboard API if available
			if (navigator.clipboard && window.isSecureContext) {
				navigator.clipboard.writeText(panel_summary).then(function() {
					console.log('Panel summary copied to clipboard successfully');
					showCopyNotificationSummary('✅ Panel summary copied to clipboard!', 'success');
				}).catch(function(err) {
					console.error('Failed to copy panel summary: ', err);
					showCopyNotificationSummary('❌ Failed to copy panel summary', 'error');
				});
			} else {
				// Fallback for older browsers or non-secure contexts
				const textArea = document.createElement('textarea');
				textArea.value = panel_summary;
				textArea.style.position = 'fixed';
				textArea.style.left = '-999999px';
				textArea.style.top = '-999999px';
				document.body.appendChild(textArea);
				textArea.focus();
				textArea.select();
				try {
					document.execCommand('copy');
					console.log('Panel summary copied to clipboard successfully (fallback)');
					showCopyNotificationSummary('✅ Panel summary copied to clipboard!', 'success');
				} catch (err) {
					console.error('Failed to copy panel summary (fallback): ', err);
					showCopyNotificationSummary('❌ Failed to copy panel summary', 'error');
				}
				document.body.removeChild(textArea);
			}
		}
		return window.dash_clientside.no_update;
	}
	
	function showCopyNotificationSummary(message, type) {
		const notification = document.getElementById('copy-notification-summary');
		if (notification) {
			notification.textContent = message;
			notification.style.color = type === 'success' ? '#28a745' : '#dc3545';
			notification.style.fontWeight = 'bold';
			notification.style.fontSize = '14px';
			
			// Clear the notification after 3 seconds
			setTimeout(function() {
				notification.textContent = '';
			}, 3000);
		}
	}
	""",
	Output("panel-summary-output", "id"),  # Dummy output since we don't need to update anything
	Input("panel-summary-output", "value")
)

# =============================================================================
# APP RUN
# =============================================================================

#if __name__ == "__main__":
#	app.run(debug=True)

if __name__ == '__main__':
	port = int(os.environ.get("PORT", 8050))
	app.run(host="0.0.0.0", port=port)
