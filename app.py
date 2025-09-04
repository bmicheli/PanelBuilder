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
from datetime import datetime, timedelta
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import numpy as np
import threading
import schedule
from utils.panelapp_api import (
	fetch_panels,
	fetch_panel_genes,
	PANELAPP_UK_BASE,
	PANELAPP_AU_BASE
)

# =============================================================================
# PERFORMANCE IMPORTS
# =============================================================================

import concurrent.futures
from functools import lru_cache

# =============================================================================
# GLOBAL VARIABLES FOR PANEL DATA
# =============================================================================

panels_uk_df = None
panels_au_df = None
internal_df = None
internal_panels = None
last_refresh = None

# =============================================================================
# PANEL PRESETS CONFIGURATION
# =============================================================================

PANEL_PRESETS = {
	"neurodevelopmental": {
		"name": "Neurodevelopmental Disorders",
		"icon": "mdi:head-cog",
		"uk_panels": [285],
		"au_panels": [250],
		"internal": [8801],
		"conf": [3],
		"manual": [],
		"hpo_terms": [] 
	}
}

# =============================================================================
# PANEL REFRESH FUNCTIONS
# =============================================================================

def refresh_panels():
	"""Rafraîchir les données des panels"""
	global panels_uk_df, panels_au_df, internal_df, internal_panels, last_refresh
	
	try:
		print(f"🔄 Refreshing panels at {datetime.now()}")
		
		# Clear existing cache
		fetch_panel_genes_cached.cache_clear()
		fetch_hpo_term_details_cached.cache_clear()
		fetch_panel_disorders_cached.cache_clear()
		
		# Reload panels
		print("Fetching UK panels...")
		panels_uk_df = fetch_panels(PANELAPP_UK_BASE)
		print(f"✅ Loaded {len(panels_uk_df)} UK panels")
		
		print("Fetching AU panels...")
		panels_au_df = fetch_panels(PANELAPP_AU_BASE)
		print(f"✅ Loaded {len(panels_au_df)} AU panels")
		
		print("Loading internal panels...")
		internal_df, internal_panels = load_internal_panels_from_files()
		print(f"✅ Loaded {len(internal_panels)} internal panels")
		
		last_refresh = datetime.now()
		print(f"✅ Panels refresh completed at {last_refresh}")
		
	except Exception as e:
		print(f"❌ Error refreshing panels: {e}")

def schedule_panel_refresh():
	"""Planifier le rafraîchissement périodique"""
	# Rafraîchir tous les jours à 6h du matin
	schedule.every().monday.at("06:00").do(refresh_panels)
	
	def run_scheduler():
		while True:
			schedule.run_pending()
			time.sleep(60)  # Vérifier toutes les minutes
	
	# Lancer le scheduler dans un thread séparé
	scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
	scheduler_thread.start()
	print("📅 Panel refresh scheduler started")

def initialize_panels():
	"""Initialisation des panels au démarrage"""
	print("Initializing PanelBuilder...")
	start_time = time.time()
	
	refresh_panels()  # Premier chargement
	schedule_panel_refresh()  # Démarrer le scheduler
	
	print(f"Initialization completed in {time.time() - start_time:.2f} seconds")

# =============================================================================
# PERFORMANCE OPTIMIZATIONS - CACHING
# =============================================================================

@lru_cache(maxsize=200)
def fetch_panel_genes_cached(base_url, panel_id):
	"""Cached version of fetch_panel_genes - avoids repeated API calls"""
	try:
		return fetch_panel_genes(base_url, panel_id)
	except Exception as e:
		print(f"Error fetching panel {panel_id}: {e}")
		return pd.DataFrame(), {}

@lru_cache(maxsize=500)
def fetch_hpo_term_details_cached(term_id):
	"""Cached version of HPO term fetching"""
	return fetch_hpo_term_details(term_id)

@lru_cache(maxsize=100)
def fetch_panel_disorders_cached(base_url, panel_id):
	"""Cached version of panel disorders fetching"""
	return fetch_panel_disorders(base_url, panel_id)

# =============================================================================
# PERFORMANCE OPTIMIZATIONS - PARALLEL PROCESSING
# =============================================================================

def fetch_panels_parallel(uk_ids=None, au_ids=None, max_workers=5):
	"""Fetch multiple panels in parallel instead of sequentially"""
	results = {}
	
	if not uk_ids and not au_ids:
		return results
	
	with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
		# Submit all requests
		future_to_panel = {}
		
		if uk_ids:
			for panel_id in uk_ids:
				future = executor.submit(fetch_panel_genes_cached, PANELAPP_UK_BASE, panel_id)
				future_to_panel[future] = ('UK', panel_id)
		
		if au_ids:
			for panel_id in au_ids:
				future = executor.submit(fetch_panel_genes_cached, PANELAPP_AU_BASE, panel_id)
				future_to_panel[future] = ('AU', panel_id)
		
		# Collect results with timeout
		for future in concurrent.futures.as_completed(future_to_panel, timeout=30):
			source, panel_id = future_to_panel[future]
			try:
				df, panel_info = future.result()
				results[f"{source}_{panel_id}"] = (df, panel_info)
			except Exception as e:
				print(f"Failed to fetch {source} panel {panel_id}: {e}")
				# Add empty result to avoid breaking the app
				results[f"{source}_{panel_id}"] = (pd.DataFrame(), {})
	
	return results

def fetch_hpo_terms_parallel(hpo_terms, max_workers=10):
	"""Fetch multiple HPO terms in parallel"""
	if not hpo_terms:
		return []
	
	results = []
	with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
		future_to_term = {
			executor.submit(fetch_hpo_term_details_cached, term_id): term_id 
			for term_id in hpo_terms
		}
		
		for future in concurrent.futures.as_completed(future_to_term, timeout=20):
			try:
				result = future.result()
				results.append(result)
			except Exception as e:
				term_id = future_to_term[future]
				print(f"Failed to fetch HPO term {term_id}: {e}")
				# Add fallback result
				results.append({
					"id": term_id,
					"name": term_id,
					"definition": "Unable to fetch definition"
				})
	
	return results

# =============================================================================
# INTERNAL PANELS MANAGEMENT
# =============================================================================

def load_internal_panels_from_files(directory_path="data/internal_panels"):
	"""Load internal panels directly from .txt files in the specified directory"""
	
	internal_data = []
	panel_info = []
	
	if not os.path.exists(directory_path):
		print(f"Warning: Directory {directory_path} does not exist")
		return pd.DataFrame(), pd.DataFrame()
	
	txt_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.txt')])
	
	def generate_stable_id(filename):

		import hashlib
		
		# Parse filename to extract just the panel name part
		base_name = filename.replace('.txt', '')
		parts = base_name.split('_')
		
		# Find version part (starts with 'v')
		version_idx = -1
		for i, part in enumerate(parts):
			if part.startswith('v') and part[1:].isdigit():
				version_idx = i
				break
		
		if version_idx == -1:
			# No version found, use whole filename
			panel_name_for_id = base_name
		else:
			# Extract gene count (part before version)
			if version_idx > 0 and parts[version_idx - 1].isdigit():
				# Panel name is everything before gene count
				panel_name_parts = parts[:version_idx - 1]
			else:
				# Panel name is everything before version
				panel_name_parts = parts[:version_idx]
			
			panel_name_for_id = '_'.join(panel_name_parts)
		
		# Generate hash from panel name only (not gene count or version)
		hash_obj = hashlib.md5(panel_name_for_id.encode())
		hash_hex = hash_obj.hexdigest()[:8]  # Take first 8 characters
		hash_int = int(hash_hex, 16) % 8999 + 2000  # Range: 2000-10999
		return hash_int
	
	for file_name in txt_files:
		try:
			# Parse filename: Name_NumberOfGenes_vVersion.txt
			base_name = file_name.replace('.txt', '')
			parts = base_name.split('_')
			
			# Find version part (starts with 'v')
			version_idx = -1
			for i, part in enumerate(parts):
				if part.startswith('v') and part[1:].isdigit():
					version_idx = i
					break
			
			if version_idx == -1:
				print(f"Warning: Could not parse version from {file_name}")
				continue
			
			# Extract version
			version = int(parts[version_idx][1:])
			
			# Extract gene count (part before version)
			if version_idx > 0 and parts[version_idx - 1].isdigit():
				gene_count_from_filename = int(parts[version_idx - 1])
				# Panel name is everything before gene count
				panel_name_parts = parts[:version_idx - 1]
			else:
				# Fallback: assume last number before version is gene count
				gene_count_from_filename = 0
				panel_name_parts = parts[:version_idx]
			
			# Keep original filename format for panel name (with underscores)
			panel_name = '_'.join(panel_name_parts)
			
			# Generate stable ID based on filename
			panel_id = generate_stable_id(file_name)
			
			# Read genes from file
			file_path = os.path.join(directory_path, file_name)
			with open(file_path, 'r', encoding='utf-8') as f:
				genes = [line.strip() for line in f if line.strip()]
			
			actual_gene_count = len(genes)
			
			# Add panel info
			panel_info.append({
				'panel_id': panel_id,
				'panel_name': panel_name,
				'version': version,
				'gene_count': actual_gene_count,
				'gene_count_filename': gene_count_from_filename,
				'file_name': file_name,
				'base_name': base_name  # Store original filename without extension
			})
			
			# Add genes with default confidence level 3 (Green)
			for gene in genes:
				internal_data.append({
					'panel_id': panel_id,
					'panel_name': panel_name,
					'gene_symbol': gene,
					'confidence_level': 3  # Default to Green confidence
				})
		
		except Exception as e:
			print(f"Error processing file {file_name}: {e}")
			continue
	
	internal_df = pd.DataFrame(internal_data)
	internal_panels = pd.DataFrame(panel_info).sort_values('panel_id')  # Sort by ID for display
	
	return internal_df, internal_panels

# =============================================================================
# OPTIMIZED UTILITY FUNCTIONS - CONFIDENCE LEVEL PROCESSING
# =============================================================================

def clean_confidence_level_fast(df):
	"""Vectorized confidence level cleaning - much faster than original"""
	if 'confidence_level' not in df.columns:
		return df
	
	df = df.copy()
	
	# Create mapping dictionary for fast lookup
	confidence_map = {
		'3': 3, '3.0': 3, 'green': 3, 'high': 3,
		'2': 2, '2.0': 2, 'amber': 2, 'orange': 2, 'medium': 2,
		'1': 1, '1.0': 1, 'red': 1, 'low': 1,
		'0': 0, '0.0': 0, '': 0, 'nan': 0, 'none': 0
	}
	
	# Vectorized operation - much faster than apply()
	df['confidence_level'] = (df['confidence_level']
							.astype(str)
							.str.lower()
							.str.strip()
							.map(confidence_map)
							.fillna(0)
							.astype(int))
	
	return df

def deduplicate_genes_fast(df_all):
	"""Fast gene deduplication with proper confidence handling"""
	if df_all.empty:
		return df_all
	
	# Ensure confidence_level is numeric
	df_all["confidence_level"] = pd.to_numeric(df_all["confidence_level"], errors='coerce').fillna(0).astype(int)
	
	# Remove any rows with completely missing gene symbols
	df_all = df_all[df_all["gene_symbol"].notna() & (df_all["gene_symbol"] != "")]
	
	# Sort by confidence descending, then by gene name for reproducible results
	df_sorted = df_all.sort_values(['confidence_level', 'gene_symbol'], 
								ascending=[False, True])
	
	# Keep first occurrence (highest confidence) - much faster than groupby
	df_unique = df_sorted.drop_duplicates(subset=['gene_symbol'], keep='first')
	
	# Sort final result by confidence then gene name
	return df_unique.sort_values(['confidence_level', 'gene_symbol'], 
								ascending=[False, True])

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
			hpo_terms = fetch_panel_disorders_cached(PANELAPP_AU_BASE, panel_id)
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
	options = []
	for _, row in df.iterrows():
		version_text = f" v{row['version']}" if 'version' in row and pd.notna(row['version']) else ""
		# Use the base_name for display, replacing underscores with spaces for readability
		display_name = row['panel_name'].replace('_', ' ')
		label = f"{display_name}{version_text} (ID {row['panel_id']})"
		options.append({"label": label, "value": row["panel_id"]})
	return options

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
				# Ajout de l'ID dans le summary
				summary_parts.append(f"PanelApp_UK({panel_id})/{panel_name}{version}{confidence_suffix}")
	
	# Process AU panels
	if au_ids:
		for panel_id in au_ids:
			panel_row = panels_au_df[panels_au_df['id'] == panel_id]
			if not panel_row.empty:
				panel_info = panel_row.iloc[0]
				panel_name = panel_info['name'].replace(' ', '_').replace('/', '_').replace(',', '_')
				version = f"_v{panel_info['version']}" if pd.notna(panel_info.get('version')) else ""
				# Ajout de l'ID dans le summary
				summary_parts.append(f"PanelApp_AUS({panel_id})/{panel_name}{version}{confidence_suffix}")
	
	# Process Internal panels
	if internal_ids:
		for panel_id in internal_ids:
			panel_row = internal_panels[internal_panels['panel_id'] == panel_id]
			if not panel_row.empty:
				panel_info = panel_row.iloc[0]
				# Use the base_name (original filename without .txt) for the summary
				base_name = panel_info.get('base_name', panel_info['panel_name'])
				gene_count = panel_info['gene_count']
				summary_parts.append(f"Panel_HUG/{base_name}")
	
	# Add manual genes
	if manual_genes_list:
		summary_parts.extend(manual_genes_list)
	
	return ",".join(summary_parts)

# =============================================================================
# UTILITY FUNCTIONS - CHART GENERATION
# =============================================================================

def generate_panel_pie_chart(panel_df, panel_name, version=None):
	panel_df = panel_df[panel_df['confidence_level'] != 0].copy()
	
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
	"""Create an UpSet plot for visualizing intersections of multiple sets with dynamic sizing"""
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
	
	# Combine: panels first, then intersections (limit to max 15 total)
	sorted_intersections = single_sets + multi_sets
	max_intersections = min(15, len(sorted_intersections))
	sorted_intersections = sorted_intersections[:max_intersections]
	
	# DYNAMIC WIDTH SIZING BASED ON NUMBER OF INTERSECTIONS (FIXED HEIGHT)
	num_intersections = len(sorted_intersections)
	num_sets = len(sets_list)
	figure_height = 5
	dpi = 180

	if num_intersections <= 6:
		figure_width = 6.5  # Compact for simple plots
	elif num_intersections <= 10:
		figure_width = 8.5  # Medium width
	else:
		figure_width = 10  # Max width cap

	
	# Create figure with dynamic width, fixed height
	fig, (ax_bars, ax_matrix) = plt.subplots(2, 1, figsize=(figure_width, figure_height), dpi=dpi,
										gridspec_kw={'height_ratios': [1, 1]})
	
	# Dynamic bar width: better utilization of space
	if num_intersections <= 6:
		bar_width = 0.8  # Wider bars for fewer intersections
	elif num_intersections <= 10:
		bar_width = 0.7  # Medium width
	else:
		bar_width = 0.6  # Narrower for many intersections
	
	# Top plot: intersection sizes (VERTICAL bars)
	intersection_sizes = [len(genes) for _, genes in sorted_intersections]
	x_pos = np.arange(len(intersection_sizes))
	
	# Create vertical bars with different colors for panels vs intersections
	bar_colors = []
	for membership, _ in sorted_intersections:
		if len(membership) == 1:
			bar_colors.append('#3498db')  # Blue for individual panels
		else:
			bar_colors.append('#2c3e50')  # Dark for intersections
	
	bars = ax_bars.bar(x_pos, intersection_sizes, color=bar_colors, alpha=0.8, width=bar_width,
					edgecolor='white', linewidth=1)
	
	# Dynamic font sizes optimized for 3 categories
	if num_intersections <= 6:
		title_fontsize = 14
		label_fontsize = 12  
		value_fontsize = 10
		ytick_fontsize = 10
	elif num_intersections <= 10:
		title_fontsize = 13
		label_fontsize = 11
		value_fontsize = 9
		ytick_fontsize = 9
	else:
		title_fontsize = 12
		label_fontsize = 10
		value_fontsize = 8
		ytick_fontsize = 8
	
	ax_bars.set_ylabel('Number of Genes', fontsize=label_fontsize, fontweight='bold')
	ax_bars.set_title('Gene Panel Intersections', fontsize=title_fontsize, fontweight='bold', pad=20)
	ax_bars.set_xticks([])
	ax_bars.grid(True, alpha=0.3, axis='y')
	ax_bars.spines['top'].set_visible(False)
	ax_bars.spines['right'].set_visible(False)
	
	# Set x-axis limits to match matrix exactly
	ax_bars.set_xlim(-0.5, len(sorted_intersections) - 0.5)
	
	# Add value labels on top of bars with dynamic positioning
	max_height = max(intersection_sizes) if intersection_sizes else 1
	for i, (bar, size) in enumerate(zip(bars, intersection_sizes)):
		ax_bars.text(i, bar.get_height() + max_height * 0.01, 
					str(size), ha='center', va='bottom', fontweight='bold', 
					fontsize=value_fontsize)
	
	# Bottom plot: binary matrix with dynamic sizing
	matrix_data = np.zeros((len(sets_list), len(sorted_intersections)))
	for j, (membership, _) in enumerate(sorted_intersections):
		for i in membership:
			matrix_data[i, j] = 1
	
	# Clear the matrix plot
	ax_matrix.clear()
	
	# Set up the matrix plot with EXACT alignment to bars
	ax_matrix.set_xlim(-0.5, len(sorted_intersections) - 0.5)
	ax_matrix.set_ylim(-0.5, len(sets_list) - 0.5)
	

	circle_radius = 0.1
	line_width = 2.0
	
	# Draw the matrix with dynamic circles aligned with bars
	for i in range(len(sets_list)):
		for j in range(len(sorted_intersections)):
			x_center = float(j)
			y_center = float(i)
			
			if matrix_data[i, j] == 1:
				# Draw filled circle (black) - dynamically sized
				circle = plt.Circle((x_center, y_center), circle_radius, 
								color='black', zorder=2, clip_on=False)
				ax_matrix.add_patch(circle)
			else:
				# Draw empty circle (light gray) - smaller and subtle  
				empty_radius = circle_radius * 0.8  # Smaller ratio for better contrast
				circle = plt.Circle((x_center, y_center), empty_radius, 
								fill=False, color='lightgray', 
								linewidth=0.8, alpha=0.5, zorder=2, clip_on=False)
				ax_matrix.add_patch(circle)
	
	# Connect dots vertically for each intersection with dynamic lines
	for j in range(len(sorted_intersections)):
		connected = [k for k in range(len(sets_list)) if matrix_data[k, j] == 1]
		if len(connected) > 1:
			min_y, max_y = min(connected), max(connected)
			x_line = float(j)
			ax_matrix.plot([x_line, x_line], [min_y, max_y], 'k-', linewidth=line_width, 
						alpha=0.95, zorder=1, solid_capstyle='round')
	
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
	ax_matrix.set_yticklabels(display_names, fontsize=ytick_fontsize)
	ax_matrix.set_xticks([])
	
	# Remove the xlabel
	ax_matrix.set_xlabel('')
	
	# Remove grid and spines for cleaner look
	ax_matrix.grid(False)
	for spine in ax_matrix.spines.values():
		spine.set_visible(False)
	
	# Invert y-axis to match the order of bars above
	ax_matrix.invert_yaxis()
	
	# Optimized padding for 3 categories
	if num_intersections <= 10:
		pad = 1.8
	else:
		pad = 1.2
	
	plt.tight_layout(pad=pad)
	
	# Set clean background
	ax_matrix.set_facecolor('white')
	ax_bars.set_facecolor('white')
	fig.patch.set_facecolor('white')
	
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
		html.H5(f"HPO Terms ({len(hpo_details)})", className="mb-3", style={"textAlign": "center", "fontSize": "16px"}),
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
		title="Panel Presets",
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
# DASH APP INITIALIZATION
# =============================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# =============================================================================
# DATA INITIALIZATION WITH SCHEDULED REFRESH
# =============================================================================

initialize_panels()

# =============================================================================
# APP LAYOUT
# =============================================================================

app.layout = dbc.Container([
	# Download component for gene export
	dcc.Download(id="download-genes"),
	
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
		html.H1("Panel Builder"),
#        html.Small("Optimized for Performance", className="text-muted", style={"fontSize": "12px"})
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
				html.Small("HPO terms are auto-generated from Australia panels only (takes a few seconds)", 
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
#			dbc.Button("Import Panel", id="show-import-btn", color="info")
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
	dcc.Store(id="gene-data-store"),
	html.Hr(),
	html.Div(
		id="generate-code-section",
		style={"display": "none", "width": "100%"},
		children=[
			html.Div([
				dbc.Button("Generate Code", id="generate-code-btn", color="primary", className="me-2"),
				dbc.Button("Export Genes", id="export-genes-btn", color="success")
			], style={"textAlign": "center", "marginBottom": "10px"}),
			
			# Garder l'élément generated-code-output caché pour éviter de casser les callbacks
			html.Div([
				dcc.Textarea(id="generated-code-output", style={"display": "none"}, readOnly=True)
			], style={"display": "none"}),
			
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
	
	# Use parallel fetching for HPO terms
	hpo_details_list = fetch_hpo_terms_parallel(panel_hpo_terms)
	
	new_hpo_options = []
	new_hpo_values = []
	
	for hpo_details in hpo_details_list:
		hpo_id = hpo_details['id']
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
# CALLBACKS - SIDEBAR MANAGEMENT
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

# Handle preset selection with complete reset functionality
@app.callback(
	# Input controls (existing)
	Output("dropdown-uk", "value", allow_duplicate=True),
	Output("dropdown-au", "value", allow_duplicate=True),
	Output("dropdown-internal", "value", allow_duplicate=True),
	Output("confidence-filter", "value", allow_duplicate=True),
	Output("manual-genes", "value", allow_duplicate=True),
	Output("hpo-search-dropdown", "value", allow_duplicate=True),
	Output("hpo-search-dropdown", "options", allow_duplicate=True),
	Output("sidebar-offcanvas", "is_open", allow_duplicate=True),
	
	# Output displays (to reset them)
	Output("summary-table-output", "children", allow_duplicate=True),
	Output("gene-table-output", "children", allow_duplicate=True),
	Output("venn-container", "children", allow_duplicate=True),
	Output("hpo-terms-table-container", "children", allow_duplicate=True),
	Output("venn-hpo-row", "style", allow_duplicate=True),
	Output("pie-chart-container", "children", allow_duplicate=True),
	Output("pie-chart-container", "style", allow_duplicate=True),
	Output("gene-list-store", "data", allow_duplicate=True),
	Output("generated-code-output", "value", allow_duplicate=True),
	Output("panel-summary-output", "value", allow_duplicate=True),
	Output("gene-data-store", "data", allow_duplicate=True),
	Output("generate-code-section", "style", allow_duplicate=True),
	Output("hr-venn", "style", allow_duplicate=True),
	Output("hr-summary", "style", allow_duplicate=True),
	Output("hr-table", "style", allow_duplicate=True),
	
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
	
	# Create HPO options for the preset terms - use parallel fetching
	updated_hpo_options = current_hpo_options or []
	existing_option_values = [opt["value"] for opt in updated_hpo_options]
	
	new_hpo_terms = [term for term in hpo_terms if term not in existing_option_values]
	if new_hpo_terms:
		hpo_details_list = fetch_hpo_terms_parallel(new_hpo_terms)
		
		for hpo_details in hpo_details_list:
			option = {
				"label": f"{hpo_details['name']} ({hpo_details['id']})",
				"value": hpo_details['id']
			}
			updated_hpo_options.append(option)
	
	# Reset all output displays to initial empty/hidden state
	empty_summary = ""
	empty_gene_table = ""
	empty_venn = ""
	empty_hpo_table = ""
	hidden_style = {"display": "none"}
	empty_pie = ""
	empty_gene_list = []
	empty_code = ""
	empty_panel_summary = ""
	empty_gene_data = {}
	
	# Return all values: input controls + reset outputs + close sidebar (False)
	return (
		# Input controls
		uk_panels, au_panels, internal_panels, conf_levels, manual_genes_text, 
		hpo_terms, updated_hpo_options, False,
		
		# Reset outputs
		empty_summary,           # summary-table-output
		empty_gene_table,        # gene-table-output
		empty_venn,              # venn-container
		empty_hpo_table,         # hpo-terms-table-container
		hidden_style,            # venn-hpo-row
		empty_pie,               # pie-chart-container
		hidden_style,            # pie-chart-container style
		empty_gene_list,         # gene-list-store
		empty_code,              # generated-code-output
		empty_panel_summary,     # panel-summary-output
		empty_gene_data,         # gene-data-store
		hidden_style,            # generate-code-section
		hidden_style,            # hr-venn
		hidden_style,            # hr-summary
		hidden_style             # hr-table
	)

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
# CALLBACKS - GENE EXPORT
# =============================================================================

@app.callback(
	Output("download-genes", "data"),
	Input("export-genes-btn", "n_clicks"),
	State("gene-list-store", "data"),
	State("dropdown-uk", "value"),
	State("dropdown-au", "value"),
	State("dropdown-internal", "value"),
	State("manual-genes", "value"),
	prevent_initial_call=True
)
def export_gene_list(n_clicks, gene_list, uk_ids, au_ids, internal_ids, manual_genes):
	if n_clicks and gene_list:
		# Generate filename based on current timestamp
		timestamp = datetime.now().strftime("%Y%m%d_%H%M")
		
		# Create a simple descriptive filename
		panel_parts = []
		if uk_ids:
			panel_parts.append(f"UK{len(uk_ids)}")
		if au_ids:
			panel_parts.append(f"AU{len(au_ids)}")
		if internal_ids:
			panel_parts.append(f"INT{len(internal_ids)}")
		if manual_genes and manual_genes.strip():
			manual_count = len([g.strip() for g in manual_genes.strip().splitlines() if g.strip()])
			panel_parts.append(f"MAN{manual_count}")
		
		panel_desc = "_".join(panel_parts) if panel_parts else "Panel"
		filename = f"CustomPanel_{len(gene_list)}genes_{timestamp}.txt"
		
		# Create file content: one gene per line
		content = "\n".join(sorted(gene_list))
		
		return dcc.send_string(content, filename)
	
	raise dash.exceptions.PreventUpdate

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
			# Use parallel fetching for HPO terms
			hpo_details_list = fetch_hpo_terms_parallel(hpo_terms)
			hpo_options = []
			for hpo_details in hpo_details_list:
				option = {
					"label": f"{hpo_details['name']} ({hpo_details['id']})",
					"value": hpo_details['id']
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
# CALLBACKS - MAIN PANEL PROCESSING (OPTIMIZED)
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
	Output("gene-data-store", "data", allow_duplicate=True),
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
def display_panel_genes_optimized(n_clicks, selected_uk_ids, selected_au_ids, 
								selected_internal_ids, selected_confidences, 
								manual_genes, selected_hpo_terms, current_hpo_options):
	"""OPTIMIZED version of the main callback with performance improvements"""
	if not n_clicks:
		return "", "", "", "", {"display": "none"}, "", {"display": "none"}, [], [], [], "", "", {}

	start_time = time.time()
	print(f"Building panel with {len(selected_uk_ids or [])} UK, {len(selected_au_ids or [])} AU, {len(selected_internal_ids or [])} internal panels...")
	
	# Use existing HPO terms
	all_hpo_terms = selected_hpo_terms or []
	updated_hpo_options = current_hpo_options or []

	genes_combined = []
	gene_sets = {}
	manual_genes_list = []
	panel_dataframes = {} 
	panel_names = {}      
	panel_versions = {}    

	# PARALLEL FETCHING - This is the biggest performance improvement
	if selected_uk_ids or selected_au_ids:
		panel_results = fetch_panels_parallel(selected_uk_ids, selected_au_ids)
		
		# Process results from parallel fetching
		for result_key, (df, panel_info) in panel_results.items():
			if df.empty:
				continue
				
			source, pid_str = result_key.split('_', 1)
			pid = int(pid_str)
			
			# Fast confidence cleaning
			df = clean_confidence_level_fast(df)
			panel_dataframes[result_key] = df.copy()
			
			# Filter by confidence
			df_filtered = df[df["confidence_level"].isin(selected_confidences)].copy()
			
			# Ensure required columns exist with default values
			required_cols = ["gene_symbol", "confidence_level", "omim_id", "hgnc_id", "entity_type", "biotype", "mode_of_inheritance"]
			for col in required_cols:
				if col not in df_filtered.columns:
					df_filtered[col] = "" if col != "confidence_level" else 0
			
			genes_combined.append(df_filtered[required_cols])
			gene_sets[result_key] = set(df_filtered["gene_symbol"])
			
			# Panel names
			panel_name = f"{source} Panel {pid}"
			panel_version = None
			if panel_info:
				if 'name' in panel_info:
					panel_name = panel_info['name']
				if 'version' in panel_info:
					panel_version = panel_info['version']
			
			panel_names[result_key] = panel_name
			panel_versions[result_key] = panel_version

	# Process internal panels (optimized)
	if selected_internal_ids:
		for pid in selected_internal_ids:
			try:
				panel_df = internal_df[internal_df["panel_id"] == pid].copy()
				
				# Fast confidence cleaning
				panel_df = clean_confidence_level_fast(panel_df)
				
				# Add missing columns for internal panels with default values
				panel_df["omim_id"] = ""
				panel_df["hgnc_id"] = ""
				panel_df["entity_type"] = "gene"
				panel_df["biotype"] = "unknown"
				panel_df["mode_of_inheritance"] = "unknown"
				
				panel_dataframes[f"INT-{pid}"] = panel_df.copy()
				
				panel_df_filtered = panel_df[panel_df["confidence_level"].isin(selected_confidences)].copy()
				required_cols = ["gene_symbol", "confidence_level", "omim_id", "hgnc_id", "entity_type", "biotype", "mode_of_inheritance"]
				genes_combined.append(panel_df_filtered[required_cols])
				gene_sets[f"INT-{pid}"] = set(panel_df_filtered["gene_symbol"])
				
				panel_name = next((row['panel_name'] for _, row in internal_panels.iterrows() if row['panel_id'] == pid), f"Internal Panel {pid}")
				panel_names[f"INT-{pid}"] = panel_name
				panel_version = next((row['version'] for _, row in internal_panels.iterrows() if row['panel_id'] == pid), None)
				panel_versions[f"INT-{pid}"] = panel_version
				
			except Exception as e:
				print(f"Error processing internal panel {pid}: {e}")
				continue

	# Handle manual genes
	if manual_genes:
		manual_genes_list = [g.strip() for g in manual_genes.strip().splitlines() if g.strip()]
		if manual_genes_list:  
			manual_df = pd.DataFrame({
				"gene_symbol": manual_genes_list, 
				"confidence_level": [0] * len(manual_genes_list),
				"omim_id": [""] * len(manual_genes_list),
				"hgnc_id": [""] * len(manual_genes_list),
				"entity_type": ["gene"] * len(manual_genes_list),
				"biotype": ["manual"] * len(manual_genes_list),
				"mode_of_inheritance": ["manual"] * len(manual_genes_list)
			})
			genes_combined.append(manual_df)
			gene_sets["Manual"] = set(manual_genes_list)
			panel_dataframes["Manual"] = manual_df
			panel_names["Manual"] = "Manual Gene List"
			panel_versions["Manual"] = None

	if not genes_combined:
		return "No gene found.", "", "", "", {"display": "none"}, "", {"display": "none"}, [], all_hpo_terms, updated_hpo_options, "", "", {}

	# FAST GENE PROCESSING
	df_all = pd.concat(genes_combined, ignore_index=True)
	df_all = df_all.copy()
	
	# Remove any rows with completely missing gene symbols
	df_all = df_all[df_all["gene_symbol"].notna() & (df_all["gene_symbol"] != "")]
	
	if df_all.empty:
		return "No valid genes found.", "", "", "", {"display": "none"}, "", {"display": "none"}, [], all_hpo_terms, updated_hpo_options, "", "", {}
	
	# Fast deduplication
	df_unique = deduplicate_genes_fast(df_all)
	
	print(f"Data processing completed in {time.time() - start_time:.2f} seconds")
	
	# Rename columns for display
	df_unique = df_unique.rename(columns={
		"gene_symbol": "Gene Symbol",
		"confidence_level": "Confidence",
		"omim_id": "OMIM ID",
		"hgnc_id": "HGNC ID", 
		"entity_type": "Type",
		"biotype": "Biotype",
		"mode_of_inheritance": "Mode of Inheritance"
	})

	total_genes = pd.DataFrame({"Number of genes in panel": [df_unique.shape[0]]})
	summary = df_unique.groupby("Confidence").size().reset_index(name="Number of genes")
	summary_table = dbc.Row([
		dbc.Col(dash_table.DataTable(columns=[{"name": col, "id": col} for col in total_genes.columns], data=total_genes.to_dict("records"), style_cell={"textAlign": "left"}, style_table={"marginBottom": "20px", "width": "100%"}), width=4),
		dbc.Col(dash_table.DataTable(columns=[{"name": col, "id": col} for col in ["Confidence", "Number of genes"]], data=summary.to_dict("records"), style_cell={"textAlign": "left"}, style_table={"width": "100%"}), width=8)
	])

	# Visualization logic (Venn diagrams/UpSet plots)
	venn_component = html.Div()
	all_sets = {k: v for k, v in gene_sets.items() if len(v) > 0}
	total_sets = len(all_sets)

	if total_sets == 1:
		single_panel_id = next(iter(all_sets.keys()))
		panel_df = panel_dataframes[single_panel_id]
		panel_name = panel_names[single_panel_id]
		panel_version = panel_versions[single_panel_id]
		venn_component = generate_panel_pie_chart(panel_df, panel_name, panel_version)
	elif 2 <= total_sets <= 3:
		venn_sets = all_sets
		set_items = list(venn_sets.items())
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
	elif total_sets >= 4:
		upset_sets = all_sets
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

	# OPTIMIZED HPO PROCESSING - Use parallel fetching
	hpo_details = []
	if all_hpo_terms:
		hpo_details = fetch_hpo_terms_parallel(all_hpo_terms)

	hpo_table_component = html.Div()
	if hpo_details:
		hpo_table_component = create_hpo_terms_table(hpo_details)

	confidence_levels_present = sorted(df_unique["Confidence"].unique(), reverse=True)

	# Create buttons with proper IDs for the new callback system
	buttons = []
	for level in confidence_levels_present:
		button = dbc.Button(
			f"Gene list (confidence {level})", 
			id={"type": "btn-confidence", "level": str(level)}, 
			color="secondary", 
			className="me-2", 
			n_clicks=0
		)
		buttons.append(button)

	# Add manual genes button if present
	if manual_genes_list:
		manual_button = dbc.Button(
			"Manual Genes", 
			id={"type": "btn-confidence", "level": "Manual"}, 
			color="secondary", 
			className="me-2", 
			n_clicks=0
		)
		buttons.append(manual_button)

	# Define enhanced table columns
	table_columns = [
		{"name": "Gene Symbol", "id": "Gene Symbol", "type": "text"},
		{"name": "OMIM ID", "id": "OMIM ID", "type": "text", "presentation": "markdown"},
		{"name": "HGNC ID", "id": "HGNC ID", "type": "text", "presentation": "markdown"},
		{"name": "Type", "id": "Type", "type": "text"},
		{"name": "Biotype", "id": "Biotype", "type": "text"},
		{"name": "Mode of Inheritance", "id": "Mode of Inheritance", "type": "text"},
		{"name": "Confidence", "id": "Confidence", "type": "numeric"}
	]

	tables_by_level = {
		str(level): dash_table.DataTable(
			columns=table_columns,
			data=df_unique[df_unique["Confidence"] == level].to_dict("records"),
			style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"},
			style_cell={
				"textAlign": "left", 
				"padding": "6px",
				"fontSize": "11px",
				"fontFamily": "Arial, sans-serif",
				"whiteSpace": "normal",
				"height": "auto"
			},
			style_header={
				"fontWeight": "bold",
				"backgroundColor": "#f8f9fa",
				"border": "1px solid #ddd",
				"fontSize": "12px"
			},
			style_data_conditional=[
				{"if": {"filter_query": "{Confidence} = 3", "column_id": "Confidence"}, "backgroundColor": "#d4edda"},
				{"if": {"filter_query": "{Confidence} = 2", "column_id": "Confidence"}, "backgroundColor": "#fff3cd"},
				{"if": {"filter_query": "{Confidence} = 1", "column_id": "Confidence"}, "backgroundColor": "#f8d7da"},
				{"if": {"filter_query": "{Confidence} = 0", "column_id": "Confidence"}, "backgroundColor": "#d1ecf1"},
			],
			style_cell_conditional=[
				{"if": {"column_id": "Gene Symbol"}, "width": "100px", "minWidth": "100px"},
				{"if": {"column_id": "OMIM ID"}, "width": "150px", "minWidth": "150px"},
				{"if": {"column_id": "HGNC ID"}, "width": "110px", "minWidth": "110px"},
				{"if": {"column_id": "Type"}, "width": "70px", "minWidth": "70px"},
				{"if": {"column_id": "Biotype"}, "width": "110px", "minWidth": "110px"},
				{"if": {"column_id": "Mode of Inheritance"}, "width": "180px", "minWidth": "180px"},
				{"if": {"column_id": "Confidence"}, "width": "80px", "minWidth": "80px"},
			],
			page_action="none",
			markdown_options={"link_target": "_blank"}
		)
		for level in confidence_levels_present
	}

	# Handle manual genes table
	if manual_genes_list:
		manual_table_data = []
		for gene in manual_genes_list:
			manual_table_data.append({
				"Gene Symbol": gene,
				"OMIM ID": "",
				"HGNC ID": "",
				"Type": "manual",
				"Biotype": "manual",
				"Mode of Inheritance": "manual",
				"Confidence": 0
			})
		
		tables_by_level["Manual"] = dash_table.DataTable(
			columns=table_columns,
			data=manual_table_data,
			style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"},
			style_cell={
				"textAlign": "left", 
				"padding": "6px",
				"fontSize": "11px",
				"fontFamily": "Arial, sans-serif",
				"whiteSpace": "normal",
				"height": "auto"
			},
			style_header={
				"fontWeight": "bold",
				"backgroundColor": "#f8f9fa",
				"border": "1px solid #ddd",
				"fontSize": "12px"
			},
			style_data_conditional=[
				{"if": {"filter_query": "{Confidence} = 0", "column_id": "Confidence"}, "backgroundColor": "#d1ecf1"}
			],
			style_cell_conditional=[
				{"if": {"column_id": "Gene Symbol"}, "width": "100px", "minWidth": "100px"},
				{"if": {"column_id": "OMIM ID"}, "width": "150px", "minWidth": "150px"},
				{"if": {"column_id": "HGNC ID"}, "width": "110px", "minWidth": "110px"},
				{"if": {"column_id": "Type"}, "width": "70px", "minWidth": "70px"},
				{"if": {"column_id": "Biotype"}, "width": "110px", "minWidth": "110px"},
				{"if": {"column_id": "Mode of Inheritance"}, "width": "180px", "minWidth": "180px"},
				{"if": {"column_id": "Confidence"}, "width": "80px", "minWidth": "80px"},
			],
			page_action="none",
			markdown_options={"link_target": "_blank"}
		)

	table_output = html.Div(id="table-per-confidence")

	# SIMPLIFIED summary layout with search bar moved below the summary table (basic functionality only)
	summary_layout = html.Div([
		# Summary table first
		html.Div(summary_table, id="summary-table-content", style={"marginBottom": "20px"}),
		
		# Simple search bar below the summary table
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
		], style={"marginTop": "10px"})
	])

	print(f"Total processing time: {time.time() - start_time:.2f} seconds")

	return (summary_layout, 
			html.Div([
				html.Div(buttons, className="mb-3", style={"textAlign": "center"}),
				table_output
			]), 
			venn_component, 
			hpo_table_component,  
			venn_hpo_style,       
			html.Div(), 
			pie_style, 
			df_unique["Gene Symbol"].tolist(),
			all_hpo_terms,       
			updated_hpo_options,
			"",  # Clear unique code
			"",  # Clear panel summary
			tables_by_level)

# =============================================================================
# CALLBACKS - SIMPLIFIED GENE SEARCH (BASIC FUNCTIONALITY ONLY)
# =============================================================================

@app.callback(
	Output("gene-check-result", "children"),
	Output("gene-check-input", "value"),
	Input("gene-check-btn", "n_clicks"),
	Input("gene-check-input", "n_submit"),
	State("gene-check-input", "value"),
	State("gene-list-store", "data"),
	prevent_initial_call=True
)
def check_gene_in_panel(n_clicks, n_submit, gene_name, gene_list):
	if not gene_name or not gene_list:
		return "", ""
	
	if gene_name.upper() in [g.upper() for g in gene_list]:
		return f"Gene '{gene_name}' is present in the custom panel.", ""
	else:
		return f"Gene '{gene_name}' is NOT present in the custom panel.", ""

# =============================================================================
# CALLBACKS - SIMPLE TABLE INTERACTION
# =============================================================================

@app.callback(
	Output("table-per-confidence", "children"),
	Input({"type": "btn-confidence", "level": ALL}, "n_clicks"),
	State("gene-data-store", "data"),
	prevent_initial_call=True
)
def update_table_by_confidence(btn_clicks, data):
	ctx = dash.callback_context
	if not ctx.triggered:
		return ""
	
	# Check if any button was actually clicked (not just initialized)
	if all(n == 0 for n in btn_clicks):
		return ""
	
	triggered = ctx.triggered[0]["prop_id"].split(".")[0]
	triggered_dict = json.loads(triggered)
	level = triggered_dict["level"]
	
	return data.get(level, "")

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
			
			// Clear the notification after 6 seconds
			setTimeout(function() {
				notification.textContent = '';
			}, 6000);
		}
	}
	""",
	Output("panel-summary-output", "id"),  # Dummy output since we don't need to update anything
	Input("panel-summary-output", "value")
)

# =============================================================================
# APP RUN
# =============================================================================

# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == '__main__':
	port = int(os.environ.get("PORT", 8050))
	app.run(host="0.0.0.0", port=port)

