# utils/panelapp_api.py

import requests
import pandas as pd

PANELAPP_UK_BASE = "https://panelapp.genomicsengland.co.uk/api/v1/"
PANELAPP_AU_BASE = "https://panelapp-aus.org/api/v1/"


def fetch_panels(base_url):
    """Fetch list of panels from a PanelApp instance (UK or Australia)"""
    panels = []
    url = f"{base_url}panels/"
    
    while url:
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch panels from {url}")
        data = response.json()
        panels.extend(data.get('results', []))
        url = data.get('next')  # For pagination
    
    return pd.DataFrame(panels)

def fetch_panel_genes(base_url, panel_id):
    """Fetch gene list for a specific panel ID with detailed gene information"""
    url = f"{base_url}panels/{panel_id}/"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch panel genes from {url}")
    
    panel_data = response.json()
    genes = panel_data.get("genes", [])
    
    def format_omim_links(omim_list):
        """Format OMIM IDs as clickable links"""
        if not omim_list:
            return ""
        
        links = []
        for omim_id in omim_list:
            if omim_id:
                links.append(f'[{omim_id}](https://omim.org/entry/{omim_id})')
        
        return " | ".join(links) if links else ""
    
    def format_hgnc_link(hgnc_id):
        """Format HGNC ID as clickable link"""
        if not hgnc_id:
            return ""
        
        return f'[{hgnc_id}](https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/{hgnc_id})'
    
    df_genes = pd.DataFrame([
        {
            "gene_symbol": g["gene_data"].get("gene_symbol", ""),
            "omim_id": format_omim_links(g["gene_data"].get("omim_gene", [])),
            "hgnc_id": format_hgnc_link(g["gene_data"].get("hgnc_id", "")),
            "entity_type": g.get("entity_type", ""),
            "biotype": g["gene_data"].get("biotype", ""),
            "mode_of_inheritance": g.get("mode_of_inheritance", ""),
            "confidence_level": g.get("confidence_level"),
            "penetrance": g.get("penetrance"),
            "source": g.get("source"),
        }
        for g in genes
    ])
    
    # Return panel data which includes name, version, and other metadata
    panel_info = {
        "name": panel_data.get("name"),
        "version": panel_data.get("version"),
        "id": panel_data.get("id"),
        "status": panel_data.get("status"),
        "disease_group": panel_data.get("disease_group"),
        "disease_sub_group": panel_data.get("disease_sub_group")
    }
    
    return df_genes, panel_info
