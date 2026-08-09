"""
verify_parsing.py
=================
Script to verify LLM parser output against PTT content
"""
import pandas as pd
import json
from llm_ptt_parser import parse_ptt_items, get_openai_client

def verify_machine_rows(csv_path, machine_name, max_rows=3):
    """Parse and display LLM output for specific machine rows"""
    df = pd.read_csv(csv_path)
    machine_rows = df[df['Machine'].str.lower() == machine_name.lower()]
    
    print(f"\n=== Verifying {machine_name} ({len(machine_rows)} rows) ===\n")
    
    client = get_openai_client()
    
    for idx, (_, row) in enumerate(machine_rows.head(max_rows).iterrows()):
        print(f"\n--- Row {idx} (CSV index {row.name}) ---")
        print(f"PTT:\n{row['PTT'][:500]}...")
        
        items, source = parse_ptt_items(machine_name, row['PTT'], client, row_index=idx)
        
        print(f"\nParsed items (source: {source}):")
        for item in items:
            print(f"  [{item['number']}] {item['title']} | Type: {item['node_type']} | Status: {item['status']}")
            if item['finding']:
                print(f"    Finding: {item['finding'][:100]}...")

def verify_multiple_machines(csv_path, sample_machines=None, max_rows_per_machine=2):
    """Parse and display LLM output for multiple machines"""
    df = pd.read_csv(csv_path)
    
    if sample_machines is None:
        # Sample diverse machines
        unique_machines = df['Machine'].unique()
        sample_machines = list(unique_machines[:10])  # First 10 machines
    
    print(f"\n=== Verifying {len(sample_machines)} machines from {csv_path} ===\n")
    
    client = get_openai_client()
    
    for machine_name in sample_machines:
        machine_rows = df[df['Machine'] == machine_name]
        if len(machine_rows) == 0:
            continue
            
        print(f"\n{'='*60}")
        print(f"Machine: {machine_name} ({len(machine_rows)} rows)")
        print('='*60)
        
        for idx, (_, row) in enumerate(machine_rows.head(max_rows_per_machine).iterrows()):
            print(f"\n--- Row {idx} (CSV index {row.name}) ---")
            ptt_preview = row['PTT'][:300]
            print(f"PTT preview: {ptt_preview}...")
            
            items, source = parse_ptt_items(machine_name, row['PTT'], client, row_index=idx)
            
            print(f"Parsed {len(items)} items (source: {source}):")
            for item in items:
                print(f"  [{item['number']}] {item['title'][:40]} | Type: {item['node_type']} | Status: {item['status']}")
                if item['finding']:
                    print(f"    Finding: {item['finding'][:60]}...")

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_data.csv"
    
    if len(sys.argv) > 2:
        # Specific machine
        machine_name = sys.argv[2]
        verify_machine_rows(csv_path, machine_name)
    else:
        # Sample multiple machines
        verify_multiple_machines(csv_path)
