import csv
import argparse
from collections import defaultdict

def find_isomers(file_path):
    formula_smiles = defaultdict(set)

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            
            if len(parts) < 2:
                print(f"Skipping malformed line: {line}")
                continue
            
            formula = parts[0].split()[0].strip()  
            smiles = parts[1].strip()  
            
            formula_smiles[formula].add(smiles)

    isomers = {formula: smiles for formula, smiles in formula_smiles.items() if len(smiles) > 1}

    print(f"Total unique isomers: {len(isomers)}")
    for formula, smiles_set in isomers.items():
        print(f"Formula: {formula}")
        print(f"SMILES: {', '.join(smiles_set)}")
        print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find isomers from a SMILES CSV file.")
    parser.add_argument('input_file', type=str, help="Path to the input CSV file.")

    args = parser.parse_args()

    find_isomers(args.input_file)
