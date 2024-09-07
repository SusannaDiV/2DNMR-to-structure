from rdkit import Chem

def canonicalize_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    else:
        return "Invalid SMILES"  

def categorize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid"
        
        # 1. Very Easy
        if mol.GetNumAtoms() <= 3:
            return "Very Easy"

        # 2. Easy
        if mol.GetNumAtoms() <= 10 and not mol.GetRingInfo().IsAtomInRingOfSize(0, 4) and not Chem.FindMolChiralCenters(mol, includeUnassigned=True):
            return "Easy"
        
        # 3. Moderate
        if (Chem.FindMolChiralCenters(mol, includeUnassigned=True) or
            mol.GetRingInfo().NumRings() > 0 or
            any(atom.GetDegree() > 3 for atom in mol.GetAtoms()) or
            Chem.MolToSmiles(mol).find('B') > -1 or
            Chem.MolToSmiles(mol).find('Fe') > -1 or
            mol.GetNumAtoms() <= 20):
            return "Moderate"

        # 4. Hard
        if (is_fused_ring(mol) or
            len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6]]) > 1 or
            mol.GetNumRotatableBonds() > 3 or
            mol.GetNumAtoms() <= 50):
            return "Hard"
        
        # 5. Very Hard
        if (mol.GetNumAtoms() > 50 or
            len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)) > 1 or
            any(atom.GetIsotope() != 0 for atom in mol.GetAtoms())):
            return "Very Hard"

        return "Uncategorized"
    
    except Exception as e:
        print(f"Error processing SMILES: {smiles}, Error: {str(e)}")
        return "Invalid"

def is_fused_ring(mol):
    ring_info = mol.GetRingInfo()
    atom_ring_count = [0] * mol.GetNumAtoms()
    for ring in ring_info.AtomRings():
        for atom_idx in ring:
            atom_ring_count[atom_idx] += 1
    
    return any(count > 1 for count in atom_ring_count)

def process_smiles_file(file_path):
    with open(file_path, 'r') as file:
        smiles_list = file.readlines()

    results = []
    for smiles in smiles_list:
        smiles = canonicalize_smiles(smiles.strip())
        category = categorize_smiles(smiles)
        results.append((smiles, category))

    return results

def write_results(results, output_path):
    with open(output_path, 'w') as file:
        for smiles, category in results:
            file.write(f"{smiles}\t{category}\n")

if __name__ == "__main__":
    input_file = "/itet-stor/sdivita/net_scratch/nmr3/finale/data/train_smiles.tgt"  
    output_file = "/itet-stor/sdivita/net_scratch/nmr3/finale/data/train_smilesdifficulty.tgt" 

    results = process_smiles_file(input_file)
    write_results(results, output_file)
