import rdkit
from rdkit import Chem, DataStructs
import rdkit.Chem.QED as QED
from rdkit.Chem import AllChem

# my files
from dataset.DRD2.DRD2_predictor import DRD2


# disable rdkit error messages
def rdkit_no_error_print():
    rdkit.rdBase.DisableLog('rdApp.error')


# returns None if molecule_SMILES string does not represent legal molecule
def smiles2mol(molecule_SMILES):
    mol = Chem.MolFromSmiles(molecule_SMILES)
    if mol is None:
        raise Exception()
    return mol


# raise exception if molecule_SMILES string does not represent legal molecule,
# otherwise return its fingerprints
def smiles2fingerprint(molecule_SMILES, radius=2, nBits=2048, useChirality=False, fp_translator=False):
    try:
        mol = smiles2mol(molecule_SMILES)
        if fp_translator is True:
          fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useChirality=useChirality)
        else:
          fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useChirality=useChirality)
        return fp
    except Exception as e:
        raise Exception()


# initialize property calculation
def property_init(property):
    if property is 'DRD2':
        global DRD2_pred
        DRD2_pred = DRD2()


# raise exception if molecule_SMILES string does not represent legal molecule
def property_calc(molecule_SMILES, property):
    try:
        mol = smiles2mol(molecule_SMILES)
        if property is 'QED':
            return QED.qed(mol)
        if property is 'DRD2':
            return DRD2_pred.get_score(mol)
    except Exception as e:
        raise Exception()


# raise exception if at least one of the molecule_SMILES strings does not represent legal molecule
def similarity_calc(molecule_SMILES_1, molecule_SMILES_2):
    try:
        fp_mol_1 = smiles2fingerprint(molecule_SMILES_1)
        fp_mol_2 = smiles2fingerprint(molecule_SMILES_2)
        return DataStructs.TanimotoSimilarity(fp_mol_1, fp_mol_2)
    except Exception as e:
        raise Exception()


# canonicalize_smiles
def canonicalize_smiles(molecule_SMILES):
    try:
        molecule_canonical_SMILES = Chem.MolToSmiles(Chem.MolFromSmiles(molecule_SMILES), True)
        return molecule_canonical_SMILES
    except Exception as e:
        raise Exception()


# valid if 1) valid by rdkit. 2) the property can be calculated on that molecule.
def is_valid_molecule(molecule_SMILES, property):
    try:
        property_calc(molecule_SMILES, property)
        return True
    except:
        return False