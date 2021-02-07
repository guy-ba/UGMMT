import numpy as np
from rdkit.Chem import AllChem
import pickle
import os.path as op

"""Scores based on an ECFP classifier for activity."""

class DRD2(object):
    clf_model = None
    def __init__(self):
        name = op.join(op.dirname(__file__), 'clf_py36.pkl')
        with open(name, "rb") as f:
            self.clf_model = pickle.load(f)

    def get_score(self, mol):
        try:
            fp = self.fingerprints_from_mol(mol)
            score = self.clf_model.predict_proba(fp)[:, 1]
            return float(score)
        except Exception as e:
            raise Exception()

    def fingerprints_from_mol(self, mol):
        try:
            fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
            size = 2048
            nfp = np.zeros((1, size), np.int32)
            for idx,v in fp.GetNonzeroElements().items():
                nidx = idx%size
                nfp[0, nidx] += int(v)
            return nfp
        except Exception as e:
            raise Exception()
