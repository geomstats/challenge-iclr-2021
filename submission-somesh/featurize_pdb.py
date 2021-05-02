import os

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from frozendict import frozendict
import math
import mdtraj as md
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

elements_dict = frozendict({'H' : 1.008,'HE' : 4.003, 'LI' : 6.941, 'BE' : 9.012,\
                 'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
                 'F' : 18.998, 'NE' : 20.180, 'NA' : 22.990, 'MG' : 24.305,\
                 'AL' : 26.982, 'SI' : 28.086, 'P' : 30.974, 'S' : 32.066,\
                 'CL' : 35.453, 'AR' : 39.948, 'K' : 39.098, 'CA' : 40.078,\
                 'SC' : 44.956, 'TI' : 47.867, 'V' : 50.942, 'CR' : 51.996,\
                 'MN' : 54.938, 'FE' : 55.845, 'CO' : 58.933, 'NI' : 58.693,\
                 'CU' : 63.546, 'ZN' : 65.38, 'GA' : 69.723, 'GE' : 72.631,\
                 'AS' : 74.922, 'SE' : 78.971, 'BR' : 79.904, 'KR' : 84.798,\
                 'RB' : 84.468, 'SR' : 87.62, 'Y' : 88.906, 'ZR' : 91.224,\
                 'NB' : 92.906, 'MO' : 95.95, 'TC' : 98.907, 'RU' : 101.07,\
                 'RH' : 102.906, 'PD' : 106.42, 'AG' : 107.868, 'CD' : 112.414,\
                 'IN' : 114.818, 'SN' : 118.711, 'SB' : 121.760, 'TE' : 126.7,\
                 'I' : 126.904, 'XE' : 131.294, 'CS' : 132.905, 'BA' : 137.328,\
                 'LA' : 138.905, 'CE' : 140.116, 'PR' : 140.908, 'ND' : 144.243,\
                 'PM' : 144.913, 'SM' : 150.36, 'EU' : 151.964, 'GD' : 157.25,\
                 'TB' : 158.925, 'DY': 162.500, 'HO' : 164.930, 'ER' : 167.259,\
                 'TM' : 168.934, 'YB' : 173.055, 'LU' : 174.967, 'HF' : 178.49,\
                 'TA' : 180.948, 'W' : 183.84, 'RE' : 186.207, 'OS' : 190.23,\
                 'IR' : 192.217, 'PT' : 195.085, 'AU' : 196.967, 'HG' : 200.592,\
                 'TL' : 204.383, 'PB' : 207.2, 'BI' : 208.980, 'PO' : 208.982,\
                 'AT' : 209.987, 'RN' : 222.081, 'FR' : 223.020, 'RA' : 226.025,\
                 'AC' : 227.028, 'TH' : 232.038, 'PA' : 231.036, 'U' : 238.029,\
                 'NP' : 237, 'PU' : 244, 'AM' : 243, 'CM' : 247, 'BK' : 247,\
                 'CT' : 251, 'ES' : 252, 'FM' : 257, 'MD' : 258, 'NO' : 259,\
                 'LR' : 262, 'RF' : 261, 'DB' : 262, 'SG' : 266, 'BH' : 264,\
                 'HS' : 269, 'MT' : 268, 'DS' : 271, 'RG' : 272, 'CN' : 285,\
                 'NH' : 284, 'FL' : 289, 'MC' : 288, 'LV' : 292, 'TS' : 294,\
                 'OG' : 294})

letters = frozendict({'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H',
           'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
           'TYR':'Y','VAL':'V'})


class Featurize_PDB:
    def __init__(self):
        print('Featurize PDB as Coulomb Matrix, Point Cloud, and Protein Parameters')
    

    def _coulomb_matrix(self, rdkit_mol):
        """Featurizes RDKit mol as Coulomb matrix.

        Args: 
            rdkit_mol: RDKit mol, receptor/peptide mol object

        Returns:
            np array, (n_atoms, n_atoms) Coulomb matrix

        """
        return np.array(rdMolDescriptors.CalcCoulombMat(rdkit_mol))

    def _point_cloud(self, mdtraj_obj, mass):
        """Featurizes MDTraj object as array of coordinates.

        Args: 
            mdtraj_obj: MDTraj object, receptor/peptide MDTraj objet
            mass: bool, condition to add atomic mass.

        Returns:
            np array: (n_atoms, n_features), if mass=False, 3 Cartesian coordinates;
            elif mass = True, 3 Cartesian coordinates and mass.


        """    
        if mass is False:
            return mdtraj_obj.xyz[0]
        
        elif mass is True:
            mass = np.array([
                elements_dict[element] for element in 
                mdtraj_obj.topology.to_dataframe()[0]['element']])
            return np.hstack([mdtraj_obj.xyz[0], mass.reshape(-1, 1)])
    
    def _pdb_to_sequence(self, path_pdb):
        """Converts PDB to amino acid sequence.
        
        Args:
            path_pdb: str, path of PDB file.
            
        Returns:
            sequence: str, amino acid sequence.
            
        """
        input_file = open(path_pdb)
        
        sequence = ''
        prev = '-1'
        for line in input_file:
            toks = line.split()
            if len(toks)<1: continue
            if toks[0] != 'ATOM': continue
            if toks[5] != prev:
                sequence = ''.join((sequence, letters[toks[3]]))
            prev = toks[5]
        return sequence
    
    
    def _net_charge(self, sequence):
        """Calculates net charge of the amino acid sequence.
        
        Reference: http://www.chem.ucalgary.ca/courses/351/Carey5th/Ch27/ch27-1-4-2.html
        
        Args:
            sequence: str, amino acid sequence.
            
        Returns: 
            charge: float, net charge of the sequence.
            
        """
        
        acidic = [sequence.count('D'), sequence.count('E'), sequence.count('C'), sequence.count('Y')]
        basic = [sequence.count('R'), sequence.count('K'), sequence.count('H')]

        acidic_pKa = [math.pow(10, 3.65), math.pow(10, 4.25), math.pow(10, 8.37), math.pow(10, 10.46)]
        basic_pKa = [math.pow(10, 10.76), math.pow(10, 9.74), math.pow(10, 7.59)]

        basic_coeff = [x*(1/(x+math.pow(10, 7))) for x in basic_pKa]
        acidic_coeff = [math.pow(10, 7)/(x+math.pow(10, 7)) for x in acidic_pKa]

        charge = - sum(np.multiply(acidic_coeff, acidic)) + sum(np.multiply(basic_coeff, basic))
        
        return charge


    def _protein_parameters(self, sequence):
        """Calculates physicochemical properties for the amino acid sequence.
        
        Args:
            sequence: str, amino acid sequence.
            
        Returns: 
            property_arr: np array, vector of properties.
            
        """
        
        analysis = ProteinAnalysis(sequence)

        property_arr = []

        property_arr.append(analysis.molecular_weight())
        property_arr.append(analysis.aromaticity())
        property_arr.append(analysis.instability_index())
        property_arr.append(analysis.gravy())
        property_arr.append(analysis.isoelectric_point())

        secondary = analysis.secondary_structure_fraction()
        property_arr.append(secondary[0])
        property_arr.append(secondary[1])
        property_arr.append(secondary[2])
        
        molar_extinction_coefficient = analysis.molar_extinction_coefficient()
        property_arr.append(molar_extinction_coefficient[0])
        property_arr.append(molar_extinction_coefficient[1])

        property_arr.append(self._net_charge(sequence))
        
        return np.array(property_arr)

    def featurize(self, path_folder, file_name, feature_type):
        """
        Loads PDB and featurizes as noted.
        
        Args:
            path_folder: str, PDB ID followed by chain ID, example: 1a61_L
            file_name: str, name of PDB file, example: receptor, peptide
            feature_type: str, custom feature for PDB, example: Coulomb, PC, PC-Mass
        
        Returns:
            pdb_feature: np array

        """
        path_pdb = os.path.join(path_folder, file_name)
        
        if feature_type == 'Coulomb':
            try:
                mol_obj = Chem.rdmolfiles.MolFromPDBFile(path_pdb)
            except:
                print('Invalid PDB file.')
            return self._coulomb_matrix(mol_obj)
        
        elif 'PC' in feature_type:
            try:
                md_obj = md.load_pdb(path_pdb)
                mass = False if 'Mass' not in feature_type else True
            except:
                print('Invalid PDB trajectory.')
            return self._point_cloud(md_obj, mass=mass)
                
        elif feature_type == 'Parameters':
            try:
                sequence = self._pdb_to_sequence(path_pdb)
            except:
                print('Invalid PDB file.')
            return self._protein_parameters(sequence)
        
        else:
            print('Invalid feature type')