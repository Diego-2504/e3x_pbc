import numpy as np
from ase import Atoms
from ase.io import read
from ase.io import write


n_ds = int(input('Dame el n° de datasets \n'))
graphene_dynamics = []


for j in range(n_ds):
    var = int(input('Dame el n° de datasets por lattice \n'))
    # Dos outs por paso de dilatación
    if var == 2:
        for i in range(2):
            a = input('Nombre del dataset \n')
            b = input('Nombre del archivo de geometría \n')
            atoms = read(b)
            dataset = np.load(a)
            cell = atoms.cell
            positions, forces, energy = dataset['R'], dataset['F'], dataset['E']
            for t in range(len(positions)):
                R = positions[t]
                # Haciendo un objeto atoms de ase con los datos
                P_DM_graph = Atoms('C98', positions=R, cell=cell, pbc=[True, True, False])
                # Asignando
                P_DM_graph.arrays['forces'] = forces[t]
                P_DM_graph.info['energy'] = energy[t]
                graphene_dynamics.append(P_DM_graph)
    # Un out por dilatación
    elif var == 1:
        a = input('Nombre del dataset \n')
        b = input('Nombre del archivo de geometría \n')
        atoms = read(b)
        dataset = np.load(a)
        cell = atoms.cell
        positions, forces, energy = dataset['R'], dataset['F'], dataset['E']
        for t in range(len(positions)):
            R = positions[t]
            # Haciendo un objeto atoms de ase con los datos
            P_DM_graph = Atoms('C98', positions=R, cell=cell, pbc=[True, True, False])
            # Asignando
            P_DM_graph.arrays['forces'] = forces[t]
            P_DM_graph.info['energy'] = energy[t]
            graphene_dynamics.append(P_DM_graph)
    else:
        print('bai')

write('dm_graphene_prueba3.xyz', graphene_dynamics, format='extxyz')
