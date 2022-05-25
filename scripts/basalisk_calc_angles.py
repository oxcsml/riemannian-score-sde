#
#
import sys
from Bio.PDB.vectors import Vector, calc_dihedral
from numpy import mat, array
from math import sin, cos, sqrt, pi

#
# defines the different atoms needed to calculate the chi angles
# for all the different residue types ... well at least the common
# 20 ones
chi_atoms = {
    "ALA": {},
    "ARG": {
        "x1": ["n", "ca", "cb", "cg"],
        "x2": ["ca", "cb", "cg", "cd"],
        "x3": ["cb", "cg", "cd", "ne"],
        "x4": ["cg", "cd", "ne", "cz"],
    },
    "ASN": {"x1": ["n", "ca", "cb", "cg"], "x2": ["ca", "cb", "cg", "od1"]},
    "ASP": {"x1": ["n", "ca", "cb", "cg"], "x2": ["ca", "cb", "cg", "od1"]},
    "CYS": {"x1": ["n", "ca", "cb", "sg"]},
    "GLU": {
        "x1": ["n", "ca", "cb", "cg"],
        "x2": ["ca", "cb", "cg", "cd"],
        "x3": ["cb", "cg", "cd", "oe1"],
    },
    "GLN": {
        "x1": ["n", "ca", "cb", "cg"],
        "x2": ["ca", "cb", "cg", "cd"],
        "x3": ["cb", "cg", "cd", "oe1"],
    },
    "GLY": {},
    "HIS": {"x1": ["n", "ca", "cb", "cg"], "x2": ["ca", "cb", "cg", "cd2"]},
    "ILE": {"x1": ["n", "ca", "cb", "cg1"], "x2": ["ca", "cb", "cg1", "cd1"]},
    "LEU": {"x1": ["n", "ca", "cb", "cg"], "x2": ["ca", "cb", "cg", "cd1"]},
    "LYS": {
        "x1": ["n", "ca", "cb", "cg"],
        "x2": ["ca", "cb", "cg", "cd"],
        "x3": ["cb", "cg", "cd", "ce"],
        "x4": ["cg", "cd", "ce", "nz"],
    },
    "MET": {
        "x1": ["n", "ca", "cb", "cg"],
        "x2": ["ca", "cb", "cg", "sd"],
        "x3": ["cb", "cg", "sd", "ce"],
    },
    "PHE": {"x1": ["n", "ca", "cb", "cg"], "x2": ["ca", "cb", "cg", "cd1"]},
    "PRO": {
        "x1": ["n", "ca", "cb", "cg"],
        "x2": ["ca", "cb", "cg", "cd"],
        "x3": ["cb", "cg", "cd", "n"],
        "x4": ["cg", "cd", "n", "ca"],
    },
    "SER": {"x1": ["n", "ca", "cb", "og"]},
    "THR": {"x1": ["n", "ca", "cb", "og1"]},
    "TRP": {"x1": ["n", "ca", "cb", "cg"], "x2": ["ca", "cb", "cg", "cd1"]},
    "TYR": {"x1": ["n", "ca", "cb", "cg"], "x2": ["ca", "cb", "cg", "cd1"]},
    "VAL": {"x1": ["n", "ca", "cb", "cg1"]},
}

#
# defines the atomnames which will be considered backbone atoms.
backbone = ["n", "ca", "c", "o"]

#
# defines the expect sidechain atoms for the aminoacids.
sidechain = {
    "ALA": ["cb"],
    "ARG": ["cb", "cg", "cd", "ne", "cz", "nh1", "nh2"],
    "ASN": ["cb", "cg", "od1", "nd2"],
    "ASP": ["cb", "cg", "od1", "od2"],
    "CYS": ["cb", "sg"],
    "GLU": ["cb", "cg", "cd", "oe1", "oe2"],
    "GLN": ["cb", "cg", "cd", "oe1", "ne2"],
    "GLY": [],
    "HIS": ["cb", "cg", "nd1", "cd2", "ce1", "ne2"],
    "ILE": ["cb", "cg1", "cg2", "cd1"],
    "LEU": ["cb", "cg", "cd1", "cd2"],
    "LYS": ["cb", "cg", "cd", "ce", "nz"],
    "MET": ["cb", "cg", "sd", "ce"],
    "PHE": ["cb", "cg", "cd1", "cd2", "ce1", "ce2", "cz"],
    "PRO": ["cb", "cg", "cd"],
    "SER": ["cb", "og"],
    "THR": ["cb", "og1", "cg2"],
    "TRP": ["cb", "cg", "cd1", "cd2", "ne1", "ce2", "ce3", "cz2", "cz3", "ch2"],
    "TYR": ["cb", "cg", "cd1", "cd2", "ce1", "ce2", "cz", "oh"],
    "VAL": ["cb", "cg1", "cg2"],
}


def get_atom(residue, atomname):
    """
    Given a residue and an atomname, this method sequentially
    goes through the list and tries to find an atomname with
    the given name ... on success this atom will be returned,
    otherwise a trigger sequence will be returned.

    @param residue: an Bio.PDB.Residue
    @type residue: Bio.PDB.Residue

    @param atomname: an atomname
    @type atomname: String

    @return: the atom found or a trigger sequence indicating the error state
    @rtype: Bio.PDB.Atom if found, else String

    """
    atoms = residue.get_list()
    for atom in atoms:
        if atom.get_name().lower() == atomname.lower():
            return atom
    #
    return "@@"


def get_chi(residuetype):
    """
    For each residuetype there is a certain set of dihedral angles in the sidechain
    which again are defined by a certain set of atoms each.

    @param residuetype: String describing the residuetype
    @type residuetype: String

    @return: returns a dictionary of arrays, describing the atoms for all the angles
    @rtype: dictionary
    """
    if residuetype.upper() not in chi_atoms:
        sys.stderr.write(
            "Warning: Unknown residuetype "
            + residuetype
            + " in CalcAngles.get_chis() \n"
        )
        return {}
    #
    return chi_atoms[residuetype.upper()]


def calc_chi_angles(residue, silent=False):
    """
    This method calculates all the important dihedral angles
    in the residues sidechain.
    As every residue type of course has a different sidechain,
    we here use a given, predefined list of dihedral angles ...
    The list is defined as  chi_atoms dictionary.
    """
    # get a list of all the chi angles including all the
    # therefore necessary atoms for this residue type
    angles = get_chi(residue.get_resname().upper())
    # init the result set
    chi = {"x1": -5.0, "x2": -5.0, "x3": -5.0, "x4": -5.0}
    #
    for x in angles:
        # this could be done a lot more compact, but thought we keep
        # it legible here ... and besides, it allows the catch the
        # error state of not having all atoms at hand ..
        #
        # get the four atoms necessary to calculate the dihedral
        xa1 = get_atom(residue, angles[x][0])
        xa2 = get_atom(residue, angles[x][1])
        xa3 = get_atom(residue, angles[x][2])
        xa4 = get_atom(residue, angles[x][3])
        #
        # catch the error state ..
        if xa1 == "@@" or xa2 == "@@" or xa3 == "@@" or xa4 == "@@":
            if not silent:
                sys.stderr.write(
                    "Warning: Not enough Atoms found for "
                    + x
                    + " in "
                    + residue.get_resname()
                    + " \n"
                )
            continue
        # get the atoms positions as point vectors
        xv1 = Vector(xa1.get_coord())
        xv2 = Vector(xa2.get_coord())
        xv3 = Vector(xa3.get_coord())
        xv4 = Vector(xa4.get_coord())
        #
        # calculate the dihedral
        chi[x] = calc_dihedral(xv1, xv2, xv3, xv4)
    #
    # finally save the results dict in the residues
    # xtra compartment ...
    residue.xtra["chi"] = chi


def calc_phi_psi(residue, predecessor=0):
    """
    This method calculates the PSI and PHI angles
    for the proteins backbone.
    """
    residue.xtra["psi"] = -5.0
    residue.xtra["phi"] = -5.0

    if not predecessor:
        # there is nothing to be done here.
        return

    # Again I'll keep it a little more stretched than
    # necessary here.
    n = Vector(get_atom(residue, "n").get_coord())
    ca = Vector(get_atom(residue, "ca").get_coord())
    c = Vector(get_atom(residue, "c").get_coord())
    n_pd = Vector(get_atom(predecessor, "n").get_coord())
    ca_pd = Vector(get_atom(predecessor, "ca").get_coord())
    c_pd = Vector(get_atom(predecessor, "c").get_coord())

    # lets start with calculating the missing angles of
    # previous residue
    psi = calc_dihedral(n_pd, ca_pd, c_pd, n)

    # and then the phi angle for the latest residue
    phi = calc_dihedral(c_pd, n, ca, c)

    # and then finally store the result in the residue object again
    predecessor.xtra["psi"] = psi
    residue.xtra["phi"] = phi
