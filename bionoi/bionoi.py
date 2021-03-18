from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import pandas as pd
import matplotlib
from biopandas.mol2 import PandasMol2
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.cluster import KMeans
from math import sqrt, asin, atan2, log, pi, tan
from alignment import align

import argparse
import os
import skimage
from skimage.io import imshow
import cv2
from skimage.transform import rotate as skrotate
from skimage import img_as_ubyte
import statistics


def k_different_colors(k: int):
    colors = dict(**mcolors.CSS4_COLORS)

    def rgb(color): return mcolors.to_rgba(color)[:3]
    def hsv(color): return mcolors.rgb_to_hsv(color)

    col_dict = [(k, rgb(k)) for c, k in colors.items()]
    X = np.array([j for i, j in col_dict])

    # Perform kmeans on rqb vectors
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(X)
    # Getting the cluster labels
    labels = kmeans.predict(X)
    # Centroid values
    C = kmeans.cluster_centers_

    # Find one color near each of the k cluster centers
    closest_colors = np.array([np.sum((X - C[i]) ** 2, axis=1)
                               for i in range(C.shape[0])])
    keys = sorted(closest_colors.argmin(axis=1))

    return [col_dict[i][0] for i in keys]


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    Source
    -------
    Copied from https://gist.github.com/pv/8036995
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def fig_to_numpy(fig, alpha=1) -> np.ndarray:
    """
    Converts matplotlib figure to a numpy array.

    Source
    ------
    Adapted from https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    """

    # Setup figure
    fig.patch.set_alpha(alpha)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def miller(x, y, z):
    radius = sqrt(x ** 2 + y ** 2 + z ** 2)
    latitude = asin(z / radius)
    longitude = atan2(y, x)
    lat = 5 / 4 * log(tan(pi / 4 + 2 / 5 * latitude))
    return lat, longitude


def alignment(pocket, proj_direction):
    """Principal Axes Alignment
    Returns transformation coordinates(matrix: X*3)"""

    pocket_coords = np.array([pocket.x, pocket.y, pocket.z]).T
    # calculate mean of each column
    pocket_center = np.mean(pocket_coords, axis=0)
    pocket_coords = pocket_coords - pocket_center  # Centralization
    # get covariance matrix (of centralized data)
    inertia = np.cov(pocket_coords.T)
    # linear algebra eigenvalue eigenvector
    e_values, e_vectors = np.linalg.eig(inertia)
    # sort eigenvalues (increase)and reverse (decrease)
    sorted_index = np.argsort(e_values)[::-1]
    sorted_vectors = e_vectors[:, sorted_index]

    transformation_matrix = align(sorted_vectors, proj_direction)
    transformed_coords = (np.matmul(transformation_matrix, pocket_coords.T)).T

    return transformed_coords


def voronoi_atoms(bs, color_map, colorby, bs_out=None, size=None, dpi=None, alpha=1, save_fig=True,
                  projection=miller, proj_direction=None):
    # Suppresses warning
    pd.options.mode.chained_assignment = None

    # Read molecules in mol2 format
    mol2 = PandasMol2().read_mol2(bs)
    atoms = mol2.df[['atom_id', 'subst_name', 'atom_type',
                     'atom_name', 'x', 'y', 'z', 'charge']]
    atoms.columns = ['atom_id', colorby_conv(
        colorby), 'atom_type', 'atom_name', 'x', 'y', 'z', 'relative_charge']
    atoms['atom_id'] = atoms['atom_id'].astype(str)
    if colorby in ["hydrophobicity", "binding_prob", "residue_type"]:
        atoms[colorby] = atoms[colorby].apply(lambda x: x[0:3])

    # Align to principal Axis
    # get the transformation coordinate
    trans_coords = alignment(atoms, proj_direction)
    atoms['x'] = trans_coords[:, 0]
    atoms['y'] = trans_coords[:, 1]
    atoms['z'] = trans_coords[:, 2]

    # convert 3D  to 2D
    atoms["P(x)"] = atoms[['x', 'y', 'z']].apply(
        lambda coord: projection(coord.x, coord.y, coord.z)[0], axis=1)
    atoms["P(y)"] = atoms[['x', 'y', 'z']].apply(
        lambda coord: projection(coord.x, coord.y, coord.z)[1], axis=1)

    # setting output image size, labels off, set 120 dpi w x h
    size = 128 if size is None else size
    dpi = 120 if dpi is None else dpi

    figure = plt.figure(figsize=(int(size) / int(dpi),
                                 int(size) / int(dpi)), dpi=int(dpi))

    # figsize is in inches, dpi is the resolution of the figure
    ax = plt.subplot(111)  # default is (111)

    ax.axis('off')
    ax.tick_params(axis='both', bottom=False, left=False, right=False,
                   labelleft=False, labeltop=False,
                   labelright=False, labelbottom=False)

    # Compute Voronoi tesselation
    vor = Voronoi(atoms[['P(x)', 'P(y)']])
    regions, vertices = voronoi_finite_polygons_2d(vor)
    polygons = []
    for reg in regions:
        polygon = vertices[reg]
        polygons.append(polygon)
    atoms.loc[:, 'polygons'] = polygons

    # Check alpha
    alpha = float(alpha)

    # Colors color_map
    if colorby in ["charge", "center_dist", "sasa"]:
        colors = [color_map[_type]["color"] for _type in atoms['atom_id']]
    else:
        colors = [color_map[_type]["color"] for _type in atoms[colorby]]
    atoms["color"] = colors

    for i, row in atoms.iterrows():
        colored_cell = matplotlib.patches.Polygon(row["polygons"],
                                                  facecolor=row['color'],
                                                  edgecolor=row['color'],
                                                  alpha=alpha,
                                                  linewidth=0.2)
        ax.add_patch(colored_cell)

    # Set limits
    ax.set_xlim(vor.min_bound[0], vor.max_bound[0])
    ax.set_ylim(vor.min_bound[1], vor.max_bound[1])

    # Output image saving in any format; default jpg
    bs_out = 'out.jpg' if bs_out is None else bs_out

    # Get image as numpy array
    figure.tight_layout(pad=0)
    img = fig_to_numpy(figure, alpha=alpha)

    if save_fig:
        plt.subplots_adjust(bottom=0, top=1, left=0, right=1)
        plt.savefig(bs_out, frameon=False, pad_inches=False)

    plt.close(fig=figure)

    return atoms, vor, img


def colorby_conv(colorby):
    """change the colorby parameter to match the dictionary
    data keys and the column of the atoms dataframe"""

    if colorby in ["atom_type", "charge", "center_dist"]:
        color_by = "residue_type"
    else:
        color_by = colorby
    return color_by


def custom_colormap(color_scale):
    """takes two hex colors and creates a linear colormap"""

    color_dict = {"red_cyan": ("#ff0000", "#00ffff"), "orange_bluecyan": ("#ff7f00", "#007fff"),
                  "yellow_blue": ("#ffff00", "#0000ff"), "greenyellow_bluemagenta": ("#7fff00", "#7f00ff"),
                  "green_magenta": ("#00ff00", "#ff00ff"), "greencyan_redmagenta": ("#00ff7f", "#ff007f")}

    try:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'cmap1', color_dict[color_scale], N=256)

    except:
        cmap = None     # used for residue_type and atom_type because of predetermined color maps

    return cmap


def normalizer(dataset, colorby):
    """normalizes dataset using min and max values"""

    valnorm_lst = []
    if colorby not in ["atom_type", "residue_type"]:
        for val in dataset.values():
            val = float(val)
            # used if all values in the set are the same
            if max(dataset.values()) == min(dataset.values()):
                valnorm = 0.0
            else:
                valnorm = ((val-min(dataset.values())) /
                           (max(dataset.values())-min(dataset.values())))
            valnorm_lst.append(valnorm)

    return valnorm_lst


def colorgen(colorby, valnorm_lst, cmap, dataset):
    """creates a new dictionary that contains the color of the given key"""

    # atom type and residue type colors are predetermined
    if colorby in ["atom_type", "residue_type"]:
        color_map = "./cmaps/atom_cmap.csv" if colorby == "atom_type" else "./cmaps/res_hydro_cmap.csv"

        # Check for color mapping file, make dict
        with open(color_map, "rt") as color_mapF:
            # Parse color map file
            color_map = np.array(
                [line.replace("\n", "").split(";") for line in color_mapF.readlines() if not line.startswith("#")])
            # Create color dictionary
            color_map = {code: {"color": color, "definition": definition}
                         for code, definition, color in color_map}
            return color_map
    else:
        color_lst = []

        # Apply colormap to the normalized data
        for val in valnorm_lst:
            color = cmap(val)
            color = matplotlib.colors.rgb2hex(color)
            color_lst.append(color)

        # Create color dictionary
        color_map = dict(zip(dataset.keys(), color_lst))
        names = ['code', 'color']
        dtype = dict(names=names)
        hexcolor_array = np.asarray(list(color_map.items()))
        color_map = {code: {"color": color} for code, color in hexcolor_array}
        return color_map


def extract_charge_data(mol):
    """extracts and formats charge data from mol2 file"""

    # Extracting data from mol2
    # Suppress warning
    pd.options.mode.chained_assignment = None
    mol2 = PandasMol2().read_mol2(mol)
    # Only need atom_id and charge data
    atoms = mol2.df[['atom_id', 'charge']]
    atoms.columns = ['atom_id', 'charge']

    # Create dictionary
    charge_list = atoms['charge'].tolist()
    atomid_list = atoms['atom_id'].tolist()
    charge_data = dict(zip(atomid_list, charge_list))

    return charge_data


def extract_centerdistance_data(mol, proj_direction):
    """extracts and formats center distance from mol2 file
    after alignment to principal axes."""

    # Extracting data from mol2
    pd.options.mode.chained_assignment = None
    mol2 = PandasMol2().read_mol2(mol)
    atoms = mol2.df[['atom_id', 'x', 'y', 'z']]
    atoms.columns = ['atom_id', 'x', 'y', 'z']

    # Aligning to principal axes so that origin is the center
    # of pocket get the transformation coordinate
    trans_coords = alignment(atoms, proj_direction)
    atoms['x'] = trans_coords[:, 0]
    atoms['y'] = trans_coords[:, 1]
    atoms['z'] = trans_coords[:, 2]

    atomid_list = atoms['atom_id'].tolist()
    coordinate_list = atoms.values.tolist()

    # Calculating the distance to the center of the
    # pocket and creating dictionary
    center_dist_list = []
    for xyz in coordinate_list:
        center_dist = ((xyz[0]) ** 2 + (xyz[1]) ** 2 + (xyz[2]) ** 2) ** .5
        center_dist_list.append(center_dist)
    center_dist_data = dict(zip(atomid_list, center_dist_list))

    return center_dist_data


def extract_sasa_data(mol, pop):
    """extracts accessible surface area data from .out file generated by POPSlegacy.

        then matches the data in the .out file to the binding site in the mol2 file.

        Used POPSlegacy https://github.com/Fraternalilab/POPSlegacy"""

    # Extracting data from mol2 file
    pd.options.mode.chained_assignment = None
    mol2 = PandasMol2().read_mol2(mol)
    # only need subst_name for matching. Other data comes from .out file
    atoms = mol2.df[['subst_name']]
    atoms.columns = ['residue_type']
    siteresidue_list = atoms['residue_type'].tolist()

    # Extracting sasa data from .out file
    residue_list = []
    qsasa_list = []
    with open(pop) as popsa:  # opening .out file
        for line in popsa:
            line_list = line.split()

            # extracting relevant information
            if len(line_list) == 12:
                residue_type = line_list[2] + line_list[4]
                if residue_type in siteresidue_list:
                    qsasa = line_list[7]
                    residue_list.append(residue_type)
                    qsasa_list.append(qsasa)

    qsasa_list = [float(x) for x in qsasa_list]
    median = statistics.median(qsasa_list)
    qsasa_new = [median if x == '-nan' else x for x in qsasa_list]

    # Matching amino acids from .mol2 and .out files and
    # creating dictionary
    qsasa_data = {}
    fullprotein_data = list(zip(residue_list, qsasa_new))
    for i in range(len(fullprotein_data)):
        if fullprotein_data[i][0] in siteresidue_list:
            qsasa_data[i + 1] = float(fullprotein_data[i][1])

    return qsasa_data


def amino_single_to_triple(single):
    """converts the single letter amino acid abbreviation 
    to the triple letter abbreviation"""

    single_to_triple_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
                             'G': 'GLY', 'Q': 'GLN', 'E': 'GLU', 'H': 'HIS', 'I': 'ILE',
                             'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
                             'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}

    for i in single_to_triple_dict.keys():
        if i == single:
            triple = single_to_triple_dict[i]

    return triple


def extract_seq_entropy_data(profile, mol):
    '''extracts sequence entropy data from .profile'''

    # Extracting data from mol2
    pd.options.mode.chained_assignment = None
    mol2 = PandasMol2().read_mol2(mol)
    atoms = mol2.df[['subst_name']]
    atoms.columns = ['residue_type']
    siteresidue_list = atoms['residue_type'].tolist()

    # Opening and formatting lists of the probabilities and residues
    with open(profile) as profile:
        ressingle_list = []
        probdata_list = []
        # extracting relevant information
        for line in profile:
            line_list = line.split()
            residue_type = line_list[0]
            prob_data = line_list[1:]
            prob_data = list(map(float, prob_data))
            ressingle_list.append(residue_type)
            probdata_list.append(prob_data)

    ressingle_list = ressingle_list[1:]
    probdata_list = probdata_list[1:]

    # Changing single letter amino acid to triple letter with
    # its corresponding number
    count = 0
    restriple_list = []
    for res in ressingle_list:
        newres = res.replace(res, amino_single_to_triple(res))
        count += 1
        restriple_list.append(newres + str(count))

    # Calculating information entropy
    with np.errstate(divide='ignore'):
        prob_array = np.asarray(probdata_list)
        log_array = np.log2(prob_array)
        # change all infinite values to 0
        log_array[~np.isfinite(log_array)] = 0
        entropy_array = log_array * prob_array
        entropydata_array = np.sum(a=entropy_array, axis=1) * -1
        entropydata_list = entropydata_array.tolist()

    # Matching amino acids from .mol2 and .profile files and creating dictionary
    fullprotein_data = dict(zip(restriple_list, entropydata_list))
    seq_entropy_data = {k: float(
        fullprotein_data[k]) for k in siteresidue_list if k in fullprotein_data}

    return seq_entropy_data


# Hard coded datasets
hydrophobicity_data = {'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
                       'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
                       'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9,
                       'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8,
                       'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2}

binding_prob_data = {'ALA': 0.701, 'ARG': 0.916, 'ASN': 0.811, 'ASP': 1.015,
                     'CYS': 1.650, 'GLN': 0.669, 'GLU': 0.956, 'GLY': 0.788,
                     'HIS': 2.286, 'ILE': 1.006, 'LEU': 1.045, 'LYS': 0.468,
                     'MET': 1.894, 'PHE': 1.952, 'PRO': 0.212, 'SER': 0.883,
                     'THR': 0.730, 'TRP': 3.084, 'TYR': 1.672, 'VAL': 0.884}


def Bionoi(mol, pop, profile, bs_out, size, colorby, dpi, alpha, proj_direction):

    # Dataset and colorscale determined by colorby
    if colorby in ["atom_type", "residue_type"]:
        dataset = None
        colorscale = None
    elif colorby == "hydrophobicity":
        dataset = hydrophobicity_data
        colorscale = "red_cyan"
    elif colorby == "charge":
        dataset = extract_charge_data(mol)
        colorscale = "orange_bluecyan"
    elif colorby == "binding_prob":
        dataset = binding_prob_data
        colorscale = "greencyan_redmagenta"
    elif colorby == "center_dist":
        dataset = extract_centerdistance_data(mol, proj_direction)
        colorscale = "yellow_blue"
    elif colorby == "sasa":
        dataset = extract_sasa_data(mol, pop)
        colorscale = "greenyellow_bluemagenta"
    elif colorby == "seq_entropy":
        dataset = extract_seq_entropy_data(profile, mol)
        colorscale = "green_magenta"

    # Run
    # create colormap
    cmap = custom_colormap(colorscale)

    # normalize dataset
    valnorm_lst = normalizer(dataset, colorby)

    # apply colormap to normalized data
    color_map = colorgen(colorby, valnorm_lst, cmap, dataset)

    # create voronoi diagram
    atoms, vor, img = voronoi_atoms(
        mol, color_map, colorby,
        bs_out=bs_out, size=size,
        dpi=dpi, alpha=alpha,
        save_fig=False, proj_direction=proj_direction
    )

    return atoms, vor, img


def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-mol',
                        default="./testfiles/mol2/5iunE00.mol2",
                        required=False,
                        help='the protein/ligand mol2 file')
    parser.add_argument('-pop',
                        default="./testfiles/popsa/5iunE.out",
                        required=False,
                        help='the protein file with qsasa values, used POPSlegacy')
    parser.add_argument('-profile',
                        default="./testfiles/profile/5iunE.profile",
                        required=False,
                        help='.profile file with sequence entropy data')
    parser.add_argument('-out',
                        default="./output/",
                        required=False,
                        help='the folder of output images file')
    parser.add_argument('-dpi',
                        default=256,
                        required=False,
                        help='image quality in dpi')
    parser.add_argument('-size', default=256,
                        required=False,
                        help='image size in pixels, eg: 128')
    parser.add_argument('-alpha',
                        default=0.5,
                        required=False,
                        help='alpha for color of cells')
    parser.add_argument('-colorby',
                        default="residue_type",
                        choices=["atom_type", "residue_type", "charge", "binding_prob", "hydrophobicity",
                                 "center_dist", "sasa", "seq_entropy", "blended"],
                        required=False,
                        help='color the voronoi cells according to atom type, residue type , charge,  \
                              binding probability, hydrophobicity, center distance, solvent accessible   \
                              surface area, and sequence entropy')
    parser.add_argument('-image_type',
                        default=".png",
                        choices=[".jpg", ".png"],
                        required=False,
                        help='the type of image {.jpg, .png}')
    parser.add_argument('-direction',
                        type=int,
                        default=1,
                        choices=[0, 1, 2, 3, 4, 5, 6],
                        required=False,
                        help='The direction of projection. Input the index [0-6] for direction(s) of interest \
                             in this tuple: (ALL, xy+, xy-, yz+, yz-, zx+, zx-)')
    parser.add_argument('-rot_angle',
                        type=int,
                        default=1,
                        choices=[0, 1, 2, 3, 4],
                        required=False,
                        help='The angle of rotation. Input the index [0-4] for direction(s) of interest \
                              in this tuple: (ALL, 0, 90, 180, 270)')
    parser.add_argument('-flip',
                        type=int,
                        default=1,
                        choices=[0, 1, 2, 3],
                        required=False,
                        help='The type of flipping. Input the index [0-3] for flip of interest \
                              in this tuple: (ALL, original, up-down, left-right)')
    parser.add_argument('-save_fig',
                        default=False,
                        choices=[True, False],
                        required=False,
                        help='Whether or not the original image needs saving.')
    return parser.parse_args()


def rotate(proj_img_list, rotation_angle):
    '''rotates voronoi diagram according to the specified -rot_angle'''

    rotate_img_list = []
    for img in proj_img_list:
        # do four 90 degree rotations if rot_angle = 0
        if rotation_angle == 0:

            rotate_img_list.append(img)
            rotate_img_list.append(skrotate(img, angle=90))
            rotate_img_list.append(skrotate(img, angle=180))
            rotate_img_list.append(skrotate(img, angle=270))
        elif rotation_angle == 1:
            # no rotation
            rotate_img_list.append(skrotate(img, angle=0))
        elif rotation_angle == 2:
            # 90 degree rotation
            rotate_img_list.append(skrotate(img, angle=90))
        elif rotation_angle == 3:
            # 180 degree rotation
            rotate_img_list.append(skrotate(img, angle=180))
        elif rotation_angle == 4:
            # 270 degree rotation
            rotate_img_list.append(skrotate(img, angle=270))
    return rotate_img_list


def flip(rotate_img_list, flip):
    '''flips voronoi diagram according to specified -flip'''

    flip_img_list = []
    for img in rotate_img_list:
        if flip == 0:
            # no flip
            flip_img_list.append(img)
            # flip over x axis
            flip_img_list.append(np.flipud(img))
            # flip over y axes
            # flip_img_list.append(np.fliplr(img))
        if flip == 1:
            flip_img_list.append(img)
        if flip == 2:
            img = np.flipud(img)
            flip_img_list.append(img)
        if flip == 3:
            img = np.fliplr(img)
            flip_img_list.append(img)

    return flip_img_list


def blend_properties(img_list):
    '''blends six voronoi diagrams made to show different properties'''

    blend_list = []
    # extracting a pocket list from the main list
    for pocket_list in img_list:
        # extracting images from pocket list
        im1 = pocket_list[0]
        im2 = pocket_list[1]
        im3 = pocket_list[2]
        im4 = pocket_list[3]
        im5 = pocket_list[4]
        im6 = pocket_list[5]

        # Blending two images at a time with equal weights
        blend1 = cv2.addWeighted(im1, .5, im2, .5, 0)
        blend2 = cv2.addWeighted(im3, .5, im4, .5, 0)
        blend3 = cv2.addWeighted(im5, .5, im6, .5, 0)

        # Continue blending images, keeping weights equal
        multiblend1 = cv2.addWeighted(blend1, .5, blend2, .5, 0)
        finalblend = cv2.addWeighted(multiblend1, 2 / 3, blend3, 1 / 3, 0)
        blend_list.append(finalblend)

    return blend_list


def gen_output_filenames(direction, rotation_angle, flip):
    """generate output names based on the direction, rot_angle, and flip"""

    proj_names = []
    rot_names = []
    flip_names = []

    if direction != 0:
        name = ''
        if direction == 1:
            name = 'XOY+'
        elif direction == 2:
            name = 'XOY-'
        elif direction == 3:
            name = 'YOZ+'
        elif direction == 4:
            name = 'YOZ-'
        elif direction == 5:
            name = 'ZOX+'
        elif direction == 6:
            name = 'ZOX-'
        proj_names.append(name)
    elif direction == 0:
        proj_names = ['XOY+', 'XOY-', 'YOZ+', 'YOZ-', 'ZOX+', 'ZOX-']

    if rotation_angle != 0:
        name = ''
        if rotation_angle == 1:
            name = '_r0'
        elif rotation_angle == 2:
            name = '_r90'
        elif rotation_angle == 3:
            name = '_r180'
        elif rotation_angle == 4:
            name = '_r270'
        rot_names.append(name)
    else:
        rot_names = ['_r0', '_r90', '_r180', '_r270']

    if flip != 0:
        name = ''
        if flip == 1:
            name = '_OO'
        elif flip == 2:
            name = '_ud'
        elif flip == 3:
            name = '_lr'
        flip_names.append(name)
    else:
        # flip_names = ['_OO', '_ud', '_lr']
        flip_names = ['_OO', '_ud']
    return proj_names, rot_names, flip_names


def img_gen(args):
    """
    Generate Voronoi diagrams according to the input arguments.
    """
    mol = args.mol
    pop = args.pop
    profile = args.profile
    out_folder = args.out

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Alias args
    size = args.size
    dpi = args.dpi
    alpha = args.alpha
    imgtype = args.image_type
    colorby = args.colorby
    proj_direction = args.direction
    rotation_angle = args.rot_angle
    flip_type = args.flip

    # get the name of the binding site
    basepath = os.path.basename(mol)
    basename = os.path.splitext(basepath)[0]

    proj_names, rot_names, flip_names = gen_output_filenames(
        proj_direction, rotation_angle, flip_type)
    len_list = len(proj_names)
    proj_img_list = []
    proj_img_list_all = []
    colorby_list = ("charge", "binding_prob", "hydrophobicity",
                    "center_dist", "sasa", "seq_entropy")

    # Projecting
    for i, proj_name in enumerate(proj_names):
        if colorby == "blended":
            # creates diagrams for all six properties instead of one
            for property in colorby_list:
                atoms, vor, img = Bionoi(mol=mol,
                                         pop=pop,
                                         profile=profile,
                                         bs_out=out_folder + proj_name,
                                         size=size,
                                         dpi=dpi,
                                         alpha=alpha,
                                         colorby=property,
                                         proj_direction=i + 1 if proj_direction == 0 else proj_direction)
                proj_img_list_all.append(img)

        else:
            atoms, vor, img = Bionoi(mol=mol,
                                     pop=pop,
                                     profile=profile,
                                     bs_out=out_folder + proj_name,
                                     size=size,
                                     dpi=dpi,
                                     alpha=alpha,
                                     colorby=colorby,
                                     proj_direction=i + 1 if proj_direction == 0 else proj_direction)
            proj_img_list.append(img)

    if colorby == "blended":
        # Grouping images by pocket. 6 images to a pocket, each displaying a different property
        order_list = [(proj_img_list_all[i], proj_img_list_all[i + 1], proj_img_list_all[i + 2],
                       proj_img_list_all[i + 3], proj_img_list_all[i + 4], proj_img_list_all[i + 5]) for i in
                      range(0, len(proj_img_list_all), 6)]

        # Blend each pocket with all 6 properties
        proj_img_list = blend_properties(order_list)

    # Rotate
    rotate_img_list = rotate(proj_img_list, rotation_angle)

    # Flip
    flip_img_list = flip(rotate_img_list, flip_type)

    assert len(proj_img_list) == len(proj_names)
    assert len(rotate_img_list) == len(rot_names)*len(proj_names)
    assert len(flip_img_list) == len(flip_names)*len(rot_names)*len(proj_names)

    # Setup output folder
    filenames = []
    for file in list(os.listdir(out_folder)):
        if os.path.isfile(file):
            os.remove(file)

    # Make file names
    for pname in proj_names:
        for rname in rot_names:
            for fname in flip_names:
                saveFile = os.path.join(
                    out_folder, basename + '_' + pname + rname + fname + imgtype)
                filenames.append(saveFile)

    assert len(filenames) == len(flip_img_list)

    # Save images
    for i in range(len(filenames)):
        skimage.io.imsave(filenames[i], img_as_ubyte(flip_img_list[i]))


if __name__ == "__main__":
    args = get_args()
    img_gen(args)
