"""Write the passive arm's printable STLs (millimetres) from the simulation meshes (metres).

    python hardware/passive_arm/export_stl.py

The simulation meshes in gym_quadruped/robot_model/aliengo_follower/assets/ were exported
from the arm's CAD; this only rescales them to mm, the unit slicers expect, and reports each file's
bodies and mesh defects. Output: hardware/passive_arm/stl/.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SOURCE = os.path.join(REPO, 'gym_quadruped', 'robot_model',
                      'aliengo_follower', 'assets')
PARTS = ('base_arm_link', 'arm_link_1', 'arm_link_2', 'arm_link_3')
FACET = np.dtype([('normal', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')])


def read_binary_stl(path):
    data = open(path, 'rb').read()
    n = int(np.frombuffer(data[80:84], np.uint32)[0])
    if 84 + 50 * n != len(data):
        raise ValueError(f"{path} is not a binary STL")
    return np.frombuffer(data[84:], FACET).copy()


def write_binary_stl(path, facets, header):
    with open(path, 'wb') as f:
        f.write(header.encode()[:80].ljust(80, b' '))
        f.write(np.uint32(len(facets)).tobytes())
        f.write(facets.tobytes())


def mesh_report(tris):
    """Bodies (signed volume [cm^3], negative = an enclosed cavity), open and non-manifold edges."""
    verts, idx = np.unique(np.round(tris.reshape(-1, 3), 4), axis=0, return_inverse=True)
    faces = idx.reshape(-1, 3)
    parent = np.arange(len(verts))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b, c in faces:
        parent[find(b)] = find(a)
        parent[find(c)] = find(a)
    label = np.array([find(f[0]) for f in faces])
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1)
    _, count = np.unique(edges, axis=0, return_counts=True)
    volumes = []
    for body in np.unique(label):
        t = verts[faces[label == body]]
        volumes.append(np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6 / 1e3)
    return np.ptp(verts, 0), sorted(volumes, reverse=True), int((count == 1).sum()), int((count > 2).sum())


def main():
    os.makedirs(os.path.join(HERE, 'stl'), exist_ok=True)
    for part in PARTS:
        facets = read_binary_stl(os.path.join(SOURCE, f'{part}.stl'))
        facets['v'] *= 1000.0  # m -> mm; normals are unit vectors and stay
        write_binary_stl(os.path.join(HERE, 'stl', f'{part}.stl'), facets, f'PACC passive arm {part}, mm')
        extent, volumes, open_edges, nonmanifold = mesh_report(facets['v'].astype(float))
        print(f"{part}: {len(facets)} triangles, {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} mm, "
              f"bodies {[round(float(v), 1) for v in volumes]} cm^3, open edges {open_edges}, non-manifold edges {nonmanifold}")


if __name__ == '__main__':
    main()
