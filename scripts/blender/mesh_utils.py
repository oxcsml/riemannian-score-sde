import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def regular_square_mesh_to_obj(verticies, wrap_x=False, wrap_y=False):
    # assumes verticies are regularly spaced on a 2d grid and that you want th eUV map to be square.
    n, m = verticies.shape[:2]
    n_uv = n + 1 if wrap_x else n
    m_uv = m + 1 if wrap_y else m

    ii, jj = np.meshgrid(np.arange(n_uv), np.arange(m_uv))
    ii = ii.T / (n_uv - 1)
    jj = jj.T / (m_uv - 1)

    uv_coords = np.stack([ii, jj], axis=-1)
    uv_coords = np.flip(uv_coords, axis=-1)

    return square_mesh_to_obj(verticies, uv_coords, wrap_x=wrap_x, wrap_y=wrap_y)


def square_mesh_to_obj(verticies, uv_coords=None, wrap_x=False, wrap_y=False):
    """Turns a mesh based on quad faces as a pushforward from a 2D square to some mesh.
    Assumes faces are built from the oredering of the verticies.

    Parameters
    ----------
    verticies : np.ndarry
        (n, m, 3) array of vertex coords
    uv_coords : np.ndarry, optional
        ([n/n+1] if wrap_x, [m/m+1] if wrap_y), 2, UV coords. Assumes you've added the
        extra uv coord manually.
    wrap_x : bool, optional
        if the x edge gets wrapped around
    wrap_y : bool, optional
        if the y edge gets wrapped around
    """

    n, m = verticies.shape[:2]

    # build faces from verticies
    n_faces = n if wrap_x else n - 1
    m_faces = m if wrap_y else m - 1

    ii, jj = np.meshgrid(np.arange(n), np.arange(m))
    ii = ii.T
    jj = jj.T
    face_index = ii + n * jj

    faces = np.zeros((n_faces, m_faces, 4), int)
    for i in range(n_faces):
        for j in range(m_faces):
            faces[i, j, 0] = face_index[i, j]
            faces[i, j, 1] = face_index[(i + 1) % n, j]
            faces[i, j, 2] = face_index[(i + 1) % n, (j + 1) % m]
            faces[i, j, 3] = face_index[i, (j + 1) % m]

    n_uv = n + wrap_x
    m_uv = m + wrap_y
    ii, jj = np.meshgrid(np.arange(n_uv), np.arange(m_uv))
    ii = ii.T
    jj = jj.T
    uv_index = ii + n_uv * jj
    uv_indicies = np.zeros((n_faces, m_faces, 4), int)
    for i in range(n_faces):
        for j in range(m_faces):
            uv_indicies[i, j, 0] = uv_index[i, j]
            uv_indicies[i, j, 1] = uv_index[i + 1, j]
            uv_indicies[i, j, 2] = uv_index[i + 1, j + 1]
            uv_indicies[i, j, 3] = uv_index[i, j + 1]

    lines = []

    lines.append("# Verticies")

    for j in range(m):
        for i in range(n):
            lines.append(
                f"v {verticies[i, j, 0]:0.5f} {verticies[i, j, 1]:0.5f} {verticies[i, j, 2]:0.5f}"
            )

    if uv_coords is not None:
        lines.append("")
        lines.append("# UV Coords")

        for j in range(m_uv):
            for i in range(n_uv):
                lines.append(f"vt {uv_coords[i, j, 0]:0.5f} {uv_coords[i, j, 1]:0.5f}")

    lines.append("# Faces")
    if uv_coords is not None:
        for j in range(m_faces):
            for i in range(n_faces):
                lines.append(
                    f"f {faces[i, j, 0]+1}/{uv_indicies[i, j, 0]+1} {faces[i, j, 1]+1}/{uv_indicies[i, j, 1]+1} {faces[i, j, 2]+1}/{uv_indicies[i, j, 2]+1} {faces[i, j, 3]+1}/{uv_indicies[i, j, 3]+1}"
                )
    else:
        for j in range(m_faces):
            for i in range(n_faces):
                lines.append(
                    f"f {faces[i,j,0]+1} {faces[i,j,1]+1} {faces[i,j,2]+1} {faces[i,j,3]+1}"
                )

    return lines


def mesh_to_obj(verticies, faces, uv_coords=None):
    lines = []

    lines.append("# Verticies")

    for i in range(verticies.shape[0]):
        lines.append(
            f"v {verticies[i, 0]:0.5f} {verticies[i, 1]:0.5f} {verticies[i, 2]:0.5f}"
        )

    if uv_coords is not None:
        lines.append("")
        lines.append("# UV Coords")

        for i in range(uv_coords.shape[0]):
            lines.append(f"vt {uv_coords[i, 0]:0.5f} {uv_coords[i, 1]:0.5f}")

    lines.append("")

    lines.append("# Faces")
    if uv_coords is not None:
        if faces.shape[1] == 3:
            for i in range(faces.shape[0]):
                lines.append(
                    f"f {faces[i,0]+1}/{faces[i,0]+1} {faces[i,1]+1}/{faces[i,1]+1} {faces[i,2]+1}/{faces[i,2]+1}"
                )
        elif faces.shape[1] == 4:
            for i in range(faces.shape[0]):
                lines.append(
                    f"f {faces[i,0]+1}/{faces[i,0]+1} {faces[i,1]+1}/{faces[i,1]+1} {faces[i,2]+1}/{faces[i,2]+1} {faces[i,3]+1}/{faces[i,3]+1}"
                )
    else:
        if faces.shape[1] == 3:
            for i in range(faces.shape[0]):
                lines.append(f"f {faces[i,0]+1} {faces[i,1]+1} {faces[i,2]+1}")
        elif faces.shape[1] == 4:
            for i in range(faces.shape[0]):
                lines.append(
                    f"f {faces[i,0]+1} {faces[i,1]+1} {faces[i,2]+1} {faces[i,3]+1}"
                )

    return lines


def make_faces_from_vectors(verticies, vectors):

    vert_lines = []

    face_lines = []

    for i in range(vectors.shape[0]):
        vert = verticies[i]
        vec_norm = jnp.linalg.norm(vectors[i])
        if vectors[i, 2] != 0.0:
            orth1 = jnp.array([0, 1, -vectors[i, 1] / vectors[i, 2]])
            orth1 = orth1 / jnp.linalg.norm(orth1)
            orth2 = jnp.cross(vectors[i], orth1)
            orth2 = orth2 / jnp.linalg.norm(orth2)
        else:
            orth1 = jnp.array([0, 1, 0])
            orth2 = jnp.array([1, 0, 0])

        vert_pp = vert + vec_norm * (orth1 + orth2)
        vert_lines.append(f"v {vert_pp[0]:0.5f} {vert_pp[1]:0.5f} {vert_pp[2]:0.5f}")
        vert_pm = vert + vec_norm * (orth1 - orth2)
        vert_lines.append(f"v {vert_pm[0]:0.5f} {vert_pm[1]:0.5f} {vert_pm[2]:0.5f}")
        vert_mm = vert + vec_norm * (-orth1 - orth2)
        vert_lines.append(f"v {vert_mm[0]:0.5f} {vert_mm[1]:0.5f} {vert_mm[2]:0.5f}")
        vert_mp = vert + vec_norm * (-orth1 + orth2)
        vert_lines.append(f"v {vert_mp[0]:0.5f} {vert_mp[1]:0.5f} {vert_mp[2]:0.5f}")

        face_lines.append(f"f {i*4 + 1} {i*4 + 2} {i*4 + 3} {i*4 + 4}")

    lines = []

    lines.append("# Verticies")

    lines += vert_lines

    lines += ["", "# Faces"]

    lines += face_lines

    return lines


def make_verticies_from_vectors(verticies, vectors):

    vert_lines = []

    face_lines = []

    for i in range(vectors.shape[0]):
        vert = verticies[i]
        vec_norm = jnp.sqrt(jnp.linalg.norm(vectors[i]))
        if vectors[i, 2] != 0.0:
            orth1 = jnp.array([0, 1, -vectors[i, 1] / vectors[i, 2]])
            orth1 = orth1 / jnp.linalg.norm(orth1)
            orth2 = jnp.cross(vectors[i], orth1)
            orth2 = orth2 / jnp.linalg.norm(orth2)
        else:
            orth1 = jnp.array([0, 1, 0])
            orth2 = jnp.array([1, 0, 0])

        vert_pp = vert + vec_norm * (orth1 + orth2)
        vert_lines.append(f"v {vert_pp[0]:0.5f} {vert_pp[1]:0.5f} {vert_pp[2]:0.5f}")
        vert_pm = vert + vec_norm * (orth1 - orth2)
        vert_lines.append(f"v {vert_pm[0]:0.5f} {vert_pm[1]:0.5f} {vert_pm[2]:0.5f}")
        vert_mm = vert + vec_norm * (-orth1 - orth2)
        vert_lines.append(f"v {vert_mm[0]:0.5f} {vert_mm[1]:0.5f} {vert_mm[2]:0.5f}")
        vert_mp = vert + vec_norm * (-orth1 + orth2)
        vert_lines.append(f"v {vert_mp[0]:0.5f} {vert_mp[1]:0.5f} {vert_mp[2]:0.5f}")

        face_lines.append(f"f {i*4 + 1} {i*4 + 2} {i*4 + 3} {i*4 + 4}")

    lines = []

    lines.append("# Verticies")

    lines += vert_lines

    lines += ["", "# Faces"]

    lines += face_lines

    return lines


def make_scalar_texture(function, locations, file, cmap="viridis", **kwargs):
    locations_shape = locations.shape
    locations = locations.reshape((-1, *locations_shape[2:]))
    values = function(locations)
    values = values.reshape(locations_shape[:2])
    plt.imsave(file, values, cmap=cmap, **kwargs)


def save_obj(lines, file):
    with open(file, "w") as f:
        for line in lines:
            f.write(line + "\n")


def export_vec_field(V, vecs, file):
    np.savetxt(file, jnp.concatenate([V, vecs], axis=1), delimiter=",")


import numpy as np
import jax
from jax import grad
import jax.numpy as jnp


def mesh_to_polyscope(mesh, wrap_x=True, wrap_y=True, reverse_x=False, reverse_y=False):
    n, m, _ = mesh.shape

    n_faces = n if wrap_x else n - 1
    m_faces = m if wrap_y else m - 1

    ii, jj = np.meshgrid(np.arange(n), np.arange(m))
    ii = ii.T
    jj = jj.T
    coords = jj + m * ii

    faces = np.zeros((n_faces, m_faces, 4), int)
    for i in range(n_faces):
        for j in range(m_faces):

            c1 = [i, j]
            c2 = [(i + 1) % n, j]
            c3 = [(i + 1) % n, (j + 1) % m]
            c4 = [i, (j + 1) % m]

            # print(i, n)
            if (i == n - 1) and reverse_x:
                c2[1] = (-c2[1] - 2) % m
                c3[1] = (-c3[1] - 2) % m
                # c2[1] = (-c2[1] - int(m / 2) - 2) % m
                # c3[1] = (-c3[1] - int(m / 2) - 2) % m
            if (j == m - 1) and reverse_y:
                c3[0] = (-c3[0] - 2) % n
                c4[0] = (-c4[0] - 2) % n
                # c3[0] = (-c3[0] - int(n / 2) - 2) % n
                # c4[0] = (-c4[0] - int(n / 2) - 2) % n

            faces[i, j, 0] = coords[c1[0], c1[1]]
            faces[i, j, 1] = coords[c2[0], c2[1]]
            faces[i, j, 2] = coords[c3[0], c3[1]]
            faces[i, j, 3] = coords[c4[0], c4[1]]

            # if i == (n - 1)

    mesh_ = mesh.reshape(-1, 3)
    faces_ = faces.reshape(-1, 4)

    return mesh_, faces_


def mesh_to_polyscope_triangular(mesh, wrap_x=True, wrap_y=True):
    n, m, _ = mesh.shape

    n_faces = n if wrap_x else n - 1
    m_faces = m if wrap_y else m - 1

    ii, jj = np.meshgrid(np.arange(n), np.arange(m))
    ii = ii.T
    jj = jj.T
    coords = jj + m * ii

    faces = np.zeros((n_faces, m_faces * 2, 3), int)
    for i in range(n_faces):
        for j in range(m_faces):
            faces[i, 2 * j, 0] = coords[i, j]
            faces[i, 2 * j, 1] = coords[(i + 1) % n, j]
            faces[i, 2 * j, 2] = coords[i, (j + 1) % m]

            faces[i, (2 * j) + 1, 0] = coords[(i + 1) % n, j]
            faces[i, (2 * j) + 1, 1] = coords[(i + 1) % n, (j + 1) % m]
            faces[i, (2 * j) + 1, 2] = coords[i, (j + 1) % m]

    mesh_ = mesh.reshape(-1, 3)
    faces_ = faces.reshape(-1, 3)

    return mesh_, faces_


def project_to_3d(M, V, m_to_3d, projection_matrix_to_3d):
    X = m_to_3d(M)
    Y = (
        jnp.swapaxes(projection_matrix_to_3d(M), -1, -2) @ V[..., np.newaxis]
    ).squeeze()
    return X, Y


def cylinder_m_to_3d(M):
    theta, x = M[..., 0], M[..., 1]
    # np.take(M, 0, -1), np.take(M, 1, -1)
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    return jnp.stack([s, c, x], axis=-1)


def cylinder_projection_matrix_to_3d_1(M):
    theta, x = M[..., 0], M[..., 1]
    # np.take(M, 0, -1), np.take(M, 1, -1)
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    z = jnp.zeros_like(c)
    e1 = jnp.stack([c, -s, z], axis=-1)
    e2 = jnp.zeros_like(e1)
    return jnp.stack(
        [
            e1,
            # e2
        ],
        axis=-2,
    )


def cylinder_projection_matrix_to_3d(M):
    theta, x = M[..., 0], M[..., 1]
    # np.take(M, 0, -1), np.take(M, 1, -1)
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    z = jnp.zeros_like(c)
    o = jnp.ones_like(c)
    e1 = jnp.stack([c, -s, z], axis=-1)
    e2 = jnp.stack([z, z, o], axis=-1)
    return jnp.stack(
        [
            e1,
            e2,
        ],
        axis=-2,
    )


def t2_m_to_3d(M, R=0.7, r=0.25):
    theta1, theta2 = jnp.take(M, 0, -1), jnp.take(M, 1, -1)
    s1 = jnp.sin(theta1)
    c1 = jnp.cos(theta1)
    s2 = jnp.sin(theta2)
    c2 = jnp.cos(theta2)
    return jnp.stack([(R + r * c1) * c2, (R + r * c1) * s2, r * s1], axis=-1)


def t2_projection_matrix_to_3d(M, R=3, r=1):
    # # theta1, theta2 = M[...,0], M[...,1]
    # theta1, theta2 = jnp.take(M, 0, -1), jnp.take(M, 1, -1)
    # s1 = jnp.sin(theta1)
    # c1 = jnp.cos(theta1)
    # s2 = jnp.sin(theta2)
    # c2 = jnp.cos(theta2)
    # z = jnp.zeros_like(theta1)
    # e1 = jnp.stack([-r * s1 * c2, -r * s1 * s2, r * c1], axis=-1)
    # e2 = jnp.stack([-s2, z, z], axis=-1)
    # return jnp.stack(
    #     [
    #         e1,
    #         e2,
    #     ],
    #     axis=-2,
    # )
    grad_proj = jnp.stack(
        [jax.vmap(grad(lambda m: t2_m_to_3d(m)[..., i]))(M) for i in range(3)], axis=-1
    )
    return grad_proj / jnp.linalg.norm(grad_proj, axis=-1)[..., np.newaxis]


def klein_bottle_m_to_3d(M):
    u, v = M[..., 0], M[..., 1]

    cu = jnp.cos(u)
    su = jnp.sin(u)
    cv = jnp.cos(v)
    sv = jnp.sin(v)

    return jnp.stack(
        [
            -(2 / 15)
            * cu
            * (
                3 * cv
                - 30 * su
                + 90 * jnp.power(cu, 4) * su
                - 60 * jnp.power(cu, 6) * su
                + 5 * cu * cv * su
            ),
            -(1 / 15)
            * su
            * (
                3 * cv
                - 3 * jnp.power(cu, 2) * cv
                - 48 * jnp.power(cu, 4) * cv
                + 48 * jnp.power(cu, 6) * cv
                - 60 * su
                + 5 * cu * cv * su
                - 5 * jnp.power(cu, 3) * cv * su
                - 80 * jnp.power(cu, 5) * cv * su
                + 80 * jnp.power(cu, 7) * cv * su
            ),
            (2 / 15) * (3 + 5 * cu * su) * sv,
        ],
        axis=-1,
    )


def klein_fig8_m_to_3d(M, r=2):
    u, v = M[..., 0], M[..., 1]
    u = u * 2

    cu = jnp.cos(u)
    su = jnp.sin(u)
    cu2 = jnp.cos(u / 2)
    su2 = jnp.sin(u / 2)
    sv = jnp.sin(v)
    s2v = jnp.sin(2 * v)

    return jnp.stack(
        [
            (r + cu2 * sv - su2 * s2v) * cu,
            (r + cu2 * sv - su2 * s2v) * su,
            su2 * sv + cu2 * s2v,
        ],
        axis=-1,
    )


def klein_fig8_double_m_to_3d(M, r=2, delta=0.2):

    E = klein_fig8_m_to_3d(M, r=r)

    tangents = jnp.stack(
        [jax.vmap(grad(lambda m: klein_fig8_m_to_3d(m)[..., i]))(M) for i in range(3)],
        axis=-1,
    )
    normals = jax.vmap(lambda t: jnp.cross(t[0], t[1]))(tangents)
    normals = normals / jnp.linalg.norm(normals, axis=1, keepdims=True)

    return E + normals * delta


def sphere_m_to_3d(M):
    phi = M[..., 0]
    theta = M[..., 1]

    return jnp.stack(
        [
            jnp.sin(phi) * jnp.cos(theta),
            jnp.sin(phi) * jnp.sin(theta),
            jnp.cos(phi),
        ],
        axis=-1,
    )


def sphere_flat_m_to_3d(M):
    phi = M[..., 0]
    theta = M[..., 1]

    return jnp.stack(
        [
            jnp.zeros_like(theta),
            -(theta - jnp.pi),
            -(phi - (jnp.pi / 2)),
        ],
        axis=-1,
    )


def interp(M, embedding_1, embedding_2, t):
    return (1 - t) * embedding_1(M) + t * embedding_2(M)


def projection_matrix(M, embedding, embedding_dim=3):
    grad_proj = jnp.stack(
        [
            jax.vmap(jax.grad(lambda m: embedding(m)[..., i]))(M)
            for i in range(embedding_dim)
        ],
        axis=-2,
    )

    return grad_proj / jnp.linalg.norm(grad_proj, axis=-2)[..., np.newaxis, :]


def project(M, V, embedding):
    X = embedding(M)
    proj_mat = projection_matrix(M, embedding)
    Y = (proj_mat @ V[..., np.newaxis])[..., 0]

    return X, Y


def flatten(M, Y, embedding):
    proj_mat = projection_matrix(M, embedding)
    Y = (proj_mat @ jnp.swapaxes(proj_mat, -1, -2) @ Y[..., np.newaxis])[..., 0]
    return Y


def import_obj(file):
    vertices = []
    faces = []
    with open(file, "r") as file:
        for line in file:
            if line.startswith("v "):
                coords = line.split(" ")[1:]
                coords = jnp.array([float(c) for c in coords])
                vertices.append(coords)
            elif line.startswith("f "):
                coords = line.split(" ")[1:]
                coords = jnp.array([int(c.split("/")[0]) for c in coords])
                if len(coords) > 4:
                    continue
                faces.append(coords)
            # elif line.startswith("o "):
            #     offset = vertices_seen

    return jnp.stack(vertices, axis=0), jnp.stack(faces, axis=0) - 1
