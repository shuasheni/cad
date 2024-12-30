from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopoDS import topods_Face
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.gp import gp_Vec
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepTools import breptools



def compute_normal(v1, v2, v3):
    """Compute the normal vector for a triangle defined by vertices v1, v2, and v3."""
    vec1 = gp_Vec(v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
    vec2 = gp_Vec(v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2])
    normal = vec1.Crossed(vec2)  # Compute cross product to get the normal vector
    normal.Normalize()  # Normalize the vector to unit length
    return normal




def step_to_obj_with_normals(step_file, obj_file):
    # STEP Reader
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file)

    if status != 1:
        raise Exception(f"Error reading STEP file: {step_file}")

    # Transfer the data from STEP reader to the shape
    step_reader.TransferRoots()
    shape = step_reader.Shape()
    breptools.Clean(shape)

    # Create an incremental mesh (needed to convert BRep to mesh)
    # mesh = BRepMesh_IncrementalMesh(shape, 0.1, True, 1)  # 0.1 is the deflection (mesh resolution)
    # mesh.Perform()

    # Topology Explorer: explore faces
    topo_exp = TopologyExplorer(shape)

    # Lists to store vertices, normals, and faces for each group
    all_vertices = []
    all_normals = []
    face_meshes = []
    vertex_offset = 0
    normal_offset = 0

    for i, face in enumerate(topo_exp.faces()):

        breptools.Clean(face)
        # Discretize face to a mesh
        BRepMesh_IncrementalMesh(face, 0.2, True, 0.36).Perform()  # Ensure meshing for each face
        rev_flag = False
        if face.Orientation() == TopAbs_REVERSED:
            rev_flag = True

        # Get the triangulation for this face
        loc = TopLoc_Location()  # Initialize a TopLoc_Location object
        face_triangulation = BRep_Tool.Triangulation(topods_Face(face), loc)

        if face_triangulation is None:
            continue  # Skip faces without triangulation

        # Extract vertices and faces from the triangulation
        vertices = []
        faces = []
        normals = []  # Normals for this face

        for j in range(1, face_triangulation.NbNodes() + 1):
            vertex = face_triangulation.Node(j)
            vertices.append([vertex.X(), vertex.Y(), vertex.Z()])

        for j in range(1, face_triangulation.NbTriangles() + 1):
            triangle = face_triangulation.Triangle(j)
            # Get the vertices for the triangle
            if rev_flag:
                v1_idx = triangle.Value(3) - 1
                v2_idx = triangle.Value(2) - 1
                v3_idx = triangle.Value(1) - 1
            else:
                v1_idx = triangle.Value(1) - 1
                v2_idx = triangle.Value(2) - 1
                v3_idx = triangle.Value(3) - 1
            v1 = vertices[v1_idx]
            v2 = vertices[v2_idx]
            v3 = vertices[v3_idx]

            # Compute the normal for this triangle
            normal = compute_normal(v1, v2, v3)
            normals.append([normal.X(), normal.Y(), normal.Z()])

            # Store face with vertex indices and corresponding normal
            face_vertex_indices = [v1_idx + vertex_offset, v2_idx + vertex_offset, v3_idx + vertex_offset]
            face_normal_indices = [normal_offset + len(normals), normal_offset + len(normals),
                                   normal_offset + len(normals)]
            faces.append((face_vertex_indices, face_normal_indices))

        # Store the face mesh, vertices, and normals
        face_meshes.append((f"face {i}", faces))
        all_vertices.extend(vertices)
        all_normals.extend(normals)
        vertex_offset += len(vertices)
        normal_offset += len(normals)

    # Write to OBJ file with vertices and normals first, then face groups
    with open(obj_file, 'w') as obj:
        # Write all vertices at the beginning
        for v in all_vertices:
            obj.write(f"v {v[0]*0.1} {v[1]*0.1} {v[2]*0.1}\n")

        obj.write("\n")

        # Write all normals at the beginning
        for n in all_normals:
            obj.write(f"vn {n[0]} {n[1]} {n[2]}\n")

        obj.write("\n")

        # Write each face group
        for group_name, faces in face_meshes:
            obj.write(f"g {group_name}\n")
            for f_verts, f_normals in faces:
                obj.write(
                    f"f {f_verts[0] + 1}//{f_normals[0]} {f_verts[1] + 1}//{f_normals[1]} {f_verts[2] + 1}//{f_normals[2]}\n")
                # obj.write(
                #     f"f {f_verts[2] + 1}//{f_normals[2]} {f_verts[1] + 1}//{f_normals[1]} {f_verts[0] + 1}//{f_normals[0]}\n")
            obj.write("\n")

    print(f"Conversion complete: {obj_file}")