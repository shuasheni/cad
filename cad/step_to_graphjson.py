import json
import numpy as np
from occwl.graph import face_adjacency
from occwl.io import load_step
from occwl.uvgrid import ugrid, uvgrid
from tqdm import tqdm
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from pathlib import Path

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import *
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_LinearProperties

from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties

surface_type_reverse_map = {
    "plane": "PlaneSurfaceType",
    "cylinder": "CylinderSurfaceType",
    "cone": "ConeSurfaceType",
    "sphere": "SphereSurfaceType",
    "torus": "TorusSurfaceType",
    "bezier": "NurbsSurfaceType",
    "bspline": "NurbsSurfaceType",
    "revolution": "NurbsSurfaceType",
    "extrusion": "NurbsSurfaceType",
    "offset": "NurbsSurfaceType",
    "other": "NurbsSurfaceType"
}

curve_type_reverse_map = {
    "line": "Line3DCurveType",
    "circle": "Circle3DCurveType",
    "ellipse": "Ellipse3DCurveType",
    "hyperbola": "Degenerate3DCurveType",
    "parabola": "Degenerate3DCurveType",
    "bezier": "NurbsCurve3DCurveType",
    "bspline": "NurbsCurve3DCurveType",
    "offset": "NurbsCurve3DCurveType",
    "other": "Degenerate3DCurveType",
}


def get_boundingbox(shape, tol=1e-6, use_mesh=True):
    bbox = Bnd_Box()
    bbox.SetGap(tol)
    if use_mesh:
        mesh = BRepMesh_IncrementalMesh()
        mesh.SetParallelDefault(True)
        mesh.SetShape(shape)
        mesh.Perform()
        if not mesh.IsDone():
            raise AssertionError("Mesh not done.")
    brepbndlib_Add(shape, bbox, use_mesh)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return xmin, ymin, zmin, xmax, ymax, zmax


def build_graph_json(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    # Build face adjacency graph with B-rep entities as node and edge features
    graph = face_adjacency(solid)
    solid_shape = solid.topods_shape()
    xmin, ymin, zmin, xmax, ymax, zmax = get_boundingbox(solid_shape)

    face_num = len(graph.nodes)
    edge_num = int(len(graph.edges) / 2)

    # Compute the UV-grids for faces
    graph_face_feat = []
    for face_idx in graph.nodes:
        # Get the B-rep face
        face = graph.nodes[face_idx]["face"]

        props = GProp_GProps()
        brepgprop_SurfaceProperties(face.topods_shape(), props)

        cog = props.CentreOfMass()
        area = props.Mass()
        cog_x, cog_y, cog_z = cog.Coord()
        # ax1, ax2, ax3 = props.PrincipalProperties().FirstAxisOfInertia().Coord()

        # Compute UV-grids
        points = uvgrid(
            face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        normals = uvgrid(
            face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # Concatenate channel-wise to form face feature tensor
        face_feat = {
            "id": face_idx + 1,
            "points": (points * 0.1).transpose(1, 0, 2).reshape(-1).tolist(),
            "normals": normals.transpose(1, 0, 2).reshape(-1).tolist(),
            "trimming_mask": mask.astype(int).transpose().reshape(-1).tolist(),
            "surface_type": surface_type_reverse_map[face.surface_type()],
            "reversed": False,
            "area": area * 0.01,
            "centroid_x": cog_x * 0.1,
            "centroid_y": cog_y * 0.1,
            "centroid_z": cog_z * 0.1,
            "normal_x": normals[0][0][0],
            "normal_y": normals[0][0][1],
            "normal_z": normals[0][0][2],
        }
        surf = BRepAdaptor_Surface(face.topods_shape())
        surf_type = surf.GetType()
        if surf_type == GeomAbs_Plane:
            # print(f"{face_idx}--> plane")
            # look for the properties of the plane
            # first get the related gp_Pln
            gp_pln = surf.Plane()
            location = gp_pln.Location()  # a point of the plane
            normal = gp_pln.Axis().Direction()  # the plane normal
            # then export location and normal to the console output
            face_feat["origin_x"] = location.X() * 0.1
            face_feat["origin_y"] = location.Y() * 0.1
            face_feat["origin_z"] = location.Z() * 0.1
            face_feat["normal_x"] = normal.X()
            face_feat["normal_y"] = normal.Y()
            face_feat["normal_z"] = normal.Z()
        elif surf_type == GeomAbs_Cylinder:
            gp_cyl = surf.Cylinder()
            location = gp_cyl.Location()
            axis = gp_cyl.Axis().Direction()
            radius = gp_cyl.Radius()
            face_feat["origin_x"] = location.X() * 0.1
            face_feat["origin_y"] = location.Y() * 0.1
            face_feat["origin_z"] = location.Z() * 0.1
            face_feat["axis_x"] = axis.X()
            face_feat["axis_y"] = axis.Y()
            face_feat["axis_z"] = axis.Z()
            face_feat["radius"] = radius * 0.1
        elif surf_type == GeomAbs_Cone:
            gp_con = surf.Cone()
            location = gp_con.Location()
            axis = gp_con.Axis().Direction()
            # semi_angle = gp_con.SemiAngle()
            radius = gp_con.RefRadius()
            face_feat["origin_x"] = location.X() * 0.1
            face_feat["origin_y"] = location.Y() * 0.1
            face_feat["origin_z"] = location.Z() * 0.1
            face_feat["axis_x"] = axis.X()
            face_feat["axis_y"] = axis.Y()
            face_feat["axis_z"] = axis.Z()
            face_feat["radius"] = radius * 0.1
        elif surf_type == GeomAbs_Sphere:
            gp_sphere = surf.Sphere()
            center = gp_sphere.Location()
            radius = gp_sphere.Radius()
            face_feat["origin_x"] = center.X() * 0.1
            face_feat["origin_y"] = center.Y() * 0.1
            face_feat["origin_z"] = center.Z() * 0.1
            face_feat["radius"] = radius * 0.1
        elif surf_type == GeomAbs_Torus:
            gp_torus = surf.Torus()
            location = gp_torus.Location()
            axis = gp_torus.Axis().Direction()
            # major_radius = gp_torus.MajorRadius()
            # minor_radius = gp_torus.MinorRadius()
            face_feat["origin_x"] = location.X() * 0.1
            face_feat["origin_y"] = location.Y() * 0.1
            face_feat["origin_z"] = location.Z() * 0.1
            face_feat["axis_x"] = axis.X()
            face_feat["axis_y"] = axis.Y()
            face_feat["axis_z"] = axis.Z()
        graph_face_feat.append(face_feat)
    graph_face_feat = np.asarray(graph_face_feat)

    links = []

    # Compute the U-grids for edges
    graph_edge_feat = []
    e_idx = 0
    for edge_idx in graph.edges:
        if edge_idx[0] > edge_idx[1]:
            continue
        # Get the B-rep edge
        edge = graph.edges[edge_idx]["edge"]
        props = GProp_GProps()
        brepgprop_LinearProperties(edge.topods_shape(), props)
        ctr = props.CentreOfMass()
        leng = props.Mass()
        ctr_x, ctr_y, ctr_z = ctr.Coord()

        # Ignore dgenerate edges, e.g. at apex of cone
        if not edge.has_curve():
            continue
        # Compute U-grids
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)

        # Concatenate channel-wise to form edge feature tensor
        edge_feat = {
            "id": e_idx,
            "is_degenerate": False,
            "points": (points * 0.1).reshape(-1).tolist(),
            "tangents": tangents.reshape(-1).tolist(),
            "curve_type": curve_type_reverse_map[edge.curve_type()],
            "length": leng * 0.1,
            "reversed": False,
            "convexity": 'None',
            "dihedral_angle": 0,
        }
        link1 = {
            "source": edge_idx[0] + 1,
            "target": e_idx
        }
        link2 = {
            "source": edge_idx[1] + 1,
            "target": e_idx
        }
        links.append(link1)
        links.append(link2)
        if e_idx == 0:
            e_idx = face_num + 1
        else:
            e_idx += 1

        curv = BRepAdaptor_Curve(edge.topods_shape())
        curv_type = curv.GetType()
        if curv_type == GeomAbs_Circle:
            gp_circ = curv.Circle()
            location = gp_circ.Location()  # a point of the plane
            radius = gp_circ.Radius()
            length = gp_circ.Length()
            axis = gp_circ.Axis().Direction()  # the cylinder axis
            edge_feat["length"] = length * 0.1
            edge_feat["center_x"] = location.X() * 0.1
            edge_feat["center_y"] = location.Y() * 0.1
            edge_feat["center_z"] = location.Z() * 0.1
            edge_feat["normal_x"] = axis.X()
            edge_feat["normal_y"] = axis.Y()
            edge_feat["normal_z"] = axis.Z()
            edge_feat["radius"] = radius * 0.1
        elif curv_type == GeomAbs_Line:
            gp_line = curv.Line()
            u_start = curv.FirstParameter()
            u_end = curv.LastParameter()
            start_point = curv.Value(u_start)
            end_point = curv.Value(u_end)
            length = start_point.Distance(end_point)
            edge_feat["length"] = length * 0.1
            edge_feat["start_point_x"] = start_point.X()
            edge_feat["start_point_y"] = start_point.Y()
            edge_feat["start_point_z"] = start_point.Z()
            edge_feat["end_point_x"] = end_point.X()
            edge_feat["end_point_y"] = end_point.Y()
            edge_feat["end_point_z"] = end_point.Z()
        # edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    props = GProp_GProps()
    brepgprop_VolumeProperties(solid_shape, props)
    volume = props.Mass()

    props = GProp_GProps()
    brepgprop_SurfaceProperties(solid_shape, props)
    area = props.Mass()


    json_graph = {
        'directed': False,
        'multigraph': False,
        'graph': {},
        'nodes': graph_face_feat.tolist() + graph_edge_feat.tolist(),
        'links': links,
        'properties': {
            'bounding_box': {
                "type": "BoundingBox3D",
                "max_point": {
                    "type": "Point3D",
                    "x": xmax * 0.1,
                    "y": ymax * 0.1,
                    "z": zmax * 0.1
                },
                "min_point": {
                    "type": "Point3D",
                    "x": xmin * 0.1,
                    "y": ymin * 0.1,
                    "z": zmin * 0.1
                }
            },
            "edge_count": edge_num,
            "face_count": face_num,
            "volume": volume * 0.001,
            "area": area * 0.01,
        }
    }
    return json_graph, face_num, edge_num


def process_one_file(fn, id, output_path, curv_u_samples=10, surf_u_samples=10, surf_v_samples=10):
    solid = load_step(fn)[0]  # Assume there's one solid per file
    json_graph, face_num, edge_num = build_graph_json(
        solid, curv_u_samples, surf_u_samples, surf_v_samples
    )

    with open(Path(output_path).joinpath(id + ".json"), 'w') as f:
        json.dump(json_graph, f, indent=4)

    return face_num, edge_num


def process_all(input_path, output_path):
    print(input_path)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    step_files = list(input_path.glob("*.st*p"))
    for fn in tqdm(step_files):
        process_one_file(fn, output_path)
