import sys
from occwl.graph import face_adjacency
from occwl.io import load_step
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (GeomAbs_BSplineSurface,
                              GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane,
                              GeomAbs_Sphere, GeomAbs_Torus)

from db.mongo_db import insert_step_parse, select_step_parse, update_step_parse

surface_types = [
    "plane",
    "cylinder",
    "cone",
    "sphere",
    "torus",
    "bezier",
    "bspline",
    "revolution",
    "extrusion",
    "offset",
    "other"
]

def parse_entities(solid):
    graph = face_adjacency(solid)
    face_num = len(graph.nodes)
    edge_num = int(len(graph.edges) / 2)

    # print(f"face_num: {face_num}, edge_num: {edge_num}")

    faces = []
    edges = []

    for face_idx in graph.nodes:
        face = graph.nodes[face_idx]["face"]

        face_shape = face.topods_shape()

        surface_adaptor = BRepAdaptor_Surface(face_shape)
        surf_type = surface_adaptor.GetType()
        face_info = {
            "id": face_idx + 1,
            "face_type": face.surface_type(),
            "params": [],
            "points": [],
            "labels": [],
            "note": ""
        }

        print(f"face_id: ({face_idx + 1})")
        print(f"face_type: ({face_info['face_type']})")

        if surf_type == GeomAbs_Plane:
            face_info["face_type"] = "平面"
            plane = surface_adaptor.Plane()
            ax3 = plane.Position()
            origin = ax3.Location()  # 获取原点
            normal = ax3.Direction()  # 获取法向量
            x_dir = ax3.XDirection()  # 获取X方向
            y_dir = ax3.YDirection()  # 获取Y方向
            face_info["params"].append(f"基准点: ({origin.X()}, {origin.Y()}, {origin.Z()})")
            face_info["params"].append(f"法向量: ({normal.X()}, {normal.Y()}, {normal.Z()})")
            face_info["params"].append(f"X 方向: ({x_dir.X()}, {x_dir.Y()}, {x_dir.Z()})")
            face_info["params"].append(f"Y 方向: ({y_dir.X()}, {y_dir.Y()}, {y_dir.Z()})")

        elif surf_type == GeomAbs_Cylinder:
            face_info["face_type"] = "圆柱面"
            cylinder = surface_adaptor.Cylinder()
            # 获取圆柱面的参数，例如半径和轴线
            radius = cylinder.Radius()
            axis = cylinder.Axis()
            location = axis.Location()
            direction = axis.Direction()
            face_info["params"].append(f"半径: {radius}")
            face_info["params"].append(f"基准点: {location.X()}, {location.Y()}, {location.Z()}")
            face_info["params"].append(f"轴向量: {direction.X()}, {direction.Y()}, {direction.Z()}")
        elif surf_type == GeomAbs_Cone:
            face_info["face_type"] = "圆锥面"
            cone = surface_adaptor.Cone()
            radius = cone.RefRadius()
            semi_angle = cone.SemiAngle()
            axis = cone.Axis()
            location = axis.Location()
            direction = axis.Direction()
            face_info["params"].append(f"半径: {radius}")
            face_info["params"].append(f"半角: {semi_angle}")
            face_info["params"].append(f"基准点: {location.X()}, {location.Y()}, {location.Z()}")
            face_info["params"].append(f"轴向量: {direction.X()}, {direction.Y()}, {direction.Z()}")
        elif surf_type == GeomAbs_Sphere:
            face_info["face_type"] = "球面"
            sphere = surface_adaptor.Sphere()
            radius = sphere.Radius()
            center = sphere.Location()
            face_info["params"].append(f"半径: {radius}")
            face_info["params"].append(f"中心点: {center.X()}, {center.Y()}, {center.Z()}")
        elif surf_type == GeomAbs_Torus:
            face_info["face_type"] = "环面"
            torus = surface_adaptor.Torus()
            major_radius = torus.MajorRadius()
            minor_radius = torus.MinorRadius()
            location = torus.Axis().Location()
            direction = torus.Axis().Direction()
            face_info["params"].append(f"Major Radius: {major_radius}")
            face_info["params"].append(f"Minor Radius: {minor_radius}")
            face_info["params"].append(f"Location: {location.X()}, {location.Y()}, {location.Z()}")
            face_info["params"].append(f"Direction: {direction.X()}, {direction.Y()}, {direction.Z()}")
        elif surf_type == GeomAbs_BSplineSurface:
            face_info["face_type"] = "B样条曲面"
            bspline_surface = surface_adaptor.BSpline()

            # 提取控制点
            num_poles_u = bspline_surface.NbUPoles()  # U 方向的控制点数量
            num_poles_v = bspline_surface.NbVPoles()  # V 方向的控制点数量
            face_info["params"].append(f"Number of control points in U direction: {num_poles_u}")
            face_info["params"].append(f"Number of control points in V direction: {num_poles_v}")

            for u in range(1, num_poles_u + 1):
                for v in range(1, num_poles_v + 1):
                    pole = bspline_surface.Pole(u, v)
                    face_info["points"].append(f"Control Point ({u}, {v}): ({pole.X()}, {pole.Y()}, {pole.Z()})")

            # 提取 U 和 V 方向上的节点（Knots）
            num_knots_u = bspline_surface.NbUKnots()
            num_knots_v = bspline_surface.NbVKnots()
            face_info["params"].append(f"Number of knots in U direction: {num_knots_u}")
            face_info["params"].append(f"Number of knots in V direction: {num_knots_v}")

            knots_u = [bspline_surface.UKnot(i + 1) for i in range(num_knots_u)]
            knots_v = [bspline_surface.VKnot(i + 1) for i in range(num_knots_v)]
            face_info["params"].append(f"U Knots: {knots_u}")
            face_info["params"].append(f"V Knots: {knots_v}")

            # 提取 U 和 V 方向上的阶数（Degree）
            degree_u = bspline_surface.UDegree()
            degree_v = bspline_surface.VDegree()
            face_info["params"].append(f"Degree in U direction: {degree_u}")
            face_info["params"].append(f"Degree in V direction: {degree_v}")

            # 闭合性判断
            is_u_closed = bspline_surface.IsUClosed()
            is_v_closed = bspline_surface.IsVClosed()
            face_info["params"].append(f"Is U closed: {is_u_closed}")
            face_info["params"].append(f"Is V closed: {is_v_closed}")
        else:
            print("no type")
            sys.exit()

        # print(face_info)

        faces.append(face_info)
    return faces, edges


def step_parse(body_id):
    faces, edges = select_step_parse(body_id)
    if faces is None:
        fn = f"C:\\Users\\40896\\Desktop\\data\\joint\\{body_id}.step"
        solid = load_step(fn)[0]
        faces, edges = parse_entities(solid)
        insert_step_parse(body_id, faces, edges)
    return faces, edges

def update_label_or_note(body_id, faces, edges):
    update_step_parse(body_id, faces, edges)

def update_face(body_id, index, face):
    faces, edges = select_step_parse(body_id)
    faces[index] = face
    update_step_parse(body_id, faces, edges)