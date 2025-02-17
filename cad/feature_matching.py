from collections import defaultdict

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from occwl.entity_mapper import EntityMapper
import numpy as np
from occwl.io import load_step
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import *


def node_match(node1, node2):
    return node2['type'] == 'any' or node1['type'] == node2['type']


# 定义边匹配的函数
def edge_match(edge1, edge2):
    return edge2['type'] == 'any' or edge1['type'] == edge2['type']


# 检查方向共线
def is_collinear(axis1, axis2, tol=1e-4):
    cross_product = np.cross(axis1, axis2)
    return np.linalg.norm(cross_product) < tol


def is_equal(value1, value2, tol=1e-4):
    return value1 - value2 < tol


class Axis3GraphMatcher(GraphMatcher):
    def semantic_feasibility(self, G1_node, G2_node):
        # 调用默认的节点匹配逻辑
        if not super().semantic_feasibility(G1_node, G2_node):
            return False
        # print(self.mapping)
        if not (
                1 in self.mapping and 2 in self.mapping and 3 in self.mapping and 4 in self.mapping and 5 in self.mapping):
            return True
        node0 = self.mapping[0]
        node1 = self.mapping[1]
        node2 = self.mapping[2]
        node3 = self.mapping[3]
        node4 = self.mapping[4]
        node5 = self.mapping[5]
        node6 = self.mapping[6]
        nodes = [
            self.G1.nodes[node1],
            self.G1.nodes[node3],
            self.G1.nodes[node5]
        ]
        edges = [
            self.G1[node0][node1],
            self.G1[node1][node2],
            self.G1[node2][node3],
            self.G1[node3][node4],
            self.G1[node4][node5],
            self.G1[node5][node6]
        ]
        axis_values = [
            nodes[0]["axis"],
            nodes[2]["axis"],
            nodes[4]["axis"],
            edges[0]["axis"],
            edges[1]["axis"],
            edges[2]["axis"],
            edges[3]["axis"],
            edges[4]["axis"],
            edges[5]["axis"],
        ]
        # 检查所有 axis 是否共线
        reference_axis = axis_values[0]  # 取第一个值作为参考
        for axis in axis_values[1:]:
            if not is_collinear(reference_axis, axis):
                print("no match 1")
                return False  # 匹配失败
        # 检查3号节点的半径最大
        if not (nodes[2]["radius"] > nodes[0]["radius"] and nodes[2]["radius"] > nodes[4]["radius"]):
            print("no match 2")
            return False
        return True

class AocaoGraphMatcher(GraphMatcher):
    def semantic_feasibility(self, G1_node, G2_node):
        # 调用默认的节点匹配逻辑
        if not super().semantic_feasibility(G1_node, G2_node):
            return False
        # print(self.mapping)
        if not (0 in self.mapping and 1 in self.mapping and 2 in self.mapping and 3 in self.mapping and 4 in self.mapping):
            return True
        node1 = self.mapping[1]
        node3 = self.mapping[3]
        node4 = self.mapping[4]

        check_list = [
            self.G1.nodes[node1],
            self.G1.nodes[node3],
            self.G1[node1][node4],
            self.G1[node3][node4]
        ]

        if not is_collinear(check_list[0]["axis"], check_list[2]["axis"]):
            print("no match a1")
            return False  # 匹配失败

        if not is_collinear(check_list[1]["axis"], check_list[3]["axis"]):
            print("no match a3")
            return False  # 匹配失败

        reference_r = check_list[0]["radius"]  # 取第一个值作为参考
        for noe in check_list[1:]:
            if not is_equal(reference_r, noe["radius"]):
                print("no match r")
                return False  # 匹配失败
        return True


def get_axis3_result(graph, subgraph_matches):
    used_faces = []
    param_set = set()
    grouped_results = defaultdict(dict)
    for subgraph in subgraph_matches:
        mapper = {v: k for k, v in subgraph.items()}
        origin = graph[mapper[0]][mapper[1]]["origin"]
        r1 = graph.nodes[mapper[1]]["radius"]
        r2 = graph.nodes[mapper[3]]["radius"]
        r3 = graph.nodes[mapper[5]]["radius"]
        l1 = np.linalg.norm(
            np.array(graph[mapper[0]][mapper[1]]["origin"]) - np.array(graph[mapper[1]][mapper[2]]["origin"]))
        l2 = np.linalg.norm(
            np.array(graph[mapper[2]][mapper[3]]["origin"]) - np.array(graph[mapper[3]][mapper[4]]["origin"]))
        l3 = np.linalg.norm(
            np.array(graph[mapper[4]][mapper[5]]["origin"]) - np.array(graph[mapper[5]][mapper[6]]["origin"]))
        param = tuple([round(origin[0], 3), round(origin[1], 3), round(origin[2], 3),
                       round(r1, 3), round(r2, 3), round(r3, 3),
                       round(l1, 3), round(l2, 3), round(l3, 3)])

        if param in param_set:
            axis3result = grouped_results[param]
            if mapper[1] not in axis3result["faces"]:
                axis3result["faces"].append(mapper[1])
                axis3result["parts"][0]["faces"].append(mapper[1])
            if mapper[2] not in axis3result["faces"]:
                axis3result["faces"].append(mapper[2])
                axis3result["parts"][1]["faces"].append(mapper[2])
            if mapper[3] not in axis3result["faces"]:
                axis3result["faces"].append(mapper[3])
                axis3result["parts"][1]["faces"].append(mapper[3])
            if mapper[4] not in axis3result["faces"]:
                axis3result["faces"].append(mapper[4])
                axis3result["parts"][1]["faces"].append(mapper[4])
            if mapper[5] not in axis3result["faces"]:
                axis3result["faces"].append(mapper[5])
                axis3result["parts"][2]["faces"].append(mapper[5])
        else:
            axis3result = {
                "faces": [mapper[1], mapper[2], mapper[3], mapper[4], mapper[5]],
                "param": [
                    {
                        "name": "length",
                        "value": l1 + l2 + l3
                    }
                ],
                "note": "",
                "parts": [
                    {
                        "name": "轴段1",
                        "faces": [mapper[1]],
                        "param": [
                            {
                                "name": "radius",
                                "value": r1
                            },
                            {
                                "name": "length",
                                "value": l1
                            },
                            {
                                "name": "depth",
                                "value": r2 - r1
                            }
                        ],
                        "note": ""
                    },
                    {
                        "name": "轴段2",
                        "faces": [mapper[2], mapper[3], mapper[4]],
                        "param": [
                            {
                                "name": "radius",
                                "value": r2
                            },
                            {
                                "name": "length",
                                "value": l2
                            },
                            {
                                "name": "depth",
                                "value": 0
                            }
                        ],
                        "note": ""
                    },
                    {
                        "name": "轴段3",
                        "faces": [mapper[5]],
                        "param": [
                            {
                                "name": "radius",
                                "value": r3
                            },
                            {
                                "name": "length",
                                "value": l3
                            },
                            {
                                "name": "depth",
                                "value": r2 - r3
                            }
                        ],
                        "note": ""
                    },
                ]
            }
            grouped_results[param] = axis3result
            param_set.add(param)

    result = {
        "name": "三段阶梯轴",
        "list": []
    }

    for k, values in grouped_results.items():
        faces = values["faces"]
        if any(face in faces for face in used_faces):
            print(f"重复结果: {faces}")
        else:
            used_faces += faces
            print(f"识别到一个<三段阶梯轴>: {faces}")
            values["faces"] = []
            values["faces"].append(values["parts"][0]["faces"])
            values["faces"].append(values["parts"][1]["faces"])
            values["faces"].append(values["parts"][2]["faces"])
            result["list"].append(values)

    return len(result["list"]) > 0, result

def get_aocao_result(graph, subgraph_matches):
    used_faces = []
    param_set = set()
    grouped_results = defaultdict(dict)
    # 查找匹配并输出结果
    for subgraph in subgraph_matches:
        print("Found matching subgraph:", subgraph)
        mapper = {v: k for k, v in subgraph.items()}
        o1 = graph[mapper[1]][mapper[4]]["origin"]
        o3 = graph[mapper[3]][mapper[4]]["origin"]
        r = graph.nodes[mapper[1]]["radius"]
        d = graph[mapper[1]][mapper[2]]["length"]
        l = graph[mapper[2]][mapper[4]]["length"]
        w = r * 2
        param = tuple([round(o1[0], 3), round(o1[1], 3), round(o1[2], 3),
                       round(o3[0], 3), round(o3[1], 3), round(o3[2], 3),
                       round(r, 3), round(d, 3), round(l, 3), round(w, 3)])

        if param in param_set:
            print("inset")
            result = grouped_results[param]
            if mapper[1] not in result["faces"]:
                result["faces"].append(mapper[1])
                result["parts"][1]["faces"].append(mapper[1])
            if mapper[2] not in result["faces"]:
                result["faces"].append(mapper[2])
                result["parts"][2]["faces"].append(mapper[2])
            if mapper[3] not in result["faces"]:
                result["faces"].append(mapper[3])
                result["parts"][1]["faces"].append(mapper[3])
            if mapper[4] not in result["faces"]:
                result["faces"].append(mapper[4])
                result["parts"][0]["faces"].append(mapper[4])
        else:
            print("new param")
            result = {
                "faces": [mapper[1], mapper[2], mapper[3], mapper[4]],
                "param": [
                    {
                        "name": "凹槽深度",
                        "value": d
                    },
                    {
                        "name": "矩形长度",
                        "value": l
                    },
                    {
                        "name": "矩形宽度",
                        "value": w
                    },
                    {
                        "name": "弧面半径",
                        "value": r
                    },
                    {
                        "name": "圆心1",
                        "value": f"{o1[0]},{o1[1]},{o1[2]}"
                    },
                    {
                        "name": "圆心2",
                        "value": f"{o3[0]},{o3[1]},{o3[2]}"
                    }
                ],
                "note": "",
                "parts": [
                    {
                        "name": "凹槽底面",
                        "faces": [mapper[4]],
                        "param": [
                        ],
                        "note": ""
                    },
                    {
                        "name": "圆弧柱面",
                        "faces": [mapper[1], mapper[3]],
                        "param": [
                        ],
                        "note": ""
                    },
                    {
                        "name": "竖直面",
                        "faces": [mapper[2]],
                        "param": [
                        ],
                        "note": ""
                    },
                ]
            }
            grouped_results[param] = result
            param_set.add(param)

    result = {
        "name": "跑道形凹槽",
        "list": []
    }

    for k, values in grouped_results.items():
        faces = values["faces"]
        if any(face in faces for face in used_faces):
            print(f"重复结果: {faces}")
        else:
            used_faces += faces
            print(f"识别到一个<柱面跑道形凹槽>: {faces}")
            values["faces"] = []
            values["faces"].append(values["parts"][0]["faces"])
            values["faces"].append(values["parts"][1]["faces"])
            values["faces"].append(values["parts"][2]["faces"])
            result["list"].append(values)

    return len(result["list"]) > 0, result

def match_features(solid):
    mapper = EntityMapper(solid)
    graph = nx.Graph()
    for face in solid.faces():
        face_idx = mapper.face_index(face)
        graph.add_node(face_idx)
        surf = BRepAdaptor_Surface(face.topods_shape())
        surf_type = surf.GetType()
        if surf_type == GeomAbs_Plane:
            gp_pln = surf.Plane()
            normal = gp_pln.Axis().Direction()
            graph.nodes[face_idx]['type'] = "plain"
            graph.nodes[face_idx]['normal'] = [normal.X(), normal.Y(), normal.Z()]
        elif surf_type == GeomAbs_Cylinder:
            gp_cyl = surf.Cylinder()
            location = gp_cyl.Location()
            axis = gp_cyl.Axis().Direction()
            radius = gp_cyl.Radius()
            graph.nodes[face_idx]['type'] = "circle"
            graph.nodes[face_idx]['radius'] = radius
            graph.nodes[face_idx]['axis'] = [axis.X(), axis.Y(), axis.Z()]
            graph.nodes[face_idx]['origin'] = [location.X(), location.Y(), location.Z()]
        elif surf_type == GeomAbs_Cone:
            graph.nodes[face_idx]['type'] = "cone"
        elif surf_type == GeomAbs_Sphere:
            graph.nodes[face_idx]['type'] = "sphere"
        elif surf_type == GeomAbs_Torus:
            graph.nodes[face_idx]['type'] = "torus"
        else:
            graph.nodes[face_idx]['type'] = "other"

    for edge in solid.edges():
        if not edge.has_curve():
            continue
        connected_faces = list(solid.faces_from_edge(edge))
        if len(connected_faces) == 2:
            left_face, right_face = edge.find_left_and_right_faces(connected_faces)
            if left_face is None or right_face is None:
                continue
            edge_reversed = edge.reversed_edge()
            if not mapper.oriented_edge_exists(edge_reversed):
                continue
            edge_reversed_idx = mapper.oriented_edge_index(edge_reversed)
            left_index = mapper.face_index(left_face)
            right_index = mapper.face_index(right_face)
            graph.add_edge(left_index, right_index)
            curv = BRepAdaptor_Curve(edge.topods_shape())
            curv_type = curv.GetType()
            if curv_type == GeomAbs_Line:
                graph[left_index][right_index]['type'] = "line"
                u_start = curv.FirstParameter()
                u_end = curv.LastParameter()
                start_point = curv.Value(u_start)
                end_point = curv.Value(u_end)
                length = start_point.Distance(end_point)
                graph[left_index][right_index]['length'] = length
            elif curv_type == GeomAbs_Circle:
                gp_circ = curv.Circle()
                location = gp_circ.Location()
                radius = gp_circ.Radius()
                length = gp_circ.Length()
                axis = gp_circ.Axis().Direction()
                graph[left_index][right_index]['type'] = "circle"
                graph[left_index][right_index]['radius'] = radius
                graph[left_index][right_index]['length'] = length
                graph[left_index][right_index]['axis'] = [axis.X(), axis.Y(), axis.Z()]
                graph[left_index][right_index]['origin'] = [location.X(), location.Y(), location.Z()]
            else:
                graph[left_index][right_index]['type'] = "other"

    template_axis3 = nx.Graph()
    template_axis3.add_nodes_from([
        (0, {"type": "any"}),
        (1, {"type": "circle"}),  # 低圆柱面
        (2, {"type": "plain"}),
        (3, {"type": "circle"}),  # 高圆柱面
        (4, {"type": "plain"}),
        (5, {"type": "circle"}),  # 低圆柱面
        (6, {"type": "any"}),
    ])
    template_axis3.add_edges_from([
        (0, 1, {'type': 'circle'}),
        (1, 2, {'type': 'circle'}),
        (2, 3, {'type': 'circle'}),
        (3, 4, {'type': 'circle'}),
        (4, 5, {'type': 'circle'}),
        (5, 6, {'type': 'circle'})
    ])

    template_aocao = nx.Graph()
    template_aocao.add_nodes_from([
        (0, {"type": "any"}),  # 凹槽外面
        (1, {"type": "circle"}),  # 左圆弧柱面
        (2, {"type": "plain"}),  # 竖直面
        (3, {"type": "circle"}),  # 右圆弧柱面
        (4, {"type": "plain"})  # 底面
    ])
    template_aocao.add_edges_from([
        (0, 1, {'type': 'any'}),
        (0, 2, {'type': 'line'}),
        (0, 3, {'type': 'any'}),
        (1, 2, {'type': 'line'}),
        (1, 4, {'type': 'circle'}),
        (2, 3, {'type': 'line'}),
        (2, 4, {'type': 'line'}),
        (3, 4, {'type': 'circle'}),
    ])

    axis3matcher = Axis3GraphMatcher(graph, template_axis3, node_match=node_match, edge_match=edge_match)
    aocaomatcher = AocaoGraphMatcher(graph, template_aocao, node_match=node_match, edge_match=edge_match)

    axis3_matches = axis3matcher.subgraph_isomorphisms_iter()
    aocao_matches = aocaomatcher.subgraph_isomorphisms_iter()
    has_axis3, axis3_result = get_axis3_result(graph, axis3_matches)
    has_aocao, aocao_result = get_aocao_result(graph, aocao_matches)

    result = []
    if has_axis3:
        result.append(axis3_result)
    if has_aocao:
        result.append(aocao_result)
    print(result)
    return result