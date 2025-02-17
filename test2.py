from collections import defaultdict

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from occwl.entity_mapper import EntityMapper
import numpy as np
from occwl.io import load_step
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import *


# 加载文件
file_path = f"C:\\Users\\40896\\Desktop\\data\\zhou\\轴1.step"
solid = load_step(file_path)[0]
mapper = EntityMapper(solid)


# 构建图结构
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
    # print(f"node{face_idx}: {graph.nodes[face_idx]['type']}")

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

        # print(f"edge{left_index},{right_index}: {graph[left_index][right_index]['type']}")

# 创建阶梯轴的模板图
template_aocao = nx.Graph()
template_aocao.add_nodes_from([
    (0, {"type": "any"}), # 凹槽外面
    (1, {"type": "circle"}), # 左圆弧柱面
    (2, {"type": "plain"}), # 竖直面
    (3, {"type": "circle"}), # 右圆弧柱面
    (4, {"type": "plain"}) # 底面
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


# 定义节点匹配的函数
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
# 自定义 GraphMatcher
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

# 使用自定义的图匹配器
aocaomatcher = AocaoGraphMatcher(graph, template_aocao, node_match=node_match,edge_match=edge_match)

subgraph_matches = aocaomatcher.subgraph_isomorphisms_iter()
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

for k, values in grouped_results.items():
    faces = values["faces"]
    if any(face in faces for face in used_faces):
        print(f"重复结果: {faces}")
    else:
        used_faces += faces
        print(f"识别到一个<柱面跑道形凹槽>: {values}")



