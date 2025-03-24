from pathlib import Path

from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.gp import gp_Cylinder, gp_Pln
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
import numpy as np
import networkx as nx
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Edge
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.gp import gp_Pnt, gp_Vec
import math

def calculate_dihedral_angle_e(face1: TopoDS_Face, face2: TopoDS_Face, edge: TopoDS_Edge):
    """
    计算两个面在共享边上的二面角
    :param face1: 第一个面
    :param face2: 第二个面
    :param edge: 共享的边
    :return: 二面角（弧度）
    """
    # 1. 获取边的几何表示（曲线）和参数范围
    curve, umin, umax = BRep_Tool.Curve(edge)

    # 2. 计算边的中点参数
    mid_param = (umin + umax) / 2.0
    mid_point = curve.Value(mid_param)

    # 3. 获取面1在边中点的法向量
    props1 = BRepLProp_SLProps(face1, 1, 1e-6)
    u1, v1 = BRep_Tool.Parameters(face1, mid_point)
    props1.SetParameters(u1, v1)
    normal1 = props1.Normal()

    # 4. 获取面2在边中点的法向量
    props2 = BRepLProp_SLProps(face2, 1, 1e-6)
    u2, v2 = BRep_Tool.Parameters(face2, mid_point)
    props2.SetParameters(u2, v2)
    normal2 = props2.Normal()

    # 5. 计算法向量夹角（二面角）
    dot = normal1.Dot(normal2)
    mag1 = normal1.Magnitude()
    mag2 = normal2.Magnitude()
    angle_rad = math.acos(dot / (mag1 * mag2))

    return angle_rad

def load_step_file(file_path):
    """读取STEP文件并返回形状"""
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)
    if status != 1:
        raise ValueError("STEP文件读取失败")
    reader.TransferRoots()
    return reader.Shape()


def calculate_dihedral_angle(face1, face2):
    """简化二面角计算（实际需完整几何计算）"""

    calculate_dihedral_angle_e(face1,face2,)

    return 150  # 假设凹边角度>150度


def build_aag(shape):
    """构建属性邻接图（AAG）"""
    G = nx.Graph()
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_dict = {}

    # 添加所有面为节点
    idx = 0
    while face_explorer.More():
        face = face_explorer.Current()
        surf = BRepAdaptor_Surface(face).GetType()
        face_type = "Cylinder" if surf == gp_Cylinder else "Plane"  # 简化分类
        print(f"f{idx}: {face_type}")
        G.add_node(idx, type=face_type)
        face_dict[idx] = face
        idx += 1
        face_explorer.Next()

    # 添加邻接边及凹凸属性
    for i in face_dict:
        for j in face_dict:
            if i >= j:
                continue
            if are_faces_adjacent(face_dict[i], face_dict[j]):
                angle = calculate_dihedral_angle(face_dict[i], face_dict[j])
                edge_type = "Concave" if angle > 150 else "Convex"
                print(f"e{i},{j}: {edge_type}")
                G.add_edge(i, j, type=edge_type)
    return G


def are_faces_adjacent(face1, face2):
    """判断两个面是否邻接（简化实现）"""
    # 实际需通过共享边判断，此处返回True仅用于演示
    return True


def detect_holes(aag_graph):
    """识别孔特征"""
    holes = []
    # 查找所有圆柱面节点
    cylinder_nodes = [n for n, attr in aag_graph.nodes(data=True) if attr['type'] == 'Cylinder']

    for node in cylinder_nodes:
        neighbors = list(aag_graph.neighbors(node))
        concave_edges = 0
        # 检查邻接边是否均为凹边
        for neighbor in neighbors:
            edge_data = aag_graph.get_edge_data(node, neighbor)
            if edge_data['type'] == 'Concave':
                concave_edges += 1
        # 假设孔至少有两个凹边连接平面
        if concave_edges >= 2:
            holes.append(node)
    return holes


if __name__ == "__main__":
    # lb_dir = 'C:\\Users\\40896\\Desktop\\data\\feather\\label'
    # dt_dir = 'C:\\Users\\40896\\Desktop\\data\\feather\\step'
    # pattern = "*_*_*_[1-9].json"
    # list = [f.name for f in Path(lb_dir).glob(pattern)]
    #
    # for name in list[:100]:
    #     graph_json_file = dt_dir / name
    #     id = name.rstrip('.json')


    # 1. 加载STEP文件
    shape = load_step_file("00096860.stp")

    # 2. 构建AAG
    aag = build_aag(shape)

    # 3. 识别孔特征
    hole_nodes = detect_holes(aag)

    print(f"识别到孔特征节点: {hole_nodes}")