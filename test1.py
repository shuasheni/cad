import networkx as nx
from occwl.entity_mapper import EntityMapper
from occwl.io import load_step
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import *

# 加载文件
file_path = f"C:\\Users\\40896\\Desktop\\data\\joint\\13.step"
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
        graph.nodes[face_idx]['type'] = "plain"
    elif surf_type == GeomAbs_Cylinder:
        graph.nodes[face_idx]['type'] = "circle"
    elif surf_type == GeomAbs_Cone:
        graph.nodes[face_idx]['type'] = "cone"
    elif surf_type == GeomAbs_Sphere:
        graph.nodes[face_idx]['type'] = "sphere"
    elif surf_type == GeomAbs_Torus:
        graph.nodes[face_idx]['type'] = "torus"
    else:
        graph.nodes[face_idx]['type'] = "other"
    print(f"face{face_idx} - {graph.nodes[face_idx]['type']}")