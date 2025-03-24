import networkx as nx
from OCC.Core.IFSelect import IFSelect_ItemsByEntity
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Extend.TopologyUtils import list_of_shapes_to_compound, TopologyExplorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import *
from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper

# 加载文件
file_path = (f"C:\\Users\\40896\\Desktop\\data\\joint\\13.step")

step_reader = STEPControl_Reader()
status = step_reader.ReadFile(file_path)
as_compound=True
verbosity=False
shape = None

if verbosity:
    failsonly = False
    step_reader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity)
    step_reader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity)
transfer_result = step_reader.TransferRoots()
if not transfer_result:
    raise AssertionError("Transfer failed.")
_nbs = step_reader.NbShapes()
if _nbs == 0:
    raise AssertionError("No shape to transfer.")
elif _nbs == 1:  # most cases
    shape = step_reader.Shape(1)
    print("_nbs == 1")
elif _nbs > 1:
    print("Number of shapes:", _nbs)
    shps = []
    # loop over root shapes
    for k in range(1, _nbs + 1):
        new_shp = step_reader.Shape(k)
        if not new_shp.IsNull():
            shps.append(new_shp)
    if as_compound:
        compound, result = list_of_shapes_to_compound(shps)
        if not result:
            print("Warning: all shapes were not added to the compound")
        # return compound
        print("compound")
    else:
        print("Warning, returns a list of shapes.")
        # return shps
        print("shps")

if not isinstance(shape, TopoDS_Compound):
    print("not a TopoDS_Compound.")
    shape, success = list_of_shapes_to_compound([shape])

sl = list(Compound(shape).solids())
print(f"len: {len(sl)}")
solid = sl[1]
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