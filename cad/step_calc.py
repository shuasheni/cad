import os
import json
from pathlib import Path
import numpy as np
from OCC.Core.BRep import BRep_Builder
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.gp import gp_Trsf, gp_Pnt, gp_Ax2, gp_Dir, gp_Vec, gp_Mat, gp_Ax1

from joint_predict.utils import util


def invert_affine_transform(matrix):
    trsf = gp_Trsf()
    # print(matrix)
    # 使用 SetValues 设置 4x4 矩阵
    trsf.SetValues(
        matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3],
        matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3],
        matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3]
    )

    itrsf = trsf.Inverted()

    imatrix = np.eye(4)

    for i in range(3):
        for j in range(4):
            imatrix[i, j] = itrsf.Value(i + 1, j + 1)

    imatrix[3, 3] = 1
    return imatrix

def apply_affine_transform(shape, matrix):
    trsf = gp_Trsf()

    # 使用 SetValues 设置 4x4 矩阵
    trsf.SetValues(
        matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3] * 10,
        matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3] * 10,
        matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3] * 10
    )


    transform_api = BRepBuilderAPI_Transform(trsf)
    transform_api.Perform(shape)
    shape1 = transform_api.ModifiedShape(shape)


    return shape1

# 加载STEP文件
def load_step_file(step_file):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file)

    if status == 1:  # STEPControl_Reader returns 1 if successful
        step_reader.TransferRoots()
        shape = step_reader.OneShape()
        return shape
    else:
        raise Exception("Error: Unable to read STEP file")


# 计算形状的体积
def calculate_shape_volume(shape):
    props = GProp_GProps()
    brepgprop_VolumeProperties(shape, props)
    return props.Mass()


# 计算两个形状的重叠体积
def calculate_overlap_volume(shape1, shape2):
    common_shape = BRepAlgoAPI_Common(shape1, shape2)
    if common_shape.IsDone():
        overlap_shape = common_shape.Shape()
        return calculate_shape_volume(overlap_shape)
    else:
        raise Exception("Error: Failed to compute common volume")


# 计算体积重叠率
def calculate_volume_overlap_ratio(shape1, shape2):
    volume1 = calculate_shape_volume(shape1)
    volume2 = calculate_shape_volume(shape2)

    overlap_volume = calculate_overlap_volume(shape1, shape2)

    # 体积重叠率：交集体积 / 较小的原始体积
    smaller_volume = min(volume1, volume2)
    overlap_ratio = overlap_volume / smaller_volume

    return overlap_volume, overlap_ratio



def calculate_volume_overlap_ratio_transform(step1, step2, affine_matrix):
    step_file1 = f'C:\\Users\\40896\\Desktop\\data\\joint\\{step1}.step'
    step_file2 = f'C:\\Users\\40896\\Desktop\\data\\joint\\{step2}.step'

    overlap_ratio = -1

    if os.path.exists(step_file1) and os.path.exists(step_file2):
        # 加载STEP文件
        shape1 = load_step_file(step_file1)
        shape2 = load_step_file(step_file2)

        # 对shape1应用仿射变换
        tshape1 = apply_affine_transform(shape1, affine_matrix)

        overlap_volume, overlap_ratio = calculate_volume_overlap_ratio(tshape1, shape2)

        print(f"重叠体积: {overlap_volume}")
        print(f"重叠率: {overlap_ratio}")
    return overlap_ratio


def write_step_file(shape, file_path):
    writer = STEPControl_Writer()
    writer.Transfer(shape)
    writer.Write(file_path)


def merge_shapes(shape1, shape2):
    # Create a compound shape to hold the merged shapes
    compound = TopoDS_Compound()
    BRep_Builder().MakeCompound(compound)

    # Add shapes to the compound
    BRep_Builder().Add(compound, shape1)
    BRep_Builder().Add(compound, shape2)

    return compound


def output_joint_json(body1_id, body2_id, transform1, off, angle, p_n, prediction, path):
    print(f"prediction: {prediction}")
    body1_pred = prediction["body_one"]
    entity_type1 = body1_pred["type"]
    index1 = body1_pred["index"]

    body2_pred = prediction["body_two"]
    entity_type2 = body2_pred["type"]
    index2 = body2_pred["index"]

    origin1 = util.vector_to_np(body1_pred["origin"])
    origin1[3] = 1.0
    direction1 = util.vector_to_np(body1_pred["direction"])
    print(f"origin1: {origin1}")
    print(f"transform1: {transform1}")
    origin1 = np.dot(transform1, origin1)
    print(f"origin1t: {origin1}")
    length = np.linalg.norm(direction1)
    direction1 = direction1 / length
    direction1 = np.dot(transform1, direction1)


    origin2 = util.vector_to_np(body2_pred["origin"])
    direction2 = util.vector_to_np(body2_pred["direction"])
    length = np.linalg.norm(direction2)
    direction2 = direction2 / length

    x_axis = transform1[:3, 0]
    y_axis = transform1[:3, 1]
    z_axis = transform1[:3, 2]
    origin_t = transform1[:3, 3]

    json_joint = {
        "body_one": body1_id,
        "body_two": body2_id,
        "joints": [
            {
                "name": "SavedJoint",
                "type": "Joint",
                "joint_motion": {
                    "joint_type": "SaveJointType"
                },
                "offset": {
                    "type": "ModelParameter",
                    "value": off,
                    "name": "offset",
                    "role": "alignOffsetZ"
                },
                "angle": {
                    "type": "ModelParameter",
                    "value": angle,
                    "name": "angle",
                    "role": "alignAngle"
                },
                "is_flipped": False,
                "geometry_or_origin_one": {
                    "entity_one": {
                        "type": entity_type1,
                        # "surface_type": "PlaneSurfaceType",
                        # "index": 91,
                        "body": body1_id,
                        "index": index1
                    },
                    "transform": {
                        "origin": {
                            "type": "Point3D",
                            "x": origin_t[0],
                            "y": origin_t[1],
                            "z": origin_t[2],
                        },
                        "x_axis": {
                            "type": "Vector3D",
                            "x": x_axis[0],
                            "y": x_axis[1],
                            "z": x_axis[2],
                            "length": 1.0
                        },
                        "y_axis": {
                            "type": "Vector3D",
                            "x": y_axis[0],
                            "y": y_axis[1],
                            "z": y_axis[2],
                            "length": 1.0
                        },
                        "z_axis": {
                            "type": "Vector3D",
                            "x": z_axis[0],
                            "y": z_axis[1],
                            "z": z_axis[2],
                            "length": 1.0
                        }
                    },
                    "axis_line": {
                        "origin": {
                            "type": "Point3D",
                            "x": origin1[0],
                            "y": origin1[1],
                            "z": origin1[2]
                        },
                        "direction": {
                            "type": "Vector3D",
                            "x": direction1[0],
                            "y": direction1[1],
                            "z": direction1[2],
                            "length": 1.0
                        }
                    },
                    "entity_one_equivalents": []
                },
                "geometry_or_origin_two": {
                    "entity_one": {
                        "type": entity_type2,
                        # "surface_type": "PlaneSurfaceType",
                        # "index": 91,
                        "body": body2_id,
                        "index": index2
                    },
                    "transform": {
                        "origin": {
                            "type": "Point3D",
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0
                        },
                        "x_axis": {
                            "type": "Vector3D",
                            "x": 1.0,
                            "y": 0.0,
                            "z": 0.0,
                            "length": 1.0
                        },
                        "y_axis": {
                            "type": "Vector3D",
                            "x": 0.0,
                            "y": 1.0,
                            "z": 0.0,
                            "length": 1.0
                        },
                        "z_axis": {
                            "type": "Vector3D",
                            "x": 0.0,
                            "y": 0.0,
                            "z": 1.0,
                            "length": 1.0
                        }
                    },
                    "axis_line": {
                        "origin": {
                            "type": "Point3D",
                            "x": origin2[0],
                            "y": origin2[1],
                            "z": origin2[2]
                        },
                        "direction": {
                            "type": "Vector3D",
                            "x": direction2[0],
                            "y": direction2[1],
                            "z": direction2[2],
                            "length": 1.0
                        }
                    },
                    "entity_one_equivalents": []
                },
            }
        ],
        "contacts":[]
    }


    with open(Path(path).joinpath(f"{body1_id}_{body2_id}_{p_n}.json"), 'w') as f:
        json.dump(json_joint, f, indent=4)
    return None

def output_step_transform(step1, step2, affine_matrix, path):
    step_file1 = f'C:\\Users\\40896\\Desktop\\data\\joint\\{step1}.step'
    step_file2 = f'C:\\Users\\40896\\Desktop\\data\\joint\\{step2}.step'

    print(f"affine_matrix: {affine_matrix}")

    if os.path.exists(step_file1) and os.path.exists(step_file2):
        # 加载STEP文件
        shape1 = load_step_file(step_file1)
        shape2 = load_step_file(step_file2)

        # 对shape1应用仿射变换
        tshape1 = apply_affine_transform(shape1, affine_matrix)

        merged_shape = merge_shapes(tshape1, shape2)

        writer = STEPControl_Writer()
        writer.Transfer(merged_shape, STEPControl_AsIs)
        status = writer.Write(path)
        if status != 1:
            print("Error writing file")
            exit()

# output_step_transform("HBL2023038.00.01", "HBL2023038.00.02", "0", "out1.step")

# # 示例使用
# if __name__ == "__main__":
#     # STEP文件路径
#     step_file1 = "your_step_file_1.step"
#     step_file2 = "your_step_file_2.step"
#
#     if os.path.exists(step_file1) and os.path.exists(step_file2):
#         # 加载STEP文件
#         shape1 = load_step_file(step_file1)
#         shape2 = load_step_file(step_file2)
#
#         affine_matrix = np.array([
#             [1, 0, 0, 10],  # 平移10单位 along x-axis
#             [0, 1, 0, 20],  # 平移20单位 along y-axis
#             [0, 0, 1, 30],  # 平移30单位 along z-axis
#             [0, 0, 0, 1]  # 齐次坐标
#         ])

