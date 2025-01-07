import json
import time
from pathlib import Path
import meshplot as mp
import numpy as np
import torch
from PIL import Image

from joint_predict.joint.joint_environment import JointEnvironment
from joint_predict.joint.new_prediction_set import NewPredictionSet
from joint_predict.utils import util

from joint_predict.datasets.joint_graph_dataset import JointGraphDataset

from joint_predict.joint.joint_prediction_set import JointPredictionSet

from joint_predict.search.search_simplex import SearchSimplex
from joint_predict.train import JointPrediction

root_dir = Path().resolve().parent
mp.website()

# All files dir
data_dir = 'C:\\Users\\40896\\Desktop\\data\\joint'
# The checkpoint file
checkpoint_file = root_dir / "joint_predict/pretrained/paper/last_run_0.ckpt"

dataset = JointGraphDataset(
    root_dir=data_dir,
    split="val",
    label_scheme="Joint,JointEquivalent"
)
joint_sett = {"width": 414, "height": 414, "antialias": True, "scale": 1.5, "background": "#f1f1f1", "fov": 30}
body_sett = {"width": 200, "height": 200, "antialias": True, "scale": 1.5, "background": "#f1f1f1", "fov": 30}


def load_network(checkpoint_file):
    """Load the network"""
    if not checkpoint_file.exists():
        print("Checkpoint file does not exist")
        return None
    model = JointPrediction.load_from_checkpoint(
        checkpoint_file,
        map_location=torch.device("cpu")  # Just use the CPU
    )
    return model


model = load_network(checkpoint_file)


def show_joint_gt(jps, joint_n=0):
    """
        显示真实的装配体模型
    """
    v, f, c, e = jps.get_meshes(
        joint_index=joint_n,
        show_joint_entity_colors=False,
        show_joint_equivalent_colors=False,
    )
    p = mp.plot(v, f, c=c, shading=joint_sett);
    return p


def show_joint_pred(jps, transform):
    """
        显示预测的装配体模型
    """
    v, f, c, e, n, ni = jps.get_meshes(
        apply_transform=True,
        body_one_transform=transform,
        show_joint_entity_colors=False,
        return_vertex_normals=True
    )
    p = mp.plot(v, f, c=c, shading=joint_sett);
    return p


def show_body_gt(jps, body, joint_n=0):
    """
        显示真实装配的零件模型
    """
    v, f, c, e = jps.get_mesh(body=body, show_joint_equivalent_colors=False)
    p = mp.plot(v, f, c=c, shading=body_sett)

    # Get the joint axis line and draw it
    start_pt, end_pt = jps.get_joint_axis_line(body=body, joint_index=joint_n)
    p.add_lines(start_pt, end_pt, shading={"line_color": "green"});
    p.add_points(start_pt, shading={"point_color": "green", "point_size": 1});
    return p


def show_body_pred_n(jps, body, n=0):
    """
        显示预测装配的零件模型
    """
    v, f, c, e = jps.get_mesh(
        body=body,
        predict_index=n,
        apply_transform=False
    )

    p = mp.plot(v, f, c=c, shading=body_sett)

    # 预测的边渲染红色
    if e is not None:
        p.add_edges(v, e, shading={"line_color": "red"})

    # 预测的接头轴原点+方向线渲染绿色
    start_pts, end_pts = jps.get_joint_prediction_axis_lines_n(body=body, n=n)
    p.add_lines(start_pts, end_pts, shading={"line_color": "green"})
    p.add_points(start_pts, shading={"point_color": "green", "point_size": 1})
    return p


def predict_new_joint(g1_id, g2_id, n=0, imports=True):
    g1, g1d, face_count1, edge_count1, g1_json_file = dataset.load_graph_body(g1_id, [])
    g2, g2d, face_count2, edge_count2, g2_json_file = dataset.load_graph_body(g2_id, [])

    label_matrix = torch.zeros((face_count1 + edge_count1, face_count2 + edge_count2), dtype=torch.long)

    joint_graph = dataset.make_joint_graph(g1, g2, label_matrix)

    jps = NewPredictionSet(
        Path('C:\\Users\\40896\\Desktop\\data\\joint\\default.json').parent,
        g1, g2, joint_graph,
        model,
        g1_id=g1_id,
        g2_id=g2_id
    )

    search1 = SearchSimplex(
        eval_method="all",
        optimize_method="Nelder-Mead-4"
    )

    transform1, result,_,_ = search1.search_single(jps, n)
    off_limit = search1.cache[n]["offset_limit"]

    prediction = jps.prediction_data["predictions"][n]

    with open("templates/predict_new_joint.html", "w", encoding='utf-8') as f:
        f.write("<div style=\"width: 420px;\">\n")
        f.write(f"<div style=\"border: green 3px solid;height:444px\"> top-{n}预测结果\n")
        f.write(show_joint_pred(jps, transform1).to_html(imports=imports, html_frame=False))
        f.write("</div>\n")
        f.write("<div style=\"display: flex;width: 420px;margin-top: 10px\">\n")
        f.write(f"<div style=\"border: green 3px solid;height:230px\"> {g1_id}\n")
        f.write(show_body_pred_n(jps, 1, n).to_html(imports=False, html_frame=False))
        f.write("</div>\n")
        f.write(f"<div style=\"border: green 3px solid;margin-left: 10px;height:230px\"> {g2_id}\n")
        f.write(show_body_pred_n(jps, 2, n).to_html(imports=False, html_frame=False))
        f.write("</div>\n")
        f.write("</div>\n")
        f.write("</div>\n")
        f.close()

    return jps.prediction_data["predictions"][:50], transform1, result.x, off_limit, prediction, face_count1, face_count2


def predict_exist_joint(joint_id, n=0, joint_n=0, sc=4096):
    joint_file = f'C:\\Users\\40896\\Desktop\\data\\joint\\{joint_id}.json'

    with open(joint_file, encoding="utf8") as f:
        joint_data = json.load(f)

    g1_id = joint_data["body_one"]
    g2_id = joint_data["body_two"]
    g1, g1d, face_count1, edge_count1, g1_json_file = dataset.load_graph_body(joint_data["body_one"], [])
    g2, g2d, face_count2, edge_count2, g2_json_file = dataset.load_graph_body(joint_data["body_two"], [])

    label_matrix = dataset.get_label_matrix(
        joint_data,
        g1, g2,
        g1d, g2d,
        face_count1, face_count2,
        edge_count1, edge_count2
    )

    joint_graph = dataset.make_joint_graph(g1, g2, label_matrix)
    jps = JointPredictionSet(
        Path(joint_file),
        g1, g2, joint_graph,
        model,
        new_joint=False
    )

    print(jps.prediction_data["predictions"][0])

    joints_data = jps.joint_data["joints"]

    with open(f"templates/predict_exist_joint.html", "w", encoding='utf-8') as f:
        f.write("<div style=\"width: 420px;margin-right:10px\">\n")
        f.write(f"<div style=\"border: green 3px solid;height:444px\"> 实际装配体\n")
        f.write(show_joint_gt(jps, joint_n).to_html(imports=True, html_frame=False))
        f.write("</div>\n")
        f.write("<div style=\"display: flex;width: 420px;margin-top: 10px\">\n")
        f.write(f"<div style=\"border: green 3px solid;height:230px\"> {g1_id}\n")
        f.write(show_body_gt(jps, 1, joint_n).to_html(imports=False, html_frame=False))
        f.write("</div>\n")
        f.write(f"<div style=\"border: green 3px solid;margin-left: 10px;height:230px\"> {g2_id}\n")
        f.write(show_body_gt(jps, 2, joint_n).to_html(imports=False, html_frame=False))
        f.write("</div>\n")
        f.write("</div>\n")
        f.write("</div>\n")
        f.close()

    prediction_data, p_transform,_,_,_,_,_ = predict_new_joint(g1_id, g2_id, n, False)

    num_joints = len(jps.joint_data["joints"])
    gt_transforms = np.zeros((num_joints, 4, 4))

    for joint_index, joint in enumerate(jps.joint_data["joints"]):
        gt_transform = util.transform_to_np(joint["geometry_or_origin_one"]["transform"])
        gt_transforms[joint_index] = gt_transform

    best_iou1,best_iou2, best_cd = JointEnvironment.evaluate_vs_gt(jps, p_transform, iou=True, cd=True, num_samples=4096)

    return g1_id, g2_id, prediction_data, max(best_iou1, best_iou2), best_cd, p_transform, joints_data
