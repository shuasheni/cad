"""

Class representing a Joint Prediction Set
containing ground truth joints between two bodies
and extended with prediction information

"""

import random
import igl
from pysdf import SDF
import torch.nn.functional as F
from torch_geometric.data import Batch
from joint_predict.joint import joint_axis
import math
import warnings
import json
import numpy as np

from joint_predict.joint.joint_axis import get_vector
from joint_predict.utils import util
import trimesh
from joint_predict.geometry.obj_reader import OBJReader
from joint_predict.utils.util import are_vectors_perpendicular, are_vectors_parallel, point_to_plane_distance


class NewPredictionSet():
    def __init__(
            self,
            dataset_dir,
            g1, g2, joint_graph, model,
            load_bodies=True,
            num_samples=4096,
            seed=None,
            prediction_limit=50,
            g1_id=None,
            g2_id=None
    ):
        self.dataset_dir = dataset_dir
        body_one_name = f"{g1_id}.obj"
        body_two_name = f"{g2_id}.obj"
        self.g1_id = g1_id
        self.g2_id = g2_id

        self.body_one_obj_file = self.dataset_dir / body_one_name
        self.body_two_obj_file = self.dataset_dir / body_two_name

        if load_bodies:
            self.body_one = self.load_obj(self.body_one_obj_file)
            self.body_two = self.load_obj(self.body_two_obj_file)
            self.bodies_loaded = True
            self.body_one_mesh = self.body_one.get_mesh()
            self.body_two_mesh = self.body_two.get_mesh()
        else:
            self.body_one = None
            self.body_two = None
            self.body_one_mesh = trimesh.load(self.body_one_obj_file)
            self.body_two_mesh = trimesh.load(self.body_two_obj_file)
            self.bodies_loaded = False

        bbox1 = self.body_one_mesh.bounds
        bbox2 = self.body_two_mesh.bounds
        self.min_axis = min(np.linalg.norm(bbox1[1] - bbox1[0]), np.linalg.norm(bbox2[1] - bbox2[0]))

        # 接头实体预测结果
        self.prediction_data = self.get_prediction_data(g1, g2, joint_graph, model, prediction_limit, g1_id=g1_id,
                                                        g2_id=g2_id)

        # 对body1，分别在整体体积上采样和在表面上采样，用于计算体积重叠率和面积接触率
        self.num_samples = num_samples
        self.seed = seed
        self.volume_samples = NewPredictionSet.sample_volume_points(self.body_one_mesh, self.num_samples, self.seed)
        self.surface_samples, _ = trimesh.sample.sample_surface(self.body_one_mesh, num_samples)
        # Store the samples as n,4, so they can be easily transformed with a 4x4 matrix
        self.volume_samples = util.pad_pts(self.volume_samples)
        self.surface_samples = util.pad_pts(self.surface_samples)
        # body2应用SDF，用于计算体积重叠率和面积接触率
        self.sdf = SDF(self.body_two_mesh.vertices, self.body_two_mesh.faces)

    def get_meshes(
            self,
            predict_index=0,
            apply_transform=True,
            show_joint_entity_colors=True,
            return_vertex_normals=False,
            body_one_transform=None
    ):
        """
        Load the meshes from the joint file and
        transform based on the joint index
        """
        assert self.bodies_loaded, "Joint set bodies not loaded"
        # Combine the triangles from both bodies
        # into a single list of vertices and face indices
        vertices_list = []
        faces_list = []
        colors_list = []
        edges_ent_list = []
        normals_list = []
        normal_indices_list = []

        # Mesh 1
        m1 = self.get_mesh(
            body=1,
            predict_index=predict_index,
            apply_transform=apply_transform,
            show_joint_entity_colors=show_joint_entity_colors,
            return_vertex_normals=return_vertex_normals,
            transform=body_one_transform
        )

        # Mesh 2
        m2 = self.get_mesh(
            body=2,
            predict_index=predict_index,
            apply_transform=False,
            show_joint_entity_colors=show_joint_entity_colors,
            return_vertex_normals=return_vertex_normals
        )

        if return_vertex_normals:
            v1, f1, c1, e1, n1, ni1 = m1
            v2, f2, c2, e2, n2, ni2 = m2
            normals_list.append(n1)
            normals_list.append(n2)
            normal_indices_list.append(ni1)
            normal_indices_list.append(ni2 + n1.shape[0])
        else:
            v1, f1, c1, e1 = m1
            v2, f2, c2, e2 = m2

        vertices_list.append(v1)
        vertices_list.append(v2)
        faces_list.append(f1)
        faces_list.append(f2 + v1.shape[0])
        colors_list.append(c1)
        colors_list.append(c2)

        if e1 is not None:
            edges_ent_list.append(e1)

        if e2 is not None:
            edges_ent_list.append(e2 + v1.shape[0])

        v = np.concatenate(vertices_list)
        f = np.concatenate(faces_list)
        c = np.concatenate(colors_list)
        e = None
        if len(edges_ent_list) > 0:
            e = np.concatenate(edges_ent_list)

        if return_vertex_normals:
            n = np.concatenate(normals_list)
            ni = np.concatenate(normal_indices_list)
            return v, f, c, e, n, ni
        else:
            return v, f, c, e

    def get_mesh(
            self,
            body=1,
            predict_index=0,
            apply_transform=True,
            show_joint_entity_colors=True,
            return_vertex_normals=False,
            transform=None
    ):
        """
        Get a single mesh with an optional transform based on the joint index
        The body value can be either 1 or 2
        """
        assert self.bodies_loaded, "Joint set bodies not loaded"
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"

        if body == 1:
            solid = self.body_one
        else:
            solid = self.body_two

        c, e = self.get_mesh_colors(
            body,
            predict_index=predict_index,
            show_joint_entity_colors=show_joint_entity_colors
        )

        m = self.load_mesh_from_data(
            solid,
            apply_transform,
            return_vertex_normals=return_vertex_normals,
            transform=transform
        )
        if return_vertex_normals:
            v, f, n, ni = m
            return v, f, c, e, n, ni
        else:
            v, f = m
            return v, f, c, e

    def get_mesh_colors(
            self,
            body=1,
            predict_index=0,
            show_joint_entity_colors=True
    ):
        """
        Get the list of colors for each triangle
        """
        assert self.bodies_loaded, "Joint set bodies not loaded"
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"

        # Color map to use for triangles
        color_map = np.array([
            [0.75, 1.00, 0.75],  # Body 1
            [0.75, 0.75, 1.00],  # Body 2
            [1.00, 0.00, 0.00],  # Red for joint entities
            [1.00, 1.00, 0.00],  # Yellow for joint equivalents entities
        ], dtype=float)

        f_ent = None
        e_ent = None

        if show_joint_entity_colors:
            f_ent, e_ent = self.get_joint_predictions_n(body=body, n=predict_index)
            if f_ent is not None:
                # Set the entities to color index 2
                f_ent[f_ent == 1] = 2

        # Face colors for the triangles
        # that aren't entities
        if body == 1:
            tri_count = self.body_one.get_triangle_count()
            bc = np.zeros(tri_count, dtype=int)
        else:
            tri_count = self.body_two.get_triangle_count()
            bc = np.ones(tri_count, dtype=int)

        # Combine, giving the entities/equivalents priority
        if f_ent is not None:
            fc = np.maximum(f_ent, bc)
        else:
            fc = bc

        mesh_colors = color_map[fc]

        # Edges

        if e_ent is not None:
            ec = e_ent
        else:
            ec = None
        return mesh_colors, ec

    def load_mesh_from_data(
            self,
            solid,
            apply_transform=True,
            return_vertex_normals=False,
            transform=None
    ):
        """
        Load the mesh and transform it
        according to the joint geometry_or_origin
        """
        assert self.bodies_loaded, "Joint set bodies not loaded"

        assert transform is not None or not apply_transform, "Apply transform, but no transform"

        v = solid.get_vertices()
        f = solid.get_triangles()

        if apply_transform and transform is not None:
            # 对所有点应用仿射变换
            v = util.transform_pts_by_matrix(
                v,
                transform
            )

        if return_vertex_normals:
            n = solid.get_normals()
            ni = solid.get_normal_indices()
            if apply_transform and transform is not None:
                # 对法向量应用仿射变换
                rot_mat = np.eye(4)
                rot_mat[:3, :3] = transform[:3, :3]
                n = util.transform_pts_by_matrix(
                    n,
                    rot_mat
                )
            return v, f, n, ni
        else:
            return v, f

    def get_edge_indices(self, body=1):
        """
        Get the B-Rep edge indices for a single body
        with an optional transform based on the joint index
        The body value can be either 1 or 2
        """
        assert self.bodies_loaded, "Joint set bodies not loaded"
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        if body == 1:
            solid = self.body_one
        else:
            solid = self.body_two
        return solid.get_polylines()

    def load_obj(self, obj_file):
        """
        Load the mesh into a data structure containing the B-Rep information
        """
        obj = OBJReader(obj_file)
        return obj.read()

    def get_mesh_edges(self, body=1, apply_transform=False, transform=None):
        """获取网格边以绘制线框"""
        assert body in {1, 2}
        assert transform is not None or not apply_transform, "Apply transform, but no transform"

        if body == 1:
            mesh = self.body_one_mesh
        else:
            mesh = self.body_two_mesh

        if apply_transform and transform is not None:
            mesh = mesh.copy()
            mesh.apply_transform(transform)
        f_roll = np.roll(mesh.faces, -1, axis=1)
        e = np.column_stack((mesh.faces, f_roll)).reshape(-1, 2)
        return mesh.vertices, e

    @staticmethod
    def sample_volume_points(mesh, num_samples=4096, seed=None, sample_surface=True):
        """Sample num_samples random points within the volume of the mesh
           We use fast winding numbers, so we don't need a water tight mesh"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        all_samples = []
        all_sample_count = 0
        if sample_surface:
            # First lets make sure we get some samples on the surfaces
            surface_sample_count = int(num_samples * 0.15)
            surface_samples, surface_sample_face_indices = trimesh.sample.sample_surface(mesh, surface_sample_count)
            # Move the points back to inside the surface
            surface_sample_face_normals = mesh.face_normals[surface_sample_face_indices]
            surface_samples -= surface_sample_face_normals * 0.01
            all_samples.append(surface_samples)
            all_sample_count += surface_samples.shape[0]
        # Next lets try trimesh for internal sample, which requires a watertight mesh
        if mesh.is_watertight:
            if sample_surface:
                trimesh_samples_count = num_samples - surface_samples.shape[0]
            else:
                trimesh_samples_count = num_samples
            trimesh_samples = trimesh.sample.volume_mesh(mesh, trimesh_samples_count)
            if trimesh_samples.shape[0] > 0:
                all_samples.append(trimesh_samples)
                all_sample_count += trimesh_samples.shape[0]
            if all_sample_count == num_samples:
                all_samples_np = np.concatenate(all_samples)
                return all_samples_np[:num_samples, :]

        # We have an open mesh, so fall back to using fast winding numbers
        box_min = mesh.bounds[0]
        box_max = mesh.bounds[1]
        # Loop until we have sufficient samples
        while all_sample_count < num_samples:
            xyz_samples = np.column_stack((
                np.random.uniform(box_min[0], box_max[0], num_samples),
                np.random.uniform(box_min[1], box_max[1], num_samples),
                np.random.uniform(box_min[2], box_max[2], num_samples)
            ))
            wns = igl.fast_winding_number_for_meshes(mesh.vertices, mesh.faces, xyz_samples)
            # Add a small threshold here rather than > 0
            # due to seeing some outside points included
            inside_samples = xyz_samples[wns > 0.01]
            # If we can't generate any volume samples
            # this may be a super thin geometry so break
            if len(inside_samples) == 0:
                break
            all_samples.append(inside_samples)
            all_sample_count += inside_samples.shape[0]

        # We should only need to add additional surface samples if
        # we failed to get volume samples due to a super thin geometry
        if all_sample_count < num_samples and (box_max - box_min).min() < 1e-10:
            # Sample again from the surface
            surface_sample_count = num_samples - all_sample_count
            surface_samples, surface_sample_face_indices = trimesh.sample.sample_surface(mesh, surface_sample_count)
            # Move the points back to inside the surface
            surface_sample_face_normals = mesh.face_normals[surface_sample_face_indices]
            surface_samples -= surface_sample_face_normals * 0.01
            all_samples.append(surface_samples)
            all_sample_count += surface_samples.shape[0]

        # Concat and return num_samples only
        all_samples_np = np.concatenate(all_samples)
        return all_samples_np[:num_samples, :]

    @staticmethod
    def get_network_predictions(g1, g2, joint_graph, model):
        """Get the network predictions"""
        fake_batch = (
            Batch.from_data_list([g1]),
            Batch.from_data_list([g2]),
            Batch.from_data_list([joint_graph]),
        )
        x = model(fake_batch)
        prob = F.softmax(x, dim=0)
        return prob.view(g1.num_nodes, g2.num_nodes)

    def get_prediction_data(self, g1, g2, joint_graph, model, prediction_limit, g1_id=None, g2_id=None):
        """
        生成预测数据
        返回一个可以序列化为json进行缓存的字典
        """
        preds = NewPredictionSet.get_network_predictions(g1, g2, joint_graph, model)

        g1_file = self.dataset_dir / f"{g1_id}.json"
        g2_file = self.dataset_dir / f"{g2_id}.json"

        with open(g1_file, "r", encoding="utf-8") as f:
            g1_graph = json.load(f)
        with open(g2_file, "r", encoding="utf-8") as f:
            g2_graph = json.load(f)
        # Header information for our predictions file
        joint_data = {
            "joint_set": "new_joint",
            "body_one": g1_file.stem,
            "body_two": g2_file.stem,
            "body_one_face_count": g1_graph["properties"]["face_count"],
            "body_one_edge_count": g1_graph["properties"]["edge_count"],
            "body_two_face_count": g2_graph["properties"]["face_count"],
            "body_two_edge_count": g2_graph["properties"]["edge_count"],
            "representation": "graph",
            "prediction_method": "network",
            "predictions": []
        }
        preds_np = preds.detach().numpy()

        # print("entity web_app result")
        # 预测结果降序排列
        g1_preds, g2_preds = np.unravel_index(np.argsort(-preds_np, axis=None), preds_np.shape)
        for g1_pred, g2_pred in zip(g1_preds[:prediction_limit], g2_preds[:prediction_limit]):

            # Prediction value
            pred_value = preds_np[g1_pred, g2_pred]
            g1_node = g1_graph["nodes"][g1_pred]
            g2_node = g2_graph["nodes"][g2_pred]
            g1_type = "BRepFace" if "surface_type" in g1_node else "BRepEdge"
            g2_type = "BRepFace" if "surface_type" in g2_node else "BRepEdge"
            # Use the node features to find the axis

            # print(f"value:{pred_value}, type1:{g1_type}, type2:{g2_type}")
            # print(f"g1_node: {g1_node}, g2_node: {g1_node}")

            pt1, dir1 = joint_axis.find_axis_line(g1_node)
            pt2, dir2 = joint_axis.find_axis_line(g2_node)

            # We don't support all entities e.g. Nurbs
            # So only report those we do support
            if pt1 is not None and pt2 is not None:
                # Cast to regular int for json serialization
                prediction = {
                    "value": float(pred_value),
                    "body_one": {
                        "index": int(g1_pred),
                        "type": g1_type,
                        "origin": pt1,
                        "direction": dir1,
                    },
                    "body_two": {
                        "index": int(g2_pred),
                        "type": g2_type,
                        "origin": pt2,
                        "direction": dir2,
                    }
                }
                joint_data["predictions"].append(prediction)
                # print(prediction)
        # If we don't have any predictions we are out of luck
        if len(joint_data["predictions"]) == 0:
            return None
        # Properties
        joint_data["body_one_properties"] = g1_graph["properties"]
        joint_data["body_two_properties"] = g2_graph["properties"]
        return joint_data

    def get_joint_predictions_n(self, body=1, n=0):
        """
        Get the predictions for joint entities
        Returns an array of per-triangle predictions
        """
        assert self.bodies_loaded, "Joint set bodies not loaded"
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        if body == 1:
            solid = self.body_one
            body_key = "body_one"
        else:
            solid = self.body_two
            body_key = "body_two"
        num_faces = self.prediction_data[f"{body_key}_face_count"]

        if n >= len(self.prediction_data["predictions"]):
            return None, None

        tri_face_indices = solid.get_triangle_face_indices()
        triangles = np.zeros(len(tri_face_indices), dtype=int)
        line = None

        prediction = self.prediction_data["predictions"][n]
        body_pred = prediction[body_key]
        entity_type = body_pred["type"]
        index = body_pred["index"]

        np.set_printoptions(threshold=np.inf)

        if entity_type == "BRepFace":
            mask = (tri_face_indices == index)
            triangles[mask] = 1

        if entity_type == "BRepEdge":
            index -= num_faces
            line = None
            # line = solid.get_polyline(index)

        return triangles, line

    def get_joint_prediction_axis_lines_n(self, body=1, n=0, axis_length_scale_factor=0.35):
        """Get lines representing the predicted axis lines"""
        assert self.bodies_loaded, "Joint set bodies not loaded"
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        if body == 1:
            solid = self.body_one
        else:
            solid = self.body_two

        num_preds = len(self.prediction_data["predictions"])
        if n >= num_preds:
            return None, None

        start_points = np.zeros((1, 3))
        end_points = np.zeros((1, 3))

        origin, direction = self.get_joint_prediction_axis(body, n)
        # Get where the axis intersects the aabb
        # tmax will be the distance to the intersection in the positive direction
        # tmin the distance in the negative direction
        tmin, tmax = self.get_joint_prediction_axis_aabb_intersections(
            body=body,
            prediction_index=n,
            origin=origin,
            direction=direction,
            offset=0
        )
        if tmax is None or math.isinf(tmax) or math.isnan(tmax):
            tmax = 0
        if tmin is None or math.isinf(tmin) or math.isnan(tmin):
            tmin = 0
        # tmax will be very small if we are on a face and pointing away from it
        distance_to_aabb = abs(tmax)
        # So we want to calculate how far to extend beyond the aabb
        # Span across the aabb
        aabb_span = abs(tmin) + abs(tmax)
        # Span across the bounding box xyz
        v_min, v_max = solid.get_bounding_box()
        span = v_max - v_min
        # Add the axis span to the list of spans
        span_list = span.tolist()
        span_list.append(aabb_span)
        # Take the mean of the bbox spans and axis spans
        # To ensure a diagonal axis of really long part doesn't
        # skew the length
        mean_span = np.mean(span_list)
        # The distance beyond the aabb that we want to extend by
        # This is relative to the length of the axis inside the aabb
        distance_beyond_aabb = mean_span * axis_length_scale_factor
        # The total distance of the axis
        axis_length = distance_to_aabb + distance_beyond_aabb
        start_pt = origin
        end_pt = origin + direction * axis_length
        start_points[0] = start_pt
        end_points[0] = end_pt
        return start_points, end_points

    def get_joint_prediction_axis(self, body=1, prediction_index=0):
        """Get the joint axis (origin and direction) for the given prediction"""
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        if body == 1:
            suffix = "one"
        else:
            suffix = "two"
        prediction = self.prediction_data["predictions"][prediction_index]
        # print("prediction:")
        # print(prediction)
        origin = util.vector_to_np(prediction[f"body_{suffix}"]["origin"])[:3]
        direction = util.vector_to_np(prediction[f"body_{suffix}"]["direction"])[:3]
        length = np.linalg.norm(direction)
        if length < 0.00000001:
            return origin, None
        direction = direction / length
        return origin, direction

    def get_joint_prediction_axis_aabb_intersections(self, body=1, prediction_index=0, origin=None, direction=None,
                                                     offset=None):
        """
        Get the distances from the origin along the joint prediction axis
        where the axis intersections with the axis aligned bounding box
        """
        if origin is None and direction is None:
            origin, direction = self.get_joint_prediction_axis(body, prediction_index)
        # Offset so the origin is outside the bounding box
        if offset is None:
            offset = np.max(self.body_one_mesh.extents) + np.max(self.body_two_mesh.extents)
        origin_offset = origin + offset * direction
        if direction is None:
            return None, None
        if body == 1:
            bbox = self.body_one_mesh.bounds
        else:
            bbox = self.body_two_mesh.bounds
        # This will produce NaN's when the components of the direction are 0
        # But this is intentional and is handled downstream in intersect_ray_box()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direction_inverse = 1 / direction
        # Try to find the intersection from either direction
        tmin, tmax = util.intersect_ray_box(bbox[0], bbox[1], origin_offset, direction_inverse)
        if tmin is None or tmax is None or math.isinf(tmin) or math.isinf(tmax):
            origin_offset_inverse = origin + offset * (direction * -1)
            tmin, tmax = util.intersect_ray_box(bbox[0], bbox[1], origin_offset_inverse, direction_inverse)
        return tmin, tmax

    def get_joint_prediction_axis_convex_hull_intersections(self, body=1, prediction_index=0):
        """
        Get the points along the joint prediction axis
        where the axis intersections with the convex hull of the given body
        """
        origin, direction = self.get_joint_prediction_axis(body, prediction_index)
        if body == 1:
            suffix = "one"
        else:
            suffix = "two"

        # Cache the convex_hull
        convex_hull = getattr(self, f"body_{suffix}_convex_hull", None)
        if convex_hull is None:
            mesh = getattr(self, f"body_{suffix}_mesh", None)
            convex_hull = mesh.convex_hull
            setattr(self, f"body_{suffix}_convex_hull", convex_hull)

        # Cache the ray mesh intersector
        ray_mesh = getattr(self, f"body_{suffix}_ray_mesh", None)
        if ray_mesh is None:
            # Slower but more accurate than trimesh.ray.ray_pyembree.RayMeshIntersector
            ray_mesh = trimesh.ray.ray_triangle.RayMeshIntersector(convex_hull)
            setattr(self, f"body_{suffix}_ray_mesh", ray_mesh)

        offset = np.max(self.body_one_mesh.extents) + np.max(self.body_two_mesh.extents)
        origin_offset = origin + offset * direction
        locs, _, _ = ray_mesh.intersects_location([origin_offset], [direction])
        if len(locs) == 0:
            locs, _, _ = ray_mesh.intersects_location([origin_offset], [direction * -1])
        # Using a convex hul this should equal 2
        if len(locs) == 2:
            return locs
        return None

    def get_joint_prediction_indices(self, limit=None):
        """Get the indices of the joint predictions"""
        indices = np.arange(len(self.prediction_data["predictions"]))
        if limit is not None:
            return indices[:limit]
        return indices

    def get_joint_prediction_probabilities(self, limit=None):
        """Get the joint prediction probabilities"""
        probs = np.array([p["value"] for p in self.prediction_data["predictions"]])
        if limit is not None:
            probs = probs[:limit]
        probs /= probs.sum()
        return probs

    def get_joint_prediction_brep_indices(self):
        """
        Get a list of the predicted joint indices of the b-rep entities
        as tuples in the form:
        [ (body_one_index, body_two_index), ... ]
        """
        return [(p["body_one"]["index"], p["body_two"]["index"]) for p in self.prediction_data["predictions"]]

    def get_joint_prediction_brep_index(self, index):
        """
        Get the predicted joint indices of the b-rep entities as a tuple: (body_one_index, body_two_index)
        """
        p = self.prediction_data["predictions"][index]
        return p["body_one"]["index"], p["body_two"]["index"]

    def is_joint_body_rotationally_symmetric(self, body=1, prediction_index=0, joint_axis_direction=None):
        """
        Determine if a joint body has rotational symmetry about it's joint axis
        """
        if joint_axis_direction is None:
            _, joint_axis_direction = self.get_joint_prediction_axis(body, prediction_index)
        if body == 1:
            symmetry_axis = self.body_one_mesh.symmetry_axis
        else:
            symmetry_axis = self.body_two_mesh.symmetry_axis
        if symmetry_axis is None:
            return False
        # Check both directions
        pairs = np.array([
            [symmetry_axis, joint_axis_direction],
            [symmetry_axis, joint_axis_direction * -1]
        ])
        # pairs = np.concatenate((symmetry_axis, joint_axis_direction), axis=1).reshape((, 2, 3))
        angles = trimesh.geometry.vector_angle(pairs)
        angle_threshold = np.deg2rad(1)
        aligned_mask = angles < angle_threshold
        is_aligned = aligned_mask.sum() > 0
        return is_aligned

    def get_joint_prediction_axis_parallel_plain_nomal(self, body=1, prediction_index=0):
        """获取所有平行于轴（法线垂直于轴）的平面的法向量"""
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        if body == 1:
            suffix = "one"
            g_file = self.dataset_dir / f"{self.g1_id}.json"
        else:
            suffix = "two"
            g_file = self.dataset_dir / f"{self.g2_id}.json"
        origin, direction = self.get_joint_prediction_axis(body, prediction_index)
        results = []

        with open(g_file, "r", encoding="utf-8") as f:
            g_graph = json.load(f)

        for node in g_graph["nodes"]:
            if "surface_type" in node and node["surface_type"] == "PlaneSurfaceType":
                f_nomal = get_vector(node, "normal")
                print(f"direction: {direction}")
                print(f"f_nomal: {f_nomal}")

                if are_vectors_perpendicular(direction, f_nomal):
                    results.append(f_nomal)
        return results

    def get_joint_prediction_axis_parallel_plain_nomal(self, body=1, prediction_index=0):
        """获取所有垂直于轴（法线平行于轴）的平面，离所给点的距离"""
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        if body == 1:
            suffix = "one"
            g_file = self.dataset_dir / f"{self.g1_id}.json"
        else:
            suffix = "two"
            g_file = self.dataset_dir / f"{self.g2_id}.json"
        origin, direction = self.get_joint_prediction_axis(body, prediction_index)
        results = []

        with open(g_file, "r", encoding="utf-8") as f:
            g_graph = json.load(f)

        for node in g_graph["nodes"]:
            if "surface_type" in node and node["surface_type"] == "PlaneSurfaceType":
                f_nomal = get_vector(node, "normal")
                f_point = get_vector(node, "origin")
                print(f"direction: {direction}")
                print(f"f_nomal: {f_nomal}")

                if are_vectors_parallel(direction, f_nomal):

                    distance = point_to_plane_distance(origin, f_point, f_nomal)

                    results.append(distance)
        return results

    def get_joint_prediction_axis_parallel_hole_axis(self, body=1, prediction_index=0):
        """获取所有平行于轴的圆柱面，返回其轴与所给轴的距离和坐标偏移"""
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        if body == 1:
            suffix = "one"
            g_file = self.dataset_dir / f"{self.g1_id}.json"
        else:
            suffix = "two"
            g_file = self.dataset_dir / f"{self.g2_id}.json"
        origin, direction = self.get_joint_prediction_axis(body, prediction_index)
        results = []

        with open(g_file, "r", encoding="utf-8") as f:
            g_graph = json.load(f)

        for node in g_graph["nodes"]:
            if "surface_type" in node and node["surface_type"] == "CylinderSurfaceType":
                h_axis = get_vector(node, "axis")
                h_radius = node["radius"]
                print(f"direction: {direction}")
                print(f"h_axis: {h_axis}")

                # orientation = node.Orientation()  # 检查面方向
                # if orientation == TopAbs_INTERNAL:
                #     print("inner")

                if are_vectors_parallel(direction, h_axis):
                    results.append([h_axis, h_radius])
        return results


    def get_degrees_parallel_plains(self, prediction_index=0):
        body_one_nomals = self.get_joint_prediction_axis_parallel_plain_nomal(1, prediction_index)
        body_two_nomals = self.get_joint_prediction_axis_parallel_plain_nomal(2, prediction_index)

        a_normals = np.array(body_one_nomals)
        b_normals = np.array(body_two_nomals)

        # 计算 a_normals 和 b_normals 的模长
        a_norms = np.linalg.norm(a_normals, axis=1)
        b_norms = np.linalg.norm(b_normals, axis=1)

        # 计算点积矩阵，使用广播
        dot_products = np.dot(a_normals, b_normals.T)  # 结果是 len(a_normals) x len(b_normals) 的矩阵

        # 计算夹角的余弦值矩阵
        norms_product = np.outer(a_norms, b_norms)  # 外积，得到相同大小的模长矩阵
        cos_theta = dot_products / norms_product  # 逐元素相除
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 限制在 [-1, 1] 范围内，避免数值误差

        # 计算弧度并转换为角度
        angles_rad = np.arccos(cos_theta)
        angles_deg = np.degrees(angles_rad)

        return angles_deg.tolist()

    def get_distance_perpendicular_plains(self, prediction_index=0):


        return None

    def get_degree_distance_parallel_holes(self, prediction_index=0):
        body_one_holes = self.get_joint_prediction_axis_parallel_hole_axis(1, prediction_index)
        body_two_holes = self.get_joint_prediction_axis_parallel_hole_axis(2, prediction_index)




        return None
