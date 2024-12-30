import sys
import math
import warnings
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
import igl
import trimesh

from joint_predict.utils import util
from joint_predict.utils import plot_util
from joint_predict.joint.joint_prediction_set import JointPredictionSet
import pyvista as pv

# plotter = None
plotter_index = 0

type_index = 1
def visualize_3d(volume_samples, body_two_mesh, cost, overlap, contact_area, distance, od, name=None):
    global plotter_index

    plotter = pv.Plotter()

    # 物体1的采样点（volume_samples）是一个点云
    # 将采样点添加到图中
    points_cloud = pv.PolyData(volume_samples)

    plotter.add_mesh(points_cloud, color="red", point_size=3, render_points_as_spheres=True,
                     label='Object 1 Samples')
    # if(mi is not None):
    #     points_cloud2 = pv.PolyData([volume_samples[mi]])
    #     plotter.add_mesh(points_cloud2, color="yellow", point_size=10, render_points_as_spheres=True,
    #                  label='Object 2 Samples')

    # 物体2的几何体（body_two_mesh）
    # 从 trimesh 转换为 PyVista 格式
    body_two_faces = np.hstack((np.full((body_two_mesh.faces.shape[0], 1), 3), body_two_mesh.faces))  # 三角形面
    body_two_pv = pv.PolyData(body_two_mesh.vertices, body_two_faces)

    # 添加物体2的几何体到图中
    plotter.add_mesh(body_two_pv, color="blue", opacity=0.3, show_edges=True, label='Object 2 Mesh')

    # 计算物体2的法线
    body_two_pv.compute_normals(cell_normals=True, point_normals=False, inplace=True)

    # 添加法线的可视化（使用箭头来表示法线方向）
    glyphs = body_two_pv.glyph(orient='Normals', scale=False, factor=0.1)
    plotter.add_mesh(glyphs, color="green", label='Object 2 Normals')

    # 设置视角
    plotter.view_isometric()

    # 显示图例和坐标轴
    plotter.add_legend()
    plotter.show_axes()

    plotter.add_text(f"cost={cost},overlap={overlap},contact={contact_area}, distance={distance},od={od},name={name}",
                     position="upper_left", font_size=12,
                     color='white')

    # 渲染和显示图像
    plotter.show(interactive_update=True, auto_close=False)
    #
    if name is not None:
        plotter.screenshot(f"..\\output\\{name}.png")
        # print("screenshot")
    # else:
    #     plotter.screenshot(f"..\\output\\plotter{plotter_index}.png")
    #     print("screenshot")
    #     plotter_index += 1

    plotter.close()

    # if name is None or name != "Best all":
    #     plotter.close()


class JointEnvironment():

    @staticmethod
    def get_transform_from_parameters(
            jps,
            prediction_index=0,
            offset=0.0,
            rotation_in_degrees=0.0,
            flip=False,
            align_mat=None,
            origin2=None,
            direction2=None
    ):
        """Get a transform from a set of parameters"""
        aff_mat = np.eye(4)

        # ALIGN AXES
        if align_mat is None:
            align_mat, origin2, direction2 = JointEnvironment.get_joint_alignment_matrix(jps, prediction_index)
        aff_mat = align_mat @ aff_mat

        # ROTATION
        rot_mat = JointEnvironment.get_rotation_parameter_matrix(rotation_in_degrees, origin2, direction2)
        aff_mat = rot_mat @ aff_mat

        # OFFSET
        offset_mat = JointEnvironment.get_offset_parameter_matrix(offset, origin2, direction2, flip)
        aff_mat = offset_mat @ aff_mat

        return aff_mat

    @staticmethod
    def get_joint_alignment_matrix(jps, prediction_index=0):
        """
        Given a prediction index, get the affine matrix (4x4)
        that aligns the axis of body one with the axis of body 2
        """
        origin1, direction1 = jps.get_joint_prediction_axis(1, prediction_index)
        origin2, direction2 = jps.get_joint_prediction_axis(2, prediction_index)

        translation = origin2 - origin1
        # The rotation between the two axis directions
        # Ignore "Optimal rotation is not uniquely or poorly defined" warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rotation, _ = Rotation.align_vectors(direction2.reshape(1, -1), direction1.reshape(1, -1))
        aff_mat = np.eye(4)
        aff_mat[:3, :3] = rotation.as_matrix()
        # rotate around the given origin
        aff_mat[:3, 3] = origin1 - np.dot(aff_mat[:3, :3], origin1)
        # translate from the origin of body 2's entity
        aff_mat[:3, 3] += translation
        return aff_mat, origin2, direction2

    @staticmethod
    def get_rotation_parameter_matrix(rotation_in_degrees, origin, direction):
        """
        Get an affine matrix (4x4) to apply the rotation parameter about the provided joint axis
        """
        rotation_in_radians = np.deg2rad(rotation_in_degrees)
        # We do this manually, in case we want to move to torch
        # later on to make this differentiable
        # the below code is similar to calling:
        # rot_mat = Rotation.from_rotvec(rotation_in_radians * direction)
        x, y, z = direction
        c = math.cos(rotation_in_radians)
        s = math.sin(rotation_in_radians)
        C = 1 - c
        xs = x * s
        ys = y * s
        zs = z * s
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC
        aff_mat = np.array([
            [x * xC + c, xyC - zs, zxC + ys, 0],
            [xyC + zs, y * yC + c, yzC - xs, 0],
            [zxC - ys, yzC + xs, z * zC + c, 0],
            [0, 0, 0, 1]
        ])
        # rotation around the origin
        aff_mat[:3, 3] = origin - np.dot(aff_mat[:3, :3], origin)
        return aff_mat

    @staticmethod
    def get_offset_parameter_matrix(offset, origin, direction, flip=False):
        """
        Get an affine matrix (4x4) to apply the offset parameter
        """
        # The offset gets applied first
        aff_mat = np.eye(4)
        aff_mat[:3, 3] = direction * offset
        # Then if we have flip selected
        # we reflect at the origin normal to the direction
        if flip:
            # Reflection matrix
            aff_mat[:3, :3] = np.eye(3) - 2 * np.outer(direction, direction)
            # Normalized vector, vector divided by Euclidean (L2) norm
            normal = direction.squeeze()
            normal = normal / math.sqrt((normal ** 2).sum())
            # Flip at the origin point
            aff_mat[:3, 3] += (2.0 * np.dot(origin, normal)) * normal
        return aff_mat

    @staticmethod
    def evaluate(jps, transform, eval_method=None, name=None, len=None):
        volume_samples = util.transform_pts_by_matrix(jps.volume_samples, transform)
        surface_samples = util.transform_pts_by_matrix(jps.surface_samples, transform)
        volume1, volume2 = jps.prediction_data["body_one_properties"]["volume"], \
            jps.prediction_data["body_two_properties"]["volume"]
        area1, area2 = jps.prediction_data["body_one_properties"]["area"], jps.prediction_data["body_two_properties"][
            "area"]

        # Default method
        if eval_method == "default" or eval_method is None:
            overlap1, sdf_results = JointEnvironment.calculate_overlap(jps.sdf, volume_samples)
            contact_area1, _ = JointEnvironment.calculate_contact_area(jps.sdf, surface_samples, max_contact=1.0)
            overlap2 = overlap1 * volume1 / volume2
            contact_area2 = contact_area1 * area1 / area2
            overlap = np.clip(max(overlap1, overlap2), 0, 1)
            contact_area = np.clip(max(contact_area1, contact_area2), 0, 1)

            # Penalize overlap by zeroing out the contact area when we have overlap
            if overlap > 0.1:
                cost = overlap
            else:
                cost = overlap - 10 * contact_area
            return cost, overlap, contact_area

        elif eval_method == "all":
            overlap1, sdf_results = JointEnvironment.calculate_overlap(jps.sdf, volume_samples,
                                                                       threshold=min(jps.min_axis * 0.005, 0.05))
            contact_area1, sdf_results2 = JointEnvironment.calculate_contact_area(jps.sdf, surface_samples,
                                                                                  threshold=0.05, max_contact=1.0)
            overlap2 = overlap1 * volume1 / volume2
            contact_area2 = contact_area1 * area1 / area2
            c_overlap = np.clip(max(overlap1, overlap2), 0, 1)
            c_contact = np.clip(max(contact_area1, contact_area2), 0, 1)
            distance, overlap_depth = JointEnvironment.calculate_distance(sdf=jps.sdf, sdf_results=sdf_results2,
                                                                          samples=surface_samples)

            c_distance = 1 - math.exp(-distance)
            c_depth = overlap_depth / jps.min_axis

            # 容忍0.05体积重叠
            if c_overlap > 0.02:
                c_contact = 0

            cost = c_overlap + c_depth + c_distance - 10 * c_contact

            return cost, c_overlap, c_contact

    @staticmethod
    def evaluate_vs_gt(jps, pred_transform, iou=True, cd=False, num_samples=4096, symmetrical=False):
        """
        Evaluate the given transform against the ground truth
        We do this for body one only as body two is static
        """
        if not iou and not cd:
            return None, None

        # Loop over all joints and check iou against each
        num_joints = len(jps.joint_data["joints"])
        gt_transforms = np.zeros((num_joints, 4, 4))
        for joint_index, joint in enumerate(jps.joint_data["joints"]):
            gt_transform = util.transform_to_np(joint["geometry_or_origin_one"]["transform"])
            gt_transforms[joint_index] = gt_transform

        best_iou1 = 0
        best_iou2 = 0
        best_cd = sys.float_info.max
        if iou:
            best_iou1, best_iou2 = JointEnvironment.calculate_iou_batch(jps, pred_transform, gt_transforms, num_samples=num_samples, symmetrical=symmetrical)
        if cd:
            best_cd = JointEnvironment.calculate_cd_batch(jps, pred_transform, gt_transforms, num_samples=num_samples)

        if iou and cd:
            return best_iou1, best_iou2, best_cd
        elif iou:
            return best_iou1, best_iou2, None
        elif cd:
            return None, None, best_cd

    @staticmethod
    def calculate_overlap(
            sdf=None,
            samples=None,
            threshold=0.01,
            sdf_results=None
    ):
        """
        Calculate the overlap using samples and an sdf
        """
        num_samples = samples.shape[0]
        # Avoid recalculating if we can
        if sdf_results is None:
            sdf_results = sdf(samples)
        # 计算物体内部并且距离物体表面超过 0.01 的点的个数。
        overlapping = (sdf_results > threshold).sum()

        return (overlapping / num_samples), sdf_results

    @staticmethod
    def calculate_contact_area(
            sdf=None,
            samples=None,
            threshold=0.01,
            max_contact=0.1,
            sdf_results=None
    ):
        """
        Calculate the contact area using samples and an sdf with a default tolerance in cm
        and the max contact area expected e.g. half (0.5) of all samples
        """
        num_samples = samples.shape[0]
        # Avoid recalculating if we can
        if sdf_results is None:
            sdf_results = sdf(samples)
        # 距离物体表面不超过 0.01 的点的个数。
        in_contact = (np.absolute(sdf_results) < threshold).sum()
        contact_percent = in_contact / (num_samples * max_contact)
        # Cap at 1.0
        if contact_percent > 1.0:
            contact_percent = 1.0
        return contact_percent, sdf_results

    @staticmethod
    def calculate_distance(
            sdf=None,
            samples=None,
            sdf_results=None
    ):
        """
        Calculate the average distance between
        the point cloud samples and sdf
        """
        num_samples = samples.shape[0]
        # Avoid recalculating if we can
        if sdf_results is None:
            sdf_results = sdf(samples)

        # max_index = np.argmax(sdf_results)
        # print("index of max SDF Point:", max_index)
        # print("Point with max SDF:", samples[max_index])
        # print("SDF value:", sdf_results[max_index])

        overlap_depth = sdf_results.max()

        # print(f"overlap_depth{overlap_depth}")

        if overlap_depth < 0:
            distance = -overlap_depth
            overlap_depth = 0.0
        else:
            distance = 0.0
        # print(f"distance{distance}")

        return distance, overlap_depth

    @staticmethod
    def calculate_iou(
            mesh1,
            samples1,
            mesh2,
            samples2,
            threshold=0.01
    ):
        """
        Calculate the intersection over union
        between the ground truth sdf and
        the dynamic samples
        """
        wns1 = igl.fast_winding_number_for_meshes(mesh1.vertices, mesh1.faces, samples2)
        # Samples that are inside both meshes
        overlap = samples2[wns1 > threshold]
        overlap_count = len(overlap)
        # Samples only inside mesh2
        only_mesh2 = samples2[wns1 <= threshold]
        only_mesh2_count = len(only_mesh2)
        wns2 = igl.fast_winding_number_for_meshes(mesh2.vertices, mesh2.faces, samples1)
        # Samples only inside mesh1
        only_mesh1 = samples1[wns2 <= threshold]
        only_mesh1_count = len(only_mesh1)
        # Union of the samples in only mesh1/2 and overlapping
        union_count = overlap_count + only_mesh1_count + only_mesh2_count
        iou = overlap_count / union_count
        return iou

    @staticmethod
    def calculate_iou_batch(jps, pred_transform, gt_transforms, num_samples=4096, symmetrical=False):
        # Sample the points once
        # then we will translate them for each transform
        # For IoU we want samples inside of the volume
        gt_vol_pts1 = JointPredictionSet.sample_volume_points(
            jps.body_one_mesh,
            num_samples=num_samples,
            seed=jps.seed,
            sample_surface=False
        )
        # Store the samples as n,4 so they can be easily transformed with a 4x4 matrix
        gt_vol_pts1 = util.pad_pts(gt_vol_pts1)

        # Copy and transform here as simply transforming the vertices
        # causes the face normals to invert if the flip parameter is used
        pred_mesh = jps.body_one_mesh.copy()
        pred_mesh.apply_transform(pred_transform)
        # Copy the verts so igl doesn't complain after we used transpose
        pred_vol_pts = util.transform_pts_by_matrix(gt_vol_pts1, pred_transform, copy=True)

        # global type_index

        if symmetrical:
            pred_transform2 = np.linalg.inv(pred_transform)
            gt_vol_pts2 = JointPredictionSet.sample_volume_points(
                jps.body_two_mesh,
                num_samples=num_samples,
                seed=jps.seed,
                sample_surface=False
            )
            gt_vol_pts2 = util.pad_pts(gt_vol_pts2)
            pred_mesh2 = jps.body_two_mesh.copy()
            pred_mesh2.apply_transform(pred_transform2)
            pred_vol_pts2 = util.transform_pts_by_matrix(gt_vol_pts2, pred_transform2, copy=True)

        best_iou1 = 0
        best_iou2 = 0
        for gt_transform in gt_transforms:
            # Get a copy of the body one mesh
            # and apply the gt transform
            gt_mesh = trimesh.Trimesh(
                vertices=jps.body_one_mesh.vertices,
                faces=jps.body_one_mesh.faces
            )
            gt_mesh.apply_transform(gt_transform)
            # Tranform and copy the verts so igl doesn't complain after we used transpose
            gt_vol_pts_t = util.transform_pts_by_matrix(gt_vol_pts1, gt_transform, copy=True)

            iou_result1 = JointEnvironment.calculate_iou(
                pred_mesh,
                pred_vol_pts,
                gt_mesh,
                gt_vol_pts_t
            )

            if iou_result1 > best_iou1:
                best_iou1 = iou_result1

            if symmetrical:
                gt_transform2 = np.linalg.inv(gt_transform)
                gt_mesh2 = trimesh.Trimesh(
                    vertices=jps.body_two_mesh.vertices,
                    faces=jps.body_two_mesh.faces
                )
                gt_mesh2.apply_transform(gt_transform2)
                # Tranform and copy the verts so igl doesn't complain after we used transpose
                gt_vol_pts_t2 = util.transform_pts_by_matrix(gt_vol_pts2, gt_transform2, copy=True)
                iou_result2 = JointEnvironment.calculate_iou(
                    pred_mesh2,
                    pred_vol_pts2,
                    gt_mesh2,
                    gt_vol_pts_t2
                )
                if iou_result2 > best_iou2:
                    best_iou2 = iou_result2

                # gt_mesh1t = trimesh.Trimesh(
                #     vertices=jps.body_one_mesh.vertices,
                #     faces=jps.body_one_mesh.faces
                # )
                # gt_mesh2t = trimesh.Trimesh(
                #     vertices=jps.body_two_mesh.vertices,
                #     faces=jps.body_two_mesh.faces
                # )
                # visualize_3d(pred_vol_pts, gt_mesh2t, iou_result1, 0, 0, 0, 0, f"{type_index}_pred_trans_1")
                # visualize_3d(gt_vol_pts_t, gt_mesh2t, iou_result1, 0, 0, 0, 0, f"{type_index}_gt_trans_1")
                #
                # visualize_3d(pred_vol_pts2, gt_mesh1t, iou_result2, 0, 0, 0, 0, f"{type_index}_pred_trans_2")
                # visualize_3d(gt_vol_pts_t2, gt_mesh1t, iou_result2, 0, 0, 0, 0, f"{type_index}_gt_trans_2")

            # type_index += 1

        return best_iou1, best_iou2

    @staticmethod
    def get_pc_scale(pc):
        return np.sqrt(np.max(np.sum((pc - np.mean(pc, axis=0)) ** 2, axis=1)))

    @staticmethod
    def calculate_cd(pc1, pc2):
        dist = cdist(pc1, pc2)
        error = np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))
        scale = JointEnvironment.get_pc_scale(pc1) + JointEnvironment.get_pc_scale(pc2)
        return error / scale

    @staticmethod
    def calculate_cd_batch(jps, pred_transform, gt_transforms, num_samples=4096, debug_plot=False):
        # For chamfer distance we want samples on the surface
        gt_surf_pts1, _ = trimesh.sample.sample_surface(jps.body_one_mesh, num_samples)
        # Store the samples as n,4 so they can be easily transformed with a 4x4 matrix
        gt_surf_pts1 = util.pad_pts(gt_surf_pts1)
        # Transform the predicted samples
        pred_surf_pts1_t = util.transform_pts_by_matrix(gt_surf_pts1, pred_transform)
        # Transform the static body
        gt_surf_pts2, _ = trimesh.sample.sample_surface(jps.body_two_mesh, num_samples)
        gt_transform2 = util.transform_to_np(jps.joint_data["joints"][0]["geometry_or_origin_two"]["transform"])
        gt_surf_pts2_t = util.transform_pts_by_matrix(gt_surf_pts2, gt_transform2)
        # Merge the predicted samples
        pred_surf_pts_t = np.vstack([pred_surf_pts1_t, gt_surf_pts2_t])
        # Loop over each joint and take the lowest chamfer distance
        best_cd = sys.float_info.max
        for gt_transform in gt_transforms:
            gt_surf_pts1_t = util.transform_pts_by_matrix(gt_surf_pts1, gt_transform)
            gt_surf_pts_t = np.vstack([gt_surf_pts1_t, gt_surf_pts2_t])
            cd_result = JointEnvironment.calculate_cd(pred_surf_pts_t, gt_surf_pts_t)
            if cd_result < best_cd:
                best_cd = cd_result
            if debug_plot:
                plot_util.plot_point_cloud(
                    pred_surf_pts_t,
                    gt_surf_pts_t,
                )
        return best_cd
