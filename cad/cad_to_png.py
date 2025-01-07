import trimesh
import pyrender
import numpy as np
from PIL import Image

def generate_thumbnail(file_path, output_path, width=400, height=400):
    # 加载3D模型
    mesh = trimesh.load(file_path)

    # 获取模型的边界框
    bounding_box = mesh.bounding_box_oriented
    center = bounding_box.centroid  # 边界框的中心
    extents = bounding_box.extents  # 边界框的尺寸 (宽度, 高度, 深度)

    # 根据边界框尺寸计算相机位置
    max_extent = np.max(extents)
    camera_distance = 1.0 * max_extent  # 调整倍数，使模型不显得太小

    # 相机位置在模型中心的X、Y、Z轴方向上方偏移
    # 例如，将相机放在中心点的右上前方
    camera_position = np.array([center[0] + camera_distance,  # X方向偏移
                                center[1] + camera_distance,  # Y方向偏移
                                center[2] + camera_distance])  # Z方向偏移

    # 计算相机的朝向向量 (look-at 矢量)
    direction = center - camera_position
    direction = direction / np.linalg.norm(direction)  # 单位化方向向量

    # 使用 pyrender 渲染场景
    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh)

    # 设置相机参数
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)  # 增大FOV使模型更大

    # 计算相机的旋转矩阵，使其朝向模型
    up = np.array([0, 1, 0])  # 定义一个"上"方向
    z_axis = -direction
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # 构建旋转矩阵
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = np.vstack([x_axis, y_axis, z_axis]).T

    # 设置相机的位移矩阵
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = camera_position

    # 最终的相机姿态矩阵
    camera_pose = translation_matrix @ rotation_matrix

    # 将相机添加到场景中
    scene.add(camera, pose=camera_pose)

    # 添加光源
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # 创建渲染器
    r = pyrender.OffscreenRenderer(width, height)

    # 渲染场景为图像
    color, _ = r.render(scene)

    # 使用 Pillow 保存渲染图像为缩略图
    image = Image.fromarray(color)
    image.save(output_path)

    print(f"Thumbnail saved to {output_path}")

def generate_thumbnail2(file_path_1, file_path_2, transformation_matrix, output_path, width=400, height=400):
    # 加载两个3D模型
    mesh_1 = trimesh.load(file_path_1)
    mesh_2 = trimesh.load(file_path_2)

    # 对第一个网格应用仿射变换矩阵
    mesh_1.apply_transform(transformation_matrix)

    # 获取第一个网格的边界框
    bounding_box_1 = mesh_1.bounding_box_oriented
    center_1 = bounding_box_1.centroid
    extents_1 = bounding_box_1.extents
    max_extent_1 = np.max(extents_1)
    camera_distance_1 = 1.0 * max_extent_1

    # 获取第二个网格的边界框
    bounding_box_2 = mesh_2.bounding_box_oriented
    center_2 = bounding_box_2.centroid
    extents_2 = bounding_box_2.extents
    max_extent_2 = np.max(extents_2)
    camera_distance_2 = 1.0 * max_extent_2

    # 计算总的边界框以便于设置相机位置
    all_centers = np.mean([center_1, center_2], axis=0)
    all_extents = np.max([extents_1, extents_2], axis=0)
    max_extent = np.max(all_extents)
    camera_distance = 1.5 * max_extent  # 增大一点相机距离，使模型显得不太小

    # 相机位置设置为模型中心偏移一定距离
    camera_position = np.array([all_centers[0] + camera_distance,
                                all_centers[1] + camera_distance,
                                all_centers[2] + camera_distance])

    # 计算相机的朝向向量
    direction = all_centers - camera_position
    direction = direction / np.linalg.norm(direction)

    # 使用 pyrender 渲染场景
    scene = pyrender.Scene()

    # 设置第一个网格为黄色
    yellow_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[1.0, 1.0, 0.0, 1.0])  # RGBA
    mesh_1_render = pyrender.Mesh.from_trimesh(mesh_1, material=yellow_material)
    scene.add(mesh_1_render)

    # 设置第二个网格为淡蓝色
    light_blue_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.5, 0.8, 1.0, 1.0])  # RGBA
    mesh_2_render = pyrender.Mesh.from_trimesh(mesh_2, material=light_blue_material)
    scene.add(mesh_2_render)

    # 设置相机参数
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)

    # 计算相机的旋转矩阵
    up = np.array([0, 1, 0])  # "上"方向
    z_axis = -direction
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = np.vstack([x_axis, y_axis, z_axis]).T

    # 设置相机的位移矩阵
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = camera_position

    # 最终的相机姿态矩阵
    camera_pose = translation_matrix @ rotation_matrix

    # 将相机添加到场景
    scene.add(camera, pose=camera_pose)

    # 添加光源
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # 创建渲染器
    r = pyrender.OffscreenRenderer(width, height)

    # 渲染场景为图像
    color, _ = r.render(scene)

    # 使用 Pillow 保存渲染图像为缩略图
    image = Image.fromarray(color)
    image.save(output_path)

    print(f"Thumbnail saved to {output_path}")

def generate_body_png(body_id):
    fn = f"C:\\Users\\40896\\Desktop\\data\\joint\\{body_id}.obj"
    output_path = f"C:\\Users\\40896\\Desktop\\data\\joint\\{body_id}.png"
    generate_thumbnail(fn, output_path)

def generate_joint_png(body1_id, body2_id, transform, n=0):
    fn1 = f"C:\\Users\\40896\\Desktop\\data\\joint\\{body1_id}.obj"
    fn2 = f"C:\\Users\\40896\\Desktop\\data\\joint\\{body2_id}.obj"
    output_path = f"C:\\Users\\40896\\Desktop\\data\\ds\\{body1_id}_{body2_id}_{n}.png"
    generate_thumbnail2(fn1, fn2, transform, output_path)