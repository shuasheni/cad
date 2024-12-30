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


def step2png(body_id):
    fn = f"C:\\Users\\40896\\Desktop\\data\\joint\\{body_id}.obj"
    output_path = f"C:\\Users\\40896\\Desktop\\data\\joint\\{body_id}.png"
    generate_thumbnail(fn, output_path)