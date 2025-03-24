import json
import math
from pathlib import Path
import time
from occwl.io import load_step

from cad.step_graph_match import match
from cad.step_to_mesh import step_to_obj_with_normals


def get_matcher():  #这里正常是通过查数据库获取，改为直接定义两种子图
    mather = {
        'name': '通孔',
        'nodes': [
            (0, {"type": "plain"}),
            (1, {"type": "circle"}),
            (2, {"type": "plain"}),
        ],
        'edges': [
            (0, 1, {'type': 'circle'}),
            (1, 2, {'type': 'circle'})
        ],
        'comparisons': [
            {
                'a': {
                    'n1': 1,
                    'param1': 'direction',
                    'param2': 'origin'
                },
                'b': {
                    'n1': 0,
                    'n2': 1,
                    'param1': 'direction',
                    'param2': 'origin'
                },
                'func': 'collinear'
            },
            {
                'a': {
                    'n1': 1,
                    'param1': 'direction',
                    'param2': 'origin'
                },
                'b': {
                    'n1': 1,
                    'n2': 2,
                    'param1': 'direction',
                    'param2': 'origin'
                },
                'func': 'collinear'
            },
            {
                'a': {
                    'n1': 0,
                    'param1': 'normal',
                },
                'b': {
                    'n1': 2,
                    'param1': 'normal',
                },
                'func': 'reverse'
            },
            {
                'a': {
                    'n1': 1,
                    'param1': 'angle',
                },
                'b': {
                    'value': math.pi * 2,
                },
                'func': 'eq'
            },
            {
                'a': {
                    'n1': 1,
                    'param1': 'outside',
                },
                'b': {
                    'value': False,
                },
                'func': 'is'
            },

        ],
        'params': [
        ],
        'parts': [
            {
                'name': '前面',
                'faces': [0]
            },
            {
                'name': '孔面',
                'faces': [1]
            },
            {
                'name': '后面',
                'faces': [2]
            },
        ]
    }

    return mather

def graph_match(solid):
    mather = get_matcher()
    feather = match(solid, mather)
    return feather









lb_dir = 'C:\\Users\\40896\\Desktop\\data\\feather\\label'
dt_dir = 'C:\\Users\\40896\\Desktop\\data\\feather\\step'
pattern = "*.stp"
list = [f.name for f in Path(dt_dir).glob(pattern)]
# list = ['00096860.stp']

all_r_num = 0
all_gt_num = 0
all_a_num = 0
all_time = 0
for name in list[:1000]:
    print(name)
    stp_file = Path(dt_dir) / name

    id = name.rstrip('.stp')
    # step_to_obj_with_normals(str(stp_file),f"{id}.obj")

    json_name = id + '.json'
    label_file = Path(lb_dir) / json_name

    with open(label_file, encoding="utf8") as f:
        joint_data = json.load(f)
    labels = joint_data["labels"]
    solids = load_step(stp_file)
    if len(solids) != 1:
        continue
    solid = solids[0]
    start_time = time.time()
    feathers = graph_match(solid)
    end_time = time.time()

    results = []
    for feather in feathers:
        results.extend(feather['parts'][1]['faces'])
    print(results)

    gt = [index for index, value in enumerate(labels) if value == 17]
    print(gt)

    result_num = len(results)
    gt_num = len(gt)

    num = 0
    for result in results:
        if result in gt:
            num += 1

    all_a_num += num
    all_gt_num += gt_num
    all_r_num += result_num
    all_time += (end_time - start_time)

print(all_r_num, all_gt_num, all_a_num, all_time)