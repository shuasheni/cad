from flask import *

from cad.step_calc import output_step_transform, output_joint_json
from cad.step_to_graphjson import process_one_file
from cad.step_to_mesh import step_to_obj_with_normals
from cad.cad_to_png import generate_body_png, generate_joint_png
from predict_joint import predict_new_joint, predict_exist_joint
import shutil
from cad.step_parse import step_parse, update_face
import os

from db.db import connect, select_joint, select_body, delete_joint, delete_body, insert_body, authenticate_user, \
    update_body, update_joint, insert_joints, select_model, get_groups

app = Flask(__name__, template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True


app.secret_key = '12345678'
db, cursor = connect()
IMAGE_FOLDER = 'C:\\Users\\40896\\Desktop\\data\\joint'

@app.route('/get_image/<filename>')
def get_image(filename):
    # 生成图片的完整路径
    image_path = os.path.join(IMAGE_FOLDER, filename)

    # 检查图片是否存在
    if os.path.exists(image_path):
        return send_file(image_path)
    else:
        abort(404)  # 返回404错误

# 登录页面和处理逻辑
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # 验证用户名和密码
        result, department_id = authenticate_user(db, cursor, username, password)

        if department_id:
            # 登录成功，将用户名和 department_id 保存到 session
            session['username'] = username
            session['department_id'] = department_id
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash(result, 'danger')

    # GET 请求时显示登录表单
    return render_template('login.html', tt="用户登录", session=session)

# 注销逻辑
@app.route('/logout')
def logout():
    session.pop('username', None)  # 删除 session 中的用户名
    flash('You have successfully logged out!', 'info')
    return redirect(url_for('login'))  # 注销后重定向到登录页面


@app.route('/')
def index():
    return render_template('index.html', tt="装配预测系统", session=session)


@app.route('/joint_list', methods=["POST", "GET"])
def joint_list():
    joint_id = None
    joint_name = None
    body_id = None
    type = None
    min_value = None
    max_value = None

    page_size = 40
    page = 0
    if request.method == "POST":
        joint_id = request.form.get("joint_id")
        joint_name = request.form.get("name")
        body_id = request.form.get("body_id")
        type = request.form.get("type")
        min_value = request.form.get("minVallue")
        max_value = request.form.get("maxVallue")
        page_size = int(request.form.get("page_size"))
        if joint_id == '':
            joint_id = None
        if joint_name == '':
            joint_name = None
        if body_id == '':
            body_id = None
        if type == '':
            type = None
        if min_value == '' or min_value is None:
            min_value = None
        else:
            min_value = float(min_value)
        if max_value == '' or max_value is None:
            max_value = None
        else:
            max_value = float(max_value)

    pg = request.args.get("page")
    if pg is not None:
        page = int(pg) - 1

    data, page_num = select_joint(db, cursor, joint_id, joint_name, type, body_id, page_size, page, min_value, max_value)

    results = []
    #
    # for row in data:
    #     jid = row[0]
    #     cd1, cd2 = compare_cd(jid)
    #     results.append([jid, cd1, cd2])
    #
    # for row in results:
    #     print(f"joint_id: {row[0]}, old_cd: {row[1]}, new_cd: {row[2]}")
    page = page + 1
    # print(page,page_num)
    min_page = max(page - 10, 1)
    max_page = min(min_page + 20, page_num + 1)

    return render_template('joint_list.html', tt="装配体列表", session=session, table_data=data,
                           page=page, page_num=page_num, min_page = min_page, max_page = max_page)


@app.route('/body_list', methods=["POST", "GET"])
def body_list():
    body_id = None
    name = None
    tags = None
    page_size = 40
    page = 0
    if request.method == "POST":
        body_id = request.form.get("body_id")
        name = request.form.get("name")
        tags = request.form.get("tags")
        page_size = int(request.form.get("page_size"))
        if body_id == '':
            body_id = None
        if name == '':
            name = None
        if tags == '' or tags is None:
            tags = None
        else:
            tags = tags.split(',')

    pg = request.args.get("page")
    if pg is not None:
        page = int(pg) - 1
    data,page_num = select_body(db, cursor, body_id, name, page_size, page, tags)

    for row in data:
        tags = row[4]
        arr = tags.split(',')
        row[4] = arr
    page = page + 1
    # print(page,page_num)
    min_page = max(page - 10, 1)
    max_page = min(min_page + 20, page_num + 1)
    return render_template('body_list.html', tt="零件列表", session=session, table_data=data,
                        page=page, page_num=page_num, min_page = min_page, max_page = max_page)

@app.route('/model_list', methods=["POST", "GET"])
def model_list():
    model_id = None
    name = None
    group = None
    page_size = 40
    page = 0
    if request.method == "POST":
        model_id = request.form.get("model_id")
        name = request.form.get("name")
        group = request.form.get("group")
        page_size = int(request.form.get("page_size"))
        if model_id == '':
            model_id = None
        if name == '':
            name = None
        if group == '' or group is None:
            group = None

    pg = request.args.get("page")
    if pg is not None:
        page = int(pg) - 1
    data,page_num = select_model(db, cursor, model_id, name, page_size, page, group)
    groups = get_groups(db, cursor)
    page = page + 1
    min_page = max(page - 10, 1)
    max_page = min(min_page + 20, page_num + 1)
    return render_template('model_list.html', tt="特征匹配模型列表", session=session, table_data=data,
                        page=page, page_num=page_num, min_page = min_page, max_page = max_page, groups=groups)

@app.route('/joint_delete')
def joint_delete():
    joint_id = request.args.get("joint_id")
    if delete_joint(db, cursor, joint_id):
        return "删除成功"
    else:
        return "删除失败"




@app.route('/body_view')
def body_view():
    body_id = request.args.get("body_id")
    print(body_id)
    faces, edges = step_parse(body_id)

    jn = f"C:\\Users\\40896\\Desktop\\data\\joint\\{body_id}.json"

    with open(jn, encoding="utf8") as f:
        graph_json_data = json.load(f)

    maxx = graph_json_data["properties"]["bounding_box"]["max_point"]["x"]
    maxy = graph_json_data["properties"]["bounding_box"]["max_point"]["y"]
    maxz = graph_json_data["properties"]["bounding_box"]["max_point"]["z"]

    minx = graph_json_data["properties"]["bounding_box"]["min_point"]["x"]
    miny = graph_json_data["properties"]["bounding_box"]["min_point"]["y"]
    minz = graph_json_data["properties"]["bounding_box"]["min_point"]["z"]

    fn = f"C:\\Users\\40896\\Desktop\\data\\joint\\{body_id}.obj"
    shutil.copy(fn, f"static\\{body_id}.obj")

    body, _ = select_body(db, cursor, body_id)

    body[0][4] = body[0][4].split(',')


    return render_template('body_view.html', tt="零部件查看", session=session, body=body[0],
                           faces=faces, body_id=body_id, max=[maxx, maxy, maxz], min=[minx, miny, minz])

@app.route('/update_step_data', methods=['POST'])
def get_data():
    id = request.form.get('id')  # 从前端获取数据
    face_index = request.form.get('index')  # 从前端获取数据
    face = request.form.get('face')  # 从前端获取数据
    update_face(id, int(face_index), json.loads(face))
    response_data = {'message': 'ok'}  # 处理后的数据
    return jsonify(response_data)


@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        body_id = request.form.get("up_body_id")
        body_name = request.form.get("up_body_name")
        tags = request.form.get("up_body_tags")
        file = request.files.get("file")
        if file.filename:
            print(file)
            fn = f"C:\\Users\\40896\\Desktop\\data\\joint\\{body_id}.step"
            ofn = f"C:\\Users\\40896\\Desktop\\data\\joint\\{body_id}.obj"
            file.save(fn)
            step_to_obj_with_normals(fn, ofn)
            generate_body_png(body_id)
            face_num, edge_num = process_one_file(fn, body_id, f"C:\\Users\\40896\\Desktop\\data\\joint")
            insert_body(db, cursor, body_id, body_name, face_num, edge_num, tags)
            return redirect(url_for('body_list'))

    return "导入失败"


@app.route("/joint_save", methods=["POST", "GET"])
def joint_save():
    if request.method == "POST":
        joint_id = request.form.get("joint_id")
        joint_name = request.form.get("joint_name")
        body1_id = request.form.get("body1_id")
        body2_id = request.form.get("body2_id")
        predict_n = request.form.get("predict_n")

        joint_path = f'C:\\Users\\40896\\Desktop\\data\\ds\\{body1_id}_{body2_id}_{predict_n}.json'
        step_path = f'C:\\Users\\40896\\Desktop\\data\\ds\\{body1_id}_{body2_id}_{predict_n}.step'
        png_path = f'C:\\Users\\40896\\Desktop\\data\\ds\\{body1_id}_{body2_id}_{predict_n}.png'
        djoint_path = f'C:\\Users\\40896\\Desktop\\data\\joint\\{joint_id}.json'
        dstep_path = f'C:\\Users\\40896\\Desktop\\data\\joint\\{joint_id}.step'
        dpng_path = f'C:\\Users\\40896\\Desktop\\data\\joint\\{joint_id}.png'
        shutil.move(joint_path, djoint_path)
        shutil.move(step_path, dstep_path)
        shutil.move(png_path, dpng_path)

        insert_joints(db,cursor,joint_id,joint_name,"save",body1_id,body2_id,1.0)
        return redirect(url_for('joint_list'))


@app.route('/body_delete')
def body_delete():
    body_id = request.args.get("body_id")
    if delete_body(db, cursor, body_id):
        return "删除成功"
    else:
        return "删除失败"

@app.route('/body_update', methods=["POST", "GET"])
def body_update():
    if request.method == "POST":
        body_id = request.form.get("body_id")
        body_name = request.form.get("name")
        tags = request.form.get("tags")
        update_body(db, cursor, body_id, body_name, tags)
        return redirect(url_for('body_list'))


@app.route('/joint_update', methods=["POST", "GET"])
def joint_update():
    if request.method == "POST":
        body_id = request.form.get("joint_id")
        body_name = request.form.get("joint_name")
        tags = request.form.get("joint_value")
        update_joint(db, cursor, body_id, body_name, tags)
        return redirect(url_for('joint_list'))




@app.route('/joint_predict')
def joint_predict():
    n = 0
    joint_n = 0
    joint_id = request.args.get("joint_id")
    n = request.args.get("n")
    joint_n = request.args.get("joint_n")
    if n == '' or n == None:
        n = 0
    else:
        n = int(n)
    if joint_n == '' or joint_n == None:
        joint_n = 0
    else:
        joint_n = int(joint_n)
    print(joint_id)

    body1_id, body2_id, prediction_data, iou, cd, t2, joints_data = predict_exist_joint(joint_id, n, joint_n)

    file_path = f'C:\\Users\\40896\\Desktop\\data\\ds\\{body1_id}_{body2_id}_{n}.step'

    output_step_transform(body1_id, body2_id, t2, file_path)

    body1, _ = select_body(db, cursor, body1_id)
    body2, _ = select_body(db, cursor, body2_id)

    body1[0][4] = body1[0][4].split(',')
    body2[0][4] = body2[0][4].split(',')

    f1, e1 = step_parse(body1_id)
    f2, e2 = step_parse(body2_id)
    # print(joints_data)
    # print(prediction_data)

    return render_template('joint_predict.html', tt="装配体再预测", session=session,faces1 = f1,faces2 = f2,edges1 = e1,edges2 = f2, joint_id = joint_id,
                           body1_id=body1_id, body2_id=body2_id, predict_n=n, joint_n=joint_n, body1=body1[0], body2=body2[0], prediction_data=prediction_data,
                           joints_data = joints_data, iou=iou, cd = cd, sc=4096)


@app.route('/predict_view')
def predict_view():
    body1_id = request.args.get("body1_id")
    body2_id = request.args.get("body2_id")
    n = request.args.get("n")
    # print(body1_id+body2_id+n)
    if n == '' or n == None:
        n = 0
    else:
        n = int(n)

    body1,_ = select_body(db, cursor, body1_id)
    body2,_ = select_body(db, cursor, body2_id)

    body1[0][4] = body1[0][4].split(',')
    body2[0][4] = body2[0][4].split(',')

    f1, e1 = step_parse(body1_id)
    f2, e2 = step_parse(body2_id)

    prediction_data, t2, x, offset_limit, prediction, fc1, fc2 = predict_new_joint(body1_id, body2_id, n)

    offset = x[0] * offset_limit
    if len(x) == 2:
        rotation = x[1]
    else:
        rotation = 0

    path = f"C:\\Users\\40896\\Desktop\\data\\ds"
    step_path = f'C:\\Users\\40896\\Desktop\\data\\ds\\{body1_id}_{body2_id}_{n}.step'

    output_step_transform(body1_id, body2_id, t2, step_path)
    output_joint_json(body1_id, body2_id, fc1, fc2, t2, offset, rotation, n, prediction, path)
    generate_joint_png(body1_id, body2_id, t2, n)
    return render_template('predict_view.html', tt="预测结果查看", session=session,faces1 = f1,faces2 = f2,
                           body1_id=body1_id, body2_id=body2_id, predict_n=n,body1=body1[0], body2=body2[0], prediction_data=prediction_data)

@app.route("/download_joint_step")
def download_joint_step():
    body1_id = request.args.get("body1_id")
    body2_id = request.args.get("body2_id")
    n = request.args.get("n")

    file_path = f'C:\\Users\\40896\\Desktop\\data\\ds\\{body1_id}_{body2_id}_{n}.step'

    return send_file(file_path, as_attachment=True)

@app.route("/download_j_json")
def download_j_json():
    joint_id = request.args.get("joint_id")
    file_path = f"C:\\Users\\40896\\Desktop\\data\\joint\\{joint_id}.json"
    return send_file(file_path, as_attachment=True)

@app.route("/download_j_step")
def download_j_step():
    joint_id = request.args.get("joint_id")
    file_path = f"C:\\Users\\40896\\Desktop\\data\\joint\\{joint_id}.step"
    return send_file(file_path, as_attachment=True)


@app.route("/download_step")
def download_step():
    body_id = request.args.get("body_id")
    file_path = f"C:\\Users\\40896\\Desktop\\data\\joint\\{body_id}.step"
    return send_file(file_path, as_attachment=True)


@app.route("/download_obj")
def download_obj():
    body_id = request.args.get("body_id")
    file_path = f"C:\\Users\\40896\\Desktop\\data\\joint\\{body_id}.obj"
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, host='0.0.0.0')
