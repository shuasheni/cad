import os
import pymysql
import sys
import json
from pathlib import Path

from cad.cad_to_png import generate_body_png


data_dir = 'C:\\Users\\40896\\Desktop\\data\\joint'

def connect():
    """
    host：ip地址
    port:端口号
    user:数据库用户名称
    password：数据库用户密码
    database: 数据库名称
    """
    # 获取数据库连接对象
    db = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456', database='cadtest', charset='utf8mb4')
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    return db, cursor

def authenticate_user(db=None,cursor=None,username=None,password=None):
    sql_query = "SELECT password, department_id FROM users WHERE username=%s"
    cursor.execute(sql_query, (username,))
    result = cursor.fetchone()

    if result:
        db_password, department_id = result
        # 验证密码是否匹配
        if db_password == password:
            return "ok", department_id  # 返回 department_id
        else:
            return "Wrong password", None  # 密码不匹配
    else:
        return "None username", None  # 用户不存在

def insert_body(db=None,cursor=None,body_id=None,name=None,face_count=0,edge_count=0,tags=None):
    sql_insert_query = """INSERT INTO bodies (body_id,name,face_count,edge_count,tags) VALUES (%s, %s, %s, %s, %s)"""
    data_to_insert = (body_id, name, face_count, edge_count, tags)
    cursor.execute(sql_insert_query, data_to_insert)
    db.commit()

def update_body(db=None, cursor=None, body_id=None, name=None, tags=None):
    sql = f"UPDATE bodies SET name = '{name}', tags = '{tags}' WHERE body_id = '{body_id}';"

    print(sql)
    cursor.execute(sql)
    db.commit()

def select_body(db=None,cursor=None, body_id = None, name = None, page_size = 50, page = 0, tags=None):
    cursor.execute("SELECT COUNT(*) FROM bodies")
    total_count = cursor.fetchone()[0]
    page_num = (total_count + page_size - 1) // page_size

    sql = f"select body_id,name,face_count,edge_count,tags from bodies LIMIT {page_size} OFFSET {page*page_size};"

    if body_id is not None:
        sql = f"select body_id,name,face_count,edge_count,tags from bodies where body_id = '{body_id}';"
        page_num = 1
    elif name is not None:
        sql = f"select body_id,name,face_count,edge_count,tags from bodies where name = '{name}';"
        page_num = 1
    elif tags is not None:
        sql = f"select body_id,name,face_count,edge_count,tags from bodies where tags like '%{tags[0]}%'"
        for tag in tags[1:]:
            sql = sql + f" and tags like '%{tag}%'"
        sql = sql + ";"
        page_num = 1

    print(sql)
    cursor.execute(sql)
    data = cursor.fetchall()
    db.commit()
    modified_result = [list(row) for row in data]
    return modified_result, page_num

def delete_body(db=None,cursor=None, body_id = None):
    sql = f"DELETE FROM bodies WHERE body_id = '{body_id}';"
    print(sql)
    try:
        cursor.execute(sql)
        db.commit()
        return True
    except pymysql.MySQLError as e:
        print("Error: ", e)
        return False


def insert_joints(db=None,cursor=None,joint_id = None, name = None, joint_type = None, body1_id = None, body2_id = None, rate=None):
    sql_insert_query = """INSERT INTO joints (joint_id, name, joint_type, body1_id, body2_id, rate) VALUES (%s, %s, %s, %s, %s, %s)"""
    data_to_insert = (joint_id, name, joint_type, body1_id, body2_id, rate)
    cursor.execute(sql_insert_query, data_to_insert)
    db.commit()

def select_joint(db=None,cursor=None, joint_id = None, name = None, joint_type = None, body_id = None, page_size = 50, page = 0, max_rate = None, min_rate = None):
    cursor.execute("SELECT COUNT(*) FROM bodies")
    total_count = cursor.fetchone()[0]
    page_num = (total_count + page_size - 1) // page_size

    sql = f"select joint_id, name, joint_type, body1_id, body2_id, rate from joints "
    if joint_id is not None:
        sql = f"select joint_id, name, joint_type, body1_id, body2_id, rate from joints where joint_id = '{joint_id}';"
        page_num = 1
    elif body_id is not None:
        sql = f"select joint_id, name, joint_type, body1_id, body2_id, rate from joints where body1_id = '{body_id}' or body2_id = '{body_id}';"
        page_num = 1
    elif name is not None:
        sql = f"select joint_id, name, joint_type, body1_id, body2_id, rate from joints where name = '{name}';"
        page_num = 1
    else:
        connect_word = "where"
        if joint_type is not None:
            sql = sql + f"{connect_word} joint_type = '{joint_type}' "
            connect_word = "and"
        if min_rate is not None:
            sql = sql + f"{connect_word} rate >= '{min_rate}' "
            connect_word = "and"
        if max_rate is not None:
            sql = sql + f"{connect_word} rate <= '{max_rate}' "
        sql = sql + f"LIMIT {page_size} OFFSET {page*page_size};"
        page_num = 1

    print(sql)
    cursor.execute(sql)
    data = cursor.fetchall()
    db.commit()
    return data, page_num

def delete_joint(db=None,cursor=None, joint_id = None):
    sql = f"DELETE FROM joints WHERE joint_id = '{joint_id}';"
    print(sql)
    try:
        cursor.execute(sql)
        db.commit()
        return True
    except pymysql.MySQLError as e:
        print("Error: ", e)
        return False

def update_joint(db=None, cursor=None, id=None, name=None, value=None):
    sql = f"UPDATE joints SET name = '{name}', rate = {value} WHERE joint_id = '{id}';"
    print(sql)
    cursor.execute(sql)
    db.commit()

def select_model(db=None,cursor=None, model_id = None, name = None, page_size = 50, page = 0, group=None):
    cursor.execute("SELECT COUNT(*) FROM model")
    total_count = cursor.fetchone()[0]
    page_num = (total_count + page_size - 1) // page_size

    sql = f"select model_id,model_name,group_name from model NATURAL JOIN group_table LIMIT {page_size} OFFSET {page*page_size};"

    if model_id is not None:
        sql = f"select model_id,model_name,group_name from model NATURAL JOIN groups where model_id = '{model_id}';"
        page_num = 1
    elif name is not None:
        sql = f"select model_id,model_name,group_name from model NATURAL JOIN groups where model_name = '{name}';"
        page_num = 1
    elif group is not None:
        sql = f"select model_id,model_name,group_name from model NATURAL JOIN groups where group_name = '{group}';"
        page_num = 1

    print(sql)
    cursor.execute(sql)
    data = cursor.fetchall()
    db.commit()
    modified_result = [list(row) for row in data]
    return modified_result, page_num

def get_groups(db=None,cursor=None):
    sql = f"select group_id, group_name from group_table;"
    print(sql)
    cursor.execute(sql)
    data = cursor.fetchall()
    db.commit()
    modified_result = [list(row) for row in data]
    return modified_result

def close(db=None):
    db.close()

def insert_all_body(db=None,cursor=None):
    sql_insert_query = """INSERT INTO bodies (body_id,name,face_count,edge_count,tags) VALUES (%s, %s, %s, %s, %s)"""
    pattern = "*_*_*_[1-9].json"
    list = [f.name for f in Path(data_dir).glob(pattern)]
    for name in list:
        graph_json_file = data_dir / name
        id = name.rstrip('.json')

        print(id)
        obj_file = data_dir / f"{id}.obj"
        png_file = data_dir / f"{id}.png"
        if os.path.exists(png_file) or not os.path.exists(obj_file):
            continue
        with open(graph_json_file, encoding="utf8") as f:
            graph_json_data = json.load(f)
            node_count = len(graph_json_data["nodes"])
            link_count = len(graph_json_data["links"])
            f.close()
        if node_count >= 2 and link_count > 0:
            # 只插入有效零件
            data_to_insert = (id, f"零件{id}", node_count, link_count, "")
            cursor.execute(sql_insert_query, data_to_insert)
            db.commit()
            generate_body_png(id)

def insert_all_joint(db=None,cursor=None):
    sql_insert_query = """INSERT INTO joints (joint_id, name, joint_type, body1_id, body2_id, rate) VALUES (%s, %s, %s, %s, %s, %s)"""
    sql_insert_query_bad = """INSERT INTO joint_bad (joint_id) VALUES (%s)"""
    pattern = "joint_set_[0-9][0-9][0-9][0-9][0-9].json"
    list = [f.name for f in Path(data_dir).glob(pattern)]
    for name in list:
        joint_json_file = data_dir / name
        with open(joint_json_file, encoding="utf8") as f:
            joint_data = json.load(f)
        g1_id = joint_data["body_one"]
        g2_id = joint_data["body_two"]
        joint_id = (name.rstrip('.json'))
        joint_type = joint_data["joints"][0]["joint_motion"]["joint_type"]

        print(joint_id)
        try:
            data_to_insert = (joint_id, f"装配体{joint_id}", joint_type, g1_id, g2_id, 1)
            cursor.execute(sql_insert_query, data_to_insert)
            db.commit()
        except Exception as e:
            db.rollback()
            cursor.execute(sql_insert_query_bad, joint_id)
            db.commit()

def init_db():
    db, cursor = connect()
    insert_all_body(db, cursor)
    insert_all_joint(db, cursor)
    db.close()




