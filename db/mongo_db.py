from pymongo import MongoClient

# 1. 连接 MongoDB
client = MongoClient("mongodb://localhost:27017/")  # 连接到本地 MongoDB 实例

# 2. 选择数据库和集合
db = client["cmd_project"]  # 如果不存在，会自动创建
collection = db["step_parse"]  # 如果不存在，会自动创建

def insert_step_parse(body_id, faces, edges, features):
    data = {
        "_id": body_id,
        "faces": faces,
        "edges": edges,
        "features": features}
    insert_result = collection.insert_one(data)  # 插入一条数据
    print(f"Inserted document ID: {insert_result.inserted_id}")


def select_step_parse(body_id):
    result = collection.find_one({"_id": body_id})
    if result:
        print(f"Found document: {body_id}")
        return result["faces"], result["edges"], result["features"]
    else:
        return None, None, None

def update_step_parse(body_id, faces, edges, features):
    update_result = collection.update_one(
        {"_id": body_id},  # 查询条件，根据 _id 查找文档
        {"$set": {"faces": faces, "edges": edges, "features": features}}  # 更新的内容
    )
    if update_result.matched_count > 0:
        print(f"Document with _id {body_id} was found.")
        if update_result.modified_count > 0:
            print(f"Document with _id {body_id} was updated.")
        else:
            print(f"No changes were made to the document with _id {body_id}.")
    else:
        print(f"No document found with _id {body_id}.")


# result = collection.find_one({"_id": 'zhou1'})