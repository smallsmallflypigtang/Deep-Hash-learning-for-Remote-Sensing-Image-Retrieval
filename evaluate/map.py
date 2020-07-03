import numpy as np


def cal_AP(query, database):
    q_data = query[0]
    q_label = query[1]
    db = database[0]
    db_label = database[1]
    top = 60
    ap = []
    distance_matrix = np.sqrt(np.sum(np.square(db - q_data), axis=1))
    id = np.argsort(distance_matrix, axis=0)[:top]
    for i in range(top):
        true_label = db_label[id[:i + 1]]
        retrieval_label = q_label
        precision = np.mean(np.equal(true_label, retrieval_label))
        ap.append(precision * np.equal(true_label[i], retrieval_label).astype('float32'))
    return np.mean(np.array(ap))


def cal_mAP(query, database):
    query_set = query[0]
    query_set_label = query[1]
    mAP = []
    for i in range(0, query_set.shape[0]):
        current_query = [query_set[i], query_set_label[i]]
        temp = cal_AP(current_query, database)
        mAP.append(temp)
    return np.mean(np.array(mAP))
