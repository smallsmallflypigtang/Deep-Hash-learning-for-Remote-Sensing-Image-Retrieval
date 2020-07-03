import numpy as np
from .progress.bar import Bar


def cal_AP(query, database, with_groudtruth=False, with_top=20):
    query_data, query_label = query[0], query[1]
    target_data, target_label = database[0], database[1]
    query_data = np.reshape(query_data, (1, query_data.shape[0]))
    target_label = np.reshape(target_label, (target_label.shape[0]))
    if with_groudtruth:
        top = np.sum(np.equal(target_label, query_label))
    else:
        top = with_top
    precision_list = []
    distance_matrix = np.linalg.norm(target_data - query_data, axis=1)
    id = np.argsort(distance_matrix, axis=0)[:top]

    for i in range(top):
        retrieved_label = target_label[id[:i + 1]]
        precision = np.mean(np.equal(retrieved_label, query_label))
        is_counted = np.equal(retrieved_label[i],
                              query_label).astype('float32')
        precision_list.append(precision * is_counted)
    average_precision = np.mean(np.array(precision_list))
    return average_precision


def cal_mAP(query, database, with_groudtruth=False, with_top=20):
    query_set = query[0]
    query_set_label = query[1]
    AP_list = []
    bar = Bar('calculating mAP', max=query_set.shape[0])
    for i in range(0, query_set.shape[0]):
        current_query = [query_set[i], query_set_label[i]]
        current_query_AP = cal_AP(current_query, database, with_groudtruth,
                                  with_top)
        AP_list.append(current_query_AP)
        temp_value = np.mean(np.array(AP_list))
        bar.suffix = 'value :{value:.4f}'.format(value=temp_value)
        bar.next()
    bar.finish()
    mAP = np.mean(np.array(AP_list))
    return mAP
