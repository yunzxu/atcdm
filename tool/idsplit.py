import numpy as np

def idsplit(n_subject, ratio, shuffle=True):
    '''
    split n_subject data,
    if ratio<1, then train_id=n*ratio,
    if ratio>1, then cross-validaiton
    '''
    id_list = np.arange(n_subject)
    if shuffle:
        np.random.shuffle(id_list)
    if ratio <= 1:
        n = int(np.round(n_subject*ratio))
        train_id = id_list[:n]
        test_id = id_list[n:]
    else:
        train_id = [None]*ratio
        test_id = [None]*ratio
        add_one = n_subject % ratio
        tmp = id_list[add_one:].reshape((ratio, -1))
        c = 0
        # print(id_list[:add_one])
        for n in range(ratio):
            list_one = id_list[:add_one].tolist()
            test_id[n] = tmp[n].tolist()
            if c < add_one:
                test_id[n].append(list_one.pop(c))
                c += 1
            train_id[n] = np.delete(tmp, n, 0).flatten().tolist()
            train_id[n] += list_one

    return train_id, test_id


# train,val = idsplit(180, 5, shuffle=False)
# print(train[4])
# print(val[4])