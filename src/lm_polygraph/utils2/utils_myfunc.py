import json

def transfer_dict_list_format(dict_list, input_text_key='input_texts'):
    batch_size = len(dict_list[input_text_key])
    res_list = []
    for i in range(batch_size):
        mid_dict = {}
        for k, v in dict_list.items():
            try:
                if len(v) == batch_size:
                    mid_dict[k] = v[i]
            except:
                continue
        res_list.append(mid_dict)
    return res_list

def save_list_dict_to_json(list_dict, file_name):
    batch_size = len(list_dict)
    with open(file_name, 'a+') as f:
        for i in range(batch_size):
            json_str = json.dumps(
                list_dict[i]
            )
            f.write(json_str)
            f.write('\n')