import json
from typing import List,Dict
def filter_list_of_dicts_by_string_in_key(list_of_dicts:List,key_name:str,string_value:str)->List:
    result = []
    for dict in list_of_dicts:
        if key_name in dict and isinstance(dict[key_name], str) and string_value in dict[key_name]:
            result.append(dict)
    return result

def get_values_by_key(dictionary_list, key_name)->List:
    values = []
    for dictionary in dictionary_list:
        if key_name in dictionary:
            values.append(dictionary[key_name])
    return values