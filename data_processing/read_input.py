


def read_input(input_path):
    """
    :param input_path:
    :return:
    a list of input sequences
    """
    Input_List=[]
    with open(input_path,'r') as file:
        line=file.readline()
        while line:
            line=line.strip("\n")
            sequence=line
            Input_List.append(sequence)
            line=file.readline()
    return Input_List

def verify_input(input_list,seq_len):
    final_input_list=[]
    for item in input_list:
        if len(item)!=seq_len:
            continue
        final_input_list.append(item)
    return final_input_list
