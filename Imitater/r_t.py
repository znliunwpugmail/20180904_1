import os
import numpy as np
def find_index(list_user,elem):
    for i in range(list_user.count(elem)):
        first_index = list_user.index(elem)
        # second_list = list_user[]
        second_index = list_user.index(first_index)


if __name__ == '__main__':

    U_list = []
    inout_list = []
    user_id_list = []
    e_list = []
    actions = []

    with open("user_imitater_u_e_final.txt",'r') as f:
        line = f.readline()
        print(line)
        while line:
            content_str = line.split('\t')

            user_id = content_str[1]
            inout = content_str[2]
            U = content_str[3:6]
            user_id_list.append(int(user_id))
            inout_list.append(inout)

            U_list.append(U)
            e_list.append(int(content_str[6])-int(content_str[7]))
            U_list.append(U)
            line = f.readline()

    print(len(U_list))
    state_index = 0
    in_num = 0
    out_num = 0
    capacility = np.zeros(shape=[3], dtype=np.int)

    while True:
        state_index = 0
        inout = inout_list[state_index]
        if inout == 0:
            pass
