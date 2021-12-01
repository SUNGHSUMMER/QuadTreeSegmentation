def txt2list(txt):
    f = open(txt, 'r')
    row_list = list(f)
    data_list = [each.replace("\n", "") for each in row_list]
    return data_list


if __name__ == "__main__":
    train = "../DeepGlobe/data_split/train.txt"
    val = "../DeepGlobe/data_split/crossvali.txt"
    test = "../DeepGlobe/data_split/test.txt"
    a = txt2list(train)
    b = txt2list(val)
    c = txt2list(test)
    print(c)


