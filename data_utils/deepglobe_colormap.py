import numpy as np
import os

from PIL import Image


def deepglobe_color_map():
    """


    Returns the DeepGlobe color map
    -------

    """
    colorize = np.zeros([7, 3], dtype=np.int64)
    colorize[0, :] = [0, 255, 255]  # 城市 青色
    colorize[1, :] = [255, 255, 0]  # 农业 黄色
    colorize[2, :] = [255, 0, 255]  # 牧场 紫色
    colorize[3, :] = [0, 255, 0]  # 森林 绿色
    colorize[4, :] = [0, 0, 255]  # 水 蓝色
    colorize[5, :] = [255, 255, 255]  # 贫瘠 白色
    colorize[6, :] = [0, 0, 0]  # 未知 黑色

    return colorize


def gt2label():
    """
    convert ground truth images to the label images
    """

    color_maps = deepglobe_color_map()
    classes = len(color_maps)

    root = r"/home/guohao/codes/MyProject/FCtL/DeepGlobe/data"
    gth_path = os.path.join(root, "gt")
    label_path = os.path.join(root, "label")

    if not os.path.exists(label_path):
        os.makedirs(label_path)

    filename = os.listdir(gth_path)

    for name in filename:
        gth_name = os.path.join(gth_path, name)
        save_path = os.path.join(label_path, name)

        pic = Image.open(gth_name)
        pic = np.array(pic).astype(np.uint8)

        h = pic.shape[0]
        w = pic.shape[1]

        new_pic = np.zeros([h, w], dtype=int)
        for i in range(h):
            for j in range(w):
                rgb = pic[i, j, :]
                for k in range(classes):
                    if (color_maps[k, :] == rgb).all():
                        new_pic[i, j] = k
                        break
        print(new_pic.shape)
        im = Image.fromarray(np.int8(new_pic))
        im.save(save_path)
        print("finished convert of " + name)

    return 0


if __name__ == "__main__":
    gt2label()
