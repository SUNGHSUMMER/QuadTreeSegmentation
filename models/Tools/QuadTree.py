"""
整体进行完global的预测之后，先将结果保存下来
然后按照channel进行切块，对每一块进行后续的处理部分
"""

class QuadTree():
    def __init__(self):
        super(QuadTree, self).__init__()
        self.base_image = None
        self.base_predict = None
        self.predict00 = None
        self.predict01 = None
        self.predict10 = None
        self.predict11 = None


# input_stage1 = rearrange(image, 'b c (p1 h) (p2 w) -> b c h w (p1 p2)', p1=2, p2=2)
