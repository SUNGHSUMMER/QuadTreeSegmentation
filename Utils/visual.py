from prettytable import PrettyTable


def show_score_as_table(*score):
    """
    show score as table

    """
    data_length = len(score)
    tb = PrettyTable()
    if data_length == 2:

        train_score = score[0]
        train_iou = train_score["iou"]
        train_miou = train_score["iou_mean"]

        val_score = score[1]
        val_iou = val_score["iou"]
        val_miou = val_score["iou_mean"]

        tb.field_names = ["type", "train", "val"]
        tb.add_row(["mIoU", train_miou, val_miou])
        tb.add_row(["urban", train_iou[0], val_iou[0]])
        tb.add_row(["agriculture", train_iou[1], val_iou[1]])
        tb.add_row(["rangeland", train_iou[2], val_iou[2]])
        tb.add_row(["forest", train_iou[3], val_iou[3]])
        tb.add_row(["water", train_iou[4], val_iou[4]])
        tb.add_row(["barren", train_iou[5], val_iou[5]])
        print(tb)
    if data_length == 1:
        test_score = score[0]
        test_iou = test_score["iou"]
        test_miou = test_score["iou_mean"]

        tb.field_names = ["type", "test"]
        tb.add_row(["mIoU", test_miou])
        tb.add_row(["urban", test_iou[0]])
        tb.add_row(["agriculture", test_iou[1]])
        tb.add_row(["rangeland", test_iou[2]])
        tb.add_row(["forest", test_iou[3]])
        tb.add_row(["water", test_iou[4]])
        tb.add_row(["barren", test_iou[5]])
        print(tb)

