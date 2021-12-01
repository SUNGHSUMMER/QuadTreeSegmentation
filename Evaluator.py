class Evaluator(object):
    def __init__(self, n_class, size_p, size_g, sub_batch_size=6, mode=1, val=True, dataset=1, context10=2,
                 context15=3):
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.size_p = size_p
        self.size_g = size_g
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.val = val
        self.context10 = context10
        self.context15 = context15
        if not val:
            self.flip_range = [False, True]
            self.rotate_range = [0, 1, 2, 3]
        else:
            self.flip_range = [False]
            self.rotate_range = [0]

    def get_scores(self):
        score = self.metrics.get_scores()
        return score

    def reset_metrics(self):
        self.metrics.reset()

    def eval_test(self, sample, model, global_fixed_10, global_fixed_15):
        with torch.no_grad():
            images = sample['image']
            if self.val:
                labels = sample['label']  # PIL images
                labels_npy = masks_transform(labels, numpy=True)

            images = [image.copy() for image in images]
            scores = [np.zeros((1, self.n_class, images[i].size[1], images[i].size[0])) for i in range(len(images))]
            for flip in self.flip_range:
                if flip:
                    # we already rotated images for 270'
                    for b in range(len(images)):
                        images[b] = transforms.functional.rotate(images[b], 90)  # rotate back!
                        images[b] = transforms.functional.hflip(images[b])
                for angle in self.rotate_range:
                    if angle > 0:
                        for b in range(len(images)):
                            images[b] = transforms.functional.rotate(images[b], 90)
                    # prepare global images onto cuda

                    patches, coordinates, templates, sizes, ratios = global2patch(images, self.size_p)
                    predicted_patches = [np.zeros((len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1]))
                                         for i in range(len(images))]

                    if self.mode == 2 or self.mode == 3:
                        big_patches_10 = global2bigpatch(images, self.size_p, self.context10)
                    if self.mode == 3:
                        big_patches_15 = global2bigpatch(images, self.size_p, self.context15)
                    # eval with patches ###########################################
                    for i in range(len(images)):
                        j = 0
                        while j < len(coordinates[i]):
                            patches_var = images_transform(patches[i][j: j + self.sub_batch_size])  # b, c, h, w
                            big_patches_10_var = None
                            if self.mode == 2 or self.mode == 3:
                                big_patches_10_var = images_transform(big_patches_10[i][j: j + self.sub_batch_size])

                            if self.mode == 1 or self.mode == 2:
                                output_patches = model.forward(patches_var, y=big_patches_10_var)
                            else:  ##3
                                pool5_10 = global_fixed_10.forward(big_patches_10_var)
                                big_patches_15_var = images_transform(big_patches_15[i][j: j + self.sub_batch_size])
                                pool5_15 = global_fixed_15.forward(big_patches_15_var)
                                output_patches = model.forward(patches_var, pool5_10, pool5_15)
                            # patch predictions
                            predicted_patches[i][j:j + output_patches.size()[0]] += F.interpolate(output_patches,
                                                                                                  size=self.size_p,
                                                                                                  mode='nearest').data.cpu().numpy()
                            j += patches_var.size()[0]
                        if flip:
                            scores[i] += np.flip(np.rot90(np.array(
                                patch2global(predicted_patches[i:i + 1], self.n_class, sizes[i:i + 1],
                                             coordinates[i:i + 1], self.size_p)), k=angle, axes=(3, 2)),
                                axis=3)  # merge softmax scores from patches (overlaps)
                        else:
                            scores[i] += np.rot90(np.array(
                                patch2global(predicted_patches[i:i + 1], self.n_class, sizes[i:i + 1],
                                             coordinates[i:i + 1], self.size_p)), k=angle,
                                axes=(3, 2))  # merge softmax scores from patches (overlaps)
                    ###############################################################

            # patch predictions ###########################
            predictions = [score.argmax(1)[0] for score in scores]
            if self.val:
                self.metrics.update(labels_npy, predictions)
            ###################################################
            return predictions