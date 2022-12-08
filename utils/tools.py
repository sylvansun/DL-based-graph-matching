import numpy as np
from scipy.sparse import coo_matrix

def show_dataset_attributes(dataset):
    print("dataset name: ", dataset.sets)
    print("benchmark length: ", dataset.benchmark.compute_length())
    print("dataset length: ", len(dataset))
    print("number of image in each classes: ", dataset.benchmark.compute_img_num(dataset.name_classes))
    
def test_dataloading(dataset):
    for _, (img1, img2, kpts1, kpts2, A1, A2, ids, cls, label) in enumerate(dataset):
        print(img1.shape, img2.shape)
        print(kpts1.shape, kpts2.shape)
        print(A1.shape, A2.shape)
        print(label.shape)
        print(len(ids[0]), len(ids[1]))
        print(cls)
        break

def generate_pred_dict(output, ids1, ids2, cls):
    return {'ids': (ids1, ids2), 'cls': cls, 'perm_mat': output}
    

def save_eval_result(result, classes, file):
    file.write('Matching accuracy')
    for cls in classes:
        file.write('{}: {}'.format(cls, 'p = {:.4f}±{:.4f}, r = {:.4f}±{:.4f}, f1 = {:.4f}±{:.4f}, cvg = {:.4f}' \
                                      .format(result[cls]['precision'], result[cls]['precision_std'],
                                              result[cls]['recall'], result[cls]['recall_std'], result[cls]['f1'],
                                              result[cls]['f1_std'], result[cls]['coverage']
                                              )))
    file.write('average accuracy: {}'.format('p = {:.4f}±{:.4f}, r = {:.4f}±{:.4f}, f1 = {:.4f}±{:.4f}' \
                                                .format(result['mean']['precision'], result['mean']['precision_std'],
                                                        result['mean']['recall'], result['mean']['recall_std'],
                                                        result['mean']['f1'], result['mean']['f1_std']
                                                        )))