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
    return {'ids': (ids1, ids2), 'cls': cls, 'permmat': output}
    