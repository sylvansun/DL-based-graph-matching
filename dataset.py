from pygmtools.dataset import WillowObject
from pygmtools.benchmark import Benchmark
import scipy.io as sio
from PIL import Image
from jittor.dataset import Dataset

class GraphPair(Dataset):
    def __init__(self, benchmark, sets="train", obj_resize = (256,256), batch_size=32, shuffle=True, drop_last=True):
        super(GraphPair, self).__init__(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        self.data = []
        self.benchmark = benchmark
        self.sets = sets
        self.obj_resize = obj_resize
        self.batch_size = batch_size
        
        self.load_data()
        
    def load_data(self):
        graph_pair_list = benchmark.get_id_combination()
        


if __name__ == "__main__":
    benchmark = Benchmark(name="WillowObject", sets="train", obj_resize=obj_resize)
    
    test_data = ["Cars_000a", "Cars_007b"]
    data_list, perm_mat_dict, ids = benchmark.get_data(test_data, shuffle=True)
    print(len(data_list))
    print(type(data_list))
    print(data_list[0].keys())
    print(data_list[0]["img"].shape)
    print(data_list[0]["kpts"])
    print(data_list[1]["kpts"])
    print(data_list[0]["cls"])
    print(data_list[0]["univ_size"])
    print(perm_mat_dict.keys())
    print(perm_mat_dict[(0, 1)])
    print(ids)
    
    kpts1 = sio.loadmat('./data/WillowObject/WILLOW-ObjectClass/Duck/060_0001.mat')['pts_coord']
    img1 = Image.open('./data/WillowObject/WILLOW-ObjectClass/Duck/060_0001.png')
    kpts1[0] = kpts1[0] * obj_resize[0] / img1.size[0]
    kpts1[1] = kpts1[1] * obj_resize[1] / img1.size[1]
    print(kpts1)

    id_comb = benchmark.get_id_combination()
    print(len(id_comb))
    # print(id_comb)
    # print(type(id_comb[0][0]))
    # print(id_comb[0][0][0])