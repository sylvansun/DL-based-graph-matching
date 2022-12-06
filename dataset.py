from pygmtools.dataset import WillowObject
from pygmtools.benchmark import Benchmark
from jittor.dataset import Dataset
import jittor as jt
from utils.gmfunctions import delaunay_triangulation


class GraphPair(Dataset):
    def __init__(self, benchmark, sets="train", num_classes=5, obj_resize=(256,256), batch_size=32, shuffle=True, drop_last=True):
        super(GraphPair, self).__init__(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        self.data = []
        self.label = []
        self.benchmark = benchmark
        self.sets = sets
        self.obj_resize = obj_resize
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.load_data()
        self.set_attrs(total_len=len(self.data))
        
    def load_data(self):
        graph_pair_list = benchmark.get_id_combination()[0]
        for i in range(self.num_classes):
            for elem in graph_pair_list[i]:
                data_list, perm_mat_dict, ids = benchmark.get_data(list(elem), shuffle=True)
                img1, img2 = data_list[0]["img"], data_list[1]["img"]
                img1, img2 = jt.float32(img1).permute(2, 0, 1) / 256, jt.float32(img2).permute(2, 0, 1) / 256
                kpts1 = jt.float32([(kp["x"], kp["y"]) for kp in data_list[0]["kpts"]]).transpose()
                kpts2 = jt.float32([(kp["x"], kp["y"]) for kp in data_list[1]["kpts"]]).transpose()
                A1, A2 = delaunay_triangulation(kpts1), delaunay_triangulation(kpts2)
                self.label.append(perm_mat_dict[(0,1)].toarray())
                self.data.append((img1, img2, kpts1, kpts2, A1, A2, ids))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_item = self.data[index]
        label_item = self.label[index]
        return data_item[0], data_item[1], data_item[2], data_item[3], data_item[4], data_item[5], data_item[6], label_item
                
        


if __name__ == "__main__":
    benchmark = Benchmark(name="WillowObject", sets="train")
    train_data = GraphPair(benchmark, sets="train", batch_size=32, shuffle=True, drop_last=True)
    
    for batch_idx, (img1, img2, kpts1, kpts2, A1, A2, ids, label) in enumerate(train_data):
        print(img1.shape)
        print(img2.shape)
        print(kpts1.shape)
        print(kpts2.shape)
        print(A1.shape)
        print(A2.shape)
        print(label.shape)
        print(len(ids[0]), len(ids[1]))
        break
    