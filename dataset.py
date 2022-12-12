from pygmtools.dataset import WillowObject
from pygmtools.benchmark import Benchmark
from jittor.dataset import Dataset
import jittor as jt
from utils.gmfunctions import delaunay_triangulation
from utils.tools import test_dataloading, show_dataset_attributes


class GraphPair(Dataset):
    def __init__(self, 
                 sets="train", 
                 num_classes=5, 
                 obj_resize=(256,256), 
                 batch_size=32, 
                 shuffle=True, 
                 drop_last=True):
        super(GraphPair, self).__init__(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        self.data = []
        self.label = []
        self.sets = sets
        self.shuffle = shuffle
        self.benchmark = Benchmark(name="WillowObject", sets=self.sets)
        self.obj_resize = obj_resize
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.name_classes = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]
        self.load_data_list()
        self.set_attrs(total_len=len(self.data))
        
    def load_data_list(self):
        graph_pair_list = self.benchmark.get_id_combination()[0]
        for i in range(self.num_classes):
            cur_cls = self.name_classes[i]
            for elem in graph_pair_list[i]:
                self.data.append((list(elem), cur_cls))
    
    def load_data(self):
        graph_pair_list = self.benchmark.get_id_combination()[0]
        for i in range(self.num_classes):
            cur_cls = self.name_classes[i]
            for elem in graph_pair_list[i]:
                data_list, perm_mat_dict, ids = self.benchmark.get_data(list(elem), shuffle=True)
                img1, img2 = data_list[0]["img"], data_list[1]["img"]
                img1, img2 = jt.float32(img1).permute(2, 0, 1) / 256, jt.float32(img2).permute(2, 0, 1) / 256
                kpts1 = jt.float32([(kp["x"], kp["y"]) for kp in data_list[0]["kpts"]]).transpose()
                kpts2 = jt.float32([(kp["x"], kp["y"]) for kp in data_list[1]["kpts"]]).transpose()
                A1, A2 = delaunay_triangulation(kpts1), delaunay_triangulation(kpts2)
                self.label.append(perm_mat_dict[(0,1)].toarray())
                self.data.append((img1, img2, kpts1, kpts2, A1, A2, ids, cur_cls))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_list, perm_mat_dict, ids = self.benchmark.get_data(self.data[index][0], shuffle=self.shuffle)
        cls = self.data[index][1] # class name, string
        img1, img2 = data_list[0]["img"], data_list[1]["img"]
        img1, img2 = jt.float32(img1).permute(2, 0, 1) / 256, jt.float32(img2).permute(2, 0, 1) / 256
        kpts1 = jt.float32([(kp["x"], kp["y"]) for kp in data_list[0]["kpts"]]).transpose()
        kpts2 = jt.float32([(kp["x"], kp["y"]) for kp in data_list[1]["kpts"]]).transpose()
        A1, A2 = delaunay_triangulation(kpts1), delaunay_triangulation(kpts2)
        label = perm_mat_dict[(0,1)].toarray()
        return img1, img2, kpts1, kpts2, A1, A2, ids, cls, label 

          
if __name__ == "__main__":
    train_data = GraphPair(sets="train", batch_size=32, shuffle=True)
    test_data = GraphPair(sets="test", batch_size=32, shuffle=False)
    
    test_dataloading(train_data)
    test_dataloading(test_data)
    
    show_dataset_attributes(train_data)
    show_dataset_attributes(test_data)
    
    