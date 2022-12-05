from pygmtools.dataset import WillowObject
from pygmtools.benchmark import Benchmark
import json



if __name__ == "__main__":
    obj_resize = (256, 256)
    data = WillowObject(sets='train', obj_resize=obj_resize)
    benchmark = Benchmark(name="WillowObject", sets="train", obj_resize=obj_resize)
    
    test_data = ["Cars_006a", "Cars_006b"]
    data_list, perm_mat_dict, ids = benchmark.get_data(test_data)
    print(len(data_list))
    print(type(data_list))
    print(data_list[0].keys())
    print(perm_mat_dict.keys())
    print(perm_mat_dict[(0, 1)])
    print(ids)
