"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import codecs
import copy
import pdb

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, evaluation):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.filename  = filename
        # ************* item_id *****************
        opt["source_item_num"] = self.read_item("./dataset/" + filename + "/Alist.txt")
        opt["target_item_num"] = self.read_item("./dataset/" + filename + "/Blist.txt")

        # ************* sequential data *****************

        source_train_data = "./dataset/" + filename + "/traindata_new.txt"
        source_valid_data = "./dataset/" + filename + "/validdata_new.txt"
        source_test_data = "./dataset/" + filename + "/testdata_new.txt"

        if evaluation < 0:
            self.train_data = self.read_train_data(source_train_data)
            #print(len(self.train_data))
            #input()
            data,user_indices = self.preprocess(0)
        elif evaluation == 2:
            self.test_data = self.read_test_data(source_valid_data)
            data,user_indices = self.preprocess_for_predict(opt['train_user_num']+opt['test_user_num'])
        elif evaluation == 1:
            self.test_data = self.read_test_data(source_test_data)
            data,user_indices = self.preprocess_for_predict(opt['train_user_num'])
        else:
            self.test_data = self.read_test_data(source_train_data)
            data,user_indices = self.preprocess_for_predict(0)



        #user_indices=np.arange(len(data))
        #print(data[-1],user_indices[-1],len(data),len(user_indices))
        # shuffle for training
        if evaluation == -1:
            user_indices=np.arange(len(data))
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            user_indices = [user_indices[i] for i in indices]
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data)%batch_size != 0:
                user_indices += user_indices[:(batch_size-len(data)%batch_size)]
                data += data[:(batch_size-len(data)%batch_size)]
            data = data[: (len(data)//batch_size) * batch_size]
            user_indices = user_indices[: (len(user_indices)//batch_size) * batch_size]
        else :
            batch_size = 4096
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        user_indices = [user_indices[i:i+batch_size] for i in range(0, len(user_indices), batch_size)]
        print("len(data)",len(data[0][0])) #should be 19
        self.data = data
        self.user_indices = user_indices
        #input()

    def read_item(self, fname):
        item_number = 0
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            for line in fr:
                item_number += 1
        return item_number

    def read_train_data(self, train_file):
        def takeSecond(elem):
            return elem[1]
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            for id, line in enumerate(infile):
                res = []

                line = line.strip().split("\t")[2:]
                for w in line:
                    w = w.split("|")
                    res.append((int(w[0]), int(w[1])))
                res.sort(key=takeSecond)
                res_2 = []
                for r in res:
                    res_2.append(r[0])
                train_data.append(res_2)

        return train_data

    def read_test_data(self, test_file):
        def takeSecond(elem):
            return elem[1]
        with codecs.open(test_file, "r", encoding="utf-8") as infile:
            test_data = []
            for id, line in enumerate(infile):
                res = []
                line = line.strip().split("\t")[2:]
                for w in line:
                    w = w.split("|")
                    res.append((int(w[0]), int(w[1])))

                res.sort(key=takeSecond)

                res_2 = []
                for r in res[:-1]:
                    res_2.append(r[0])

                if res[-1][0] >= self.opt["source_item_num"]: # denoted the corresponding validation/test entry
                    test_data.append([res_2, 1, res[-1][0]])
                else :
                    test_data.append([res_2, 0, res[-1][0]])
        return test_data

    def preprocess_for_predict(self,padding_index):

        if "Enter" in self.filename:
            max_len = 30
            self.opt["maxlen"] = 30
        else:
            max_len = 15
            self.opt["maxlen"] = 15

        processed=[]
        user_indices=[]
        idx_idx=padding_index
        for d in self.test_data: # the pad is needed! but to be careful.
            position = list(range(len(d[0])+1))[1:]

            xd = []
            xcnt = 1
            x_position = []

            yd = []
            ycnt = 1
            y_position = []

            for w in d[0]:
                if w < self.opt["source_item_num"]:
                    xd.append(w)
                    x_position.append(xcnt)
                    xcnt += 1
                    yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    y_position.append(0)

                else:
                    xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    x_position.append(0)
                    yd.append(w)
                    y_position.append(ycnt)
                    ycnt += 1

            #print(len(d[0]),max_len)
            if len(d[0]) < max_len:
                position = [0] * (max_len - len(d[0])) + position
                x_position = [0] * (max_len - len(d[0])) + x_position
                y_position = [0] * (max_len - len(d[0])) + y_position

                xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d[0])) + xd
                yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d[0])) + yd
                seq = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(d[0])) + d[0]


            x_last = -1
            for id in range(len(x_position)):
                id += 1
                if x_position[-id]:
                    x_last = -id
                    break

            y_last = -1
            for id in range(len(y_position)):
                id += 1
                if y_position[-id]:
                    y_last = -id
                    break

            negative_sample = []
            for i in range(999):
                while True:
                    if d[1] : # in Y domain, the validation/test negative samples
                        sample = random.randint(0, self.opt["target_item_num"] - 1)
                        if sample != d[2] - self.opt["source_item_num"]:
                            negative_sample.append(sample)
                            break
                    else : # in X domain, the validation/test negative samples
                        sample = random.randint(0, self.opt["source_item_num"] - 1)
                        if sample != d[2]:
                            negative_sample.append(sample)
                            break


            if d[1]:
                processed.append([seq, xd, yd, position, x_position, y_position, x_last, y_last, d[1], d[2]-self.opt["source_item_num"], negative_sample,idx_idx])
            else:
                processed.append([seq, xd, yd, position, x_position, y_position, x_last, y_last, d[1],
                                  d[2], negative_sample,idx_idx])
            user_indices.append(idx_idx)
            idx_idx+=1
        return processed, user_indices

    def preprocess(self,padding_index):

        def myprint(a):
            for i in a:
                print("%6d" % i, end="")
            print("")
        """ Preprocess the data and convert to ids. """
        processed = []


        if "Enter" in self.filename:
            max_len = 30
            self.opt["maxlen"] = 30
        else:
            max_len = 15
            self.opt["maxlen"] = 15
        idx_idx=padding_index-1
        user_indicies=[]
        for d in self.train_data: # the pad is needed! but to be careful.
            idx_idx+=1
            ground = copy.deepcopy(d)[1:]


            share_x_ground = []
            share_x_ground_mask = []
            share_y_ground = []
            share_y_ground_mask = []
            for w in ground:
                if w < self.opt["source_item_num"]: #with_gtruth mask=[0,1,0,1] ,gound=[12,max,43,max]
                    share_x_ground.append(w)
                    share_x_ground_mask.append(1)
                    share_y_ground.append(self.opt["target_item_num"])
                    share_y_ground_mask.append(0)
                else:
                    share_x_ground.append(self.opt["source_item_num"])
                    share_x_ground_mask.append(0)
                    share_y_ground.append(w - self.opt["source_item_num"])
                    share_y_ground_mask.append(1)


            d = d[:-1]  # delete the ground truth
            position = list(range(len(d)+1))[1:]
            ground_mask = [1] * len(d)



            xd = []
            xcnt = 1
            x_position = []


            yd = []
            ycnt = 1
            y_position = []

            corru_x = []
            corru_y = []

            for w in d:
                #without _gt
                if w < self.opt["source_item_num"]:
                    corru_x.append(w)
                    xd.append(w)
                    x_position.append(xcnt)
                    xcnt += 1
                    corru_y.append(random.randint(0, self.opt["source_item_num"] - 1))
                    yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    y_position.append(0)

                else:
                    corru_x.append(random.randint(self.opt["source_item_num"], self.opt["source_item_num"] + self.opt["target_item_num"] - 1))
                    xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    x_position.append(0)
                    corru_y.append(w)
                    yd.append(w)
                    y_position.append(ycnt)
                    ycnt += 1

            now = -1
            x_ground = [self.opt["source_item_num"]] * len(xd) # caution!
            x_ground_mask = [0] * len(xd)
            for id in range(len(xd)):
                id+=1
                if x_position[-id]:
                    if now == -1:
                        now = xd[-id]
                        if ground[-1] < self.opt["source_item_num"]:
                            x_ground[-id] = ground[-1]
                            x_ground_mask[-id] = 1
                        else:
                            xd[-id] = self.opt["source_item_num"] + self.opt["target_item_num"]
                            x_position[-id] = 0
                    else:
                        x_ground[-id] = now
                        x_ground_mask[-id] = 1
                        now = xd[-id]
            if sum(x_ground_mask) == 0:
                #print("pass sequence x")
                continue

            now = -1
            y_ground = [self.opt["target_item_num"]] * len(yd) # caution!
            y_ground_mask = [0] * len(yd)
            for id in range(len(yd)):
                id+=1
                if y_position[-id]:
                    if now == -1:
                        now = yd[-id] - self.opt["source_item_num"]
                        if ground[-1] > self.opt["source_item_num"]:
                            y_ground[-id] = ground[-1] - self.opt["source_item_num"]
                            y_ground_mask[-id] = 1
                        else:
                            yd[-id] = self.opt["source_item_num"] + self.opt["target_item_num"]
                            y_position[-id] = 0
                    else:
                        y_ground[-id] = now
                        y_ground_mask[-id] = 1
                        now = yd[-id] - self.opt["source_item_num"]
            if sum(y_ground_mask) == 0:
                #print("pass sequence y")
                continue

            if len(d) < max_len:
                #paddings
                position = [0] * (max_len - len(d)) + position
                x_position = [0] * (max_len - len(d)) + x_position
                y_position = [0] * (max_len - len(d)) + y_position

                ground = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + ground
                share_x_ground = [self.opt["source_item_num"]] * (max_len - len(d)) + share_x_ground
                share_y_ground = [self.opt["target_item_num"]] * (max_len - len(d)) + share_y_ground
                x_ground = [self.opt["source_item_num"]] * (max_len - len(d)) + x_ground
                y_ground = [self.opt["target_item_num"]] * (max_len - len(d)) + y_ground

                ground_mask = [0] * (max_len - len(d)) + ground_mask
                share_x_ground_mask = [0] * (max_len - len(d)) + share_x_ground_mask
                share_y_ground_mask = [0] * (max_len - len(d)) + share_y_ground_mask
                x_ground_mask = [0] * (max_len - len(d)) + x_ground_mask
                y_ground_mask = [0] * (max_len - len(d)) + y_ground_mask

                corru_x = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + corru_x
                corru_y = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + corru_y
                xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + xd
                yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + yd
                d = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + d
            else:
                pass
                #print("pass")
            #print([d,	xd, yd, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y])
            #train_data:    0	14	0|0|0	28442|1|1	1|2|2	22560|3|3	2|4|4	18805|5|5	3|6|6	20729|7|7	4|8|8	21806|9|9	5|10|10	6|11|11	7|12|12	8|13|13	22212|14|14	
            #d              [38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	0,	28442,	1,	22560,	2,	18805,	3,	20729,	4,	21806,	5,	6,	7,	8],	
            #xd             [38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	0,	38396,	1,	38396,	2,	38396,	3,	38396,	4,	38396,	5,	6,	7,	38396],	
            #yd             [38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	28442,	38396,	22560,	38396,	18805,	38396,	20729,	38396,	21806,	38396,	38396,	38396,	38396],	
            #position       [0,	    0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		1,		2,		3,		4,		5,		6,		7,		8,		9,		10,		11,		12,		13,		14],		
            #x_position     [0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		1,		0,		2,		0,		3,		0,		4,		0,		5,		0,		6,		7,		8,		0],		
            #y_position     [0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		1,		0,		2,		0,		3,		0,		4,		0,		5,		0,		0,		0,		0],		
            #ground         [38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	28442,	1,	22560,	2,	18805,	3,	20729,	4,	21806,	5,	6,	7,	8,	22212],	
            #share_x_ground [18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	1,	18639,	2,	18639,	3,	18639,	4,	18639,	5,	6,	7,	8,	18639],	
            #share_y_ground [19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	9803,	19757,	3921,	19757,	166,	19757,	2090,	19757,	3167,	19757,	19757,	19757,	19757,	3573],	
            #x_ground       [18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	18639,	1,	18639,	2,	18639,	3,	18639,	4,	18639,	5,	18639,	6,	7,	8,	18639],	
            #y_ground       [19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	19757,	3921,	19757,	166,	19757,	2090,	19757,	3167,	19757,	3573,	19757,	19757,	19757,	19757],	
            #ground_mask    [0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		1,		1,		1,		1,		1,		1,		1,		1,		1,		1,		1,		1,		1,		1],		
    #share_x_ground_mask    [0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		1,		0,		1,		0,		1,		0,		1,		0,		1,		1,		1,		1,		0],		
    #share_y_ground_mask    [0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		1,		0,		1,		0,		1,		0,		1,		0,		1,		0,		0,		0,		0,		1],		
            #x_ground_mask  [0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		1,		0,		1,		0,		1,		0,		1,		0,		1,		0,		1,		1,		1,		0],		
            #y_ground_mask  [0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		0,		1,		0,		1,		0,		1,		0,		1,		0,		1,		0,		0,		0,		0],	
            #corru_x        [38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	0,	36933,	1,	37327,	2,	34511,	3,	35375,	4,	23884,	5,	6,	7,	8],	
            #corru_y        [38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	38396,	12513,	28442,	10832,	22560,	16259,	18805,	13349,	20729,	11437,	21806,	5482,	1405,	1426,   7540]]
            #input()
            processed.append([d, xd, yd, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y,idx_idx])
            user_indicies.append(idx_idx)
        return processed,user_indicies

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        if self.eval!=-1:
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.LongTensor(batch[6]), torch.LongTensor(batch[7]),torch.LongTensor(batch[8]), torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]))
        else :
            batch = list(zip(*batch))

            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.LongTensor(batch[6]), torch.LongTensor(batch[7]),torch.LongTensor(batch[8]), torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]), torch.LongTensor(batch[12]), torch.LongTensor(batch[13]), torch.LongTensor(batch[14]), torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]), torch.LongTensor(batch[18]))

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


