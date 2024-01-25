import codecs

def read_item(fname):
    item_number = 0
    with codecs.open(fname,"r",encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            item_number+=1
    print(fname+" has "+str(item_number)+" items!")

def read_user(fname):
    item_number = 0
    with codecs.open(fname,"r",encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            # print(line)
            # input()
            item_number+=1
    print(fname+" has "+str(item_number)+" users!")

def read_data(fname):
    item_number = 0
    mx = 0
    s_user = set()
    s_inter = set()
    s_border = set()
    mx_length = 0
    with codecs.open(fname,"r",encoding="utf-8") as fr:
        for line in fr:
            s_user.add(int(line.strip().split("\t")[0]))
            s_inter.add(int(line.strip().split("\t")[1]))
            line = line.strip().split("\t")[2:]
            s_border.add(int(line[-1].split("|")[0]))
            mx_length = max(mx_length, len(line))
            for w in line:
                w = w.split("|")
                mx = max(mx, int(w[0]))
            # print(line)
            # print(len(line))
            item_number+=1

    print(fname + " has max id item: " + str(max(s_border)))
    print(fname + " has " + str(item_number) + " interactions!")
    print(fname+" has user max: "+ str(max(s_user)))
    print(fname + " has inter max: " + str(max(s_user)))
    print(fname + " has border max: " + str(min(s_border)))
    print(fname + " has inter length max: " + str(mx_length))
    print()
    print()

dataset = "Food-Kitchen"
# dataset = "Movie-Book"
# dataset = "Entertainment-Education"

read_item(dataset+"/Alist.txt")
read_item(dataset+"/Blist.txt")

read_user(dataset+"/userlist.txt")

# read_data(dataset+"/traindata.txt")
# read_data(dataset+"/validdata.txt")
# read_data(dataset+"/testdata.txt")


read_data(dataset+"/traindata_new.txt")
read_data(dataset+"/validdata_new.txt")
read_data(dataset+"/testdata_new.txt")

"""
Food-Kitchen/Alist.txt has 29207 items!
Food-Kitchen/Blist.txt has 34886 items!

Food-Kitchen/userlist.txt has 16579 users!

Food-Kitchen/traindata.txt has 25766 interactions!
Food-Kitchen/validdata.txt has 7650 interactions!
Food-Kitchen/testdata.txt has 17280 interactions!


64093



Movie-Book/Alist.txt has 36845 items!
Movie-Book/Blist.txt has 63937 items!

Movie-Book/userlist.txt has 15352 users!

Movie-Book/traindata.txt has 44732 interactions!
Movie-Book/validdata.txt has 9274 interactions!
Movie-Book/testdata.txt has 19861 interactions!

100776
"""
