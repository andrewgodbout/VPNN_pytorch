import torch
import torch.utils.data
from nltk.corpus import stopwords
import nltk

"""
returns custom pytorch datasets for aclimdb
takes in list of review dicts, constructs tuple of tensor (inp, label)
inp_size is the size of the input vector being returned
also returns a dict that maps inp vector index to word
and another dict that maps word to inp vector index
lower indices are more common

download at https://ai.stanford.edu/~amaas/data/sentiment/
expects folder aclImdb in DATA_PATH

downloads stopwords if not found from nltk

parses from labeledBow.feat in train and test, and imdb.vocab
"""

class AclImdb_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, inp_size, preprocessing=None):
        self.data = data
        self.preprocessing = preprocessing
        self.inp_size = inp_size

    def __getitem__(self, idx):
        inp = torch.zeros(self.inp_size)
        label = torch.tensor(self.data[idx]['score'])
        # map label to positive or negative, ternary operator
        label = torch.where(label>=7, torch.ones(1), torch.zeros(1)).long()[0]
        
        # most common words have the smallest indicies thankfully
        # converts dict to one hot taking smallest indicies up to some number
        # could do in overhead, but very memory expensive ~144 GB max
        try:
            indices, values = zip(
                  *filter(lambda x: x[0] != 'score', self.data[idx].items())
                  )
            inp[list(indices)] = torch.tensor(list(values)).float()
            
        except ValueError: # case of no words in review in top inp_size words
            pass 

        # above code equivalent to the following, aprox 13x slower
#        for i in range(self.inp_size):
#            try:
#                inp[i] = self.data[idx][i]
#            except KeyError:
#                pass

        if self.preprocessing is not None:
            inp = self.preprocessing(inp)
        return (inp, label)

    def __len__(self):
        return len(self.data)

def aclimdb_load(DATA_PATH, inp_size, preprocessing=None, remove_stopwords=True):
    vocab_map = dict()  # maps word -> num
    num_map = dict()  # maps num -> word
    old_new_map = dict()  # for moving indicices, maps old -> new
    stop_nums = []  # optional stop words to remove
    test_reviews = []  # list of dicts of test reviews
    train_reviews = []  # list of dicts of train reviews
    
    # load vocab_map, num_map
    f = open(DATA_PATH+'/aclImdb/imdb.vocab')
    for i,word in enumerate(f.readlines()):
        vocab_map.update([(word.replace('\n', ''), i)])  # stray '\n's
        num_map.update([(i, word.replace('\n', ''))])
    f.close()
    
    # case inp_size too large
    if inp_size > max(num_map.keys())+1:
        print('inp_size to large, max is', max(num_map.keys())+1)
        return
    
    if remove_stopwords:
        try:
            stopws = stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
            stopws = stopwords.words('english')
            
        # load stop_nums from nltk stopword list
        for stop_word in stopws:
            try:
                stop_nums.append(vocab_map[stop_word])
            except KeyError:
                pass
   
        # remove stop numbers
        for num in stop_nums:
            del num_map[num]
   
        # new indicies for test, train
        for new_num, old_num in enumerate(sorted(num_map)):
            old_new_map.update([(old_num, new_num)])

        # case inp_size too large after removing stop words
        if inp_size > max(old_new_map.values())+1:
            print('inp_size to large, max is', max(old_new_map.values())+1)
            return
        
    # load test
    f = open(DATA_PATH+'/aclImdb/test/labeledBow.feat')
    for review in f.readlines():
        d = dict()  # dict of a review
        review = review.split(' ')  # 'a:b c:d' -> ['a:b', 'c:d']
        d['score']=int(review[0])   # first position is score
        review = map(lambda x: tuple(x.split(':')), review[1:])  # 'x:y' to ('x','y')
        review = map(lambda x: (int(x[0]), int(x[1])), review)  # int cast from str
        if remove_stopwords:
            review = filter(lambda x: x[0] not in stop_nums, review) # filter out stop 
            review = map(lambda x: (old_new_map[x[0]], x[1]), review) # use new indicies
        review = filter(lambda x: x[0] <  inp_size, review) # removing less common words
        d.update(review)
        test_reviews.append(d)
    f.close()

    # load train
    f = open(DATA_PATH+'/aclImdb/train/labeledBow.feat')
    for review in f.readlines():
        d = dict()  # dict of a review
        review = review.split(' ')  # 'a:b c:d' -> ['a:b', 'c:d']
        d['score']=int(review[0])   # first position is score
        review = map(lambda x: tuple(x.split(':')), review[1:])  # 'x:y' to ('x','y')
        review = map(lambda x: (int(x[0]), int(x[1])), review)  # int cast from str
        if remove_stopwords:
            review = filter(lambda x: x[0] not in stop_nums, review)  # filter out stop
            review = map(lambda x: (old_new_map[x[0]], x[1]), review)  # use new indices
        review = filter(lambda x: x[0] <  inp_size, review) # removing less common words
        d.update(review)
        train_reviews.append(d)
    f.close()   
    
    if remove_stopwords:
        nmap = dict(zip(old_new_map.values(), num_map.values()))
        wmap = dict(zip(num_map.values(), old_new_map.values()))
    else:
        nmap = num_map
        wmap = vocab_map
    train_ds = AclImdb_Dataset(train_reviews, inp_size, preprocessing)
    test_ds = AclImdb_Dataset(test_reviews, inp_size, preprocessing)
    return train_ds, test_ds, nmap, wmap
                        
# to test a custom string on model
def test_review(review, wmap, inp_size, model): 
    review = review.split(' ')
    review = filter(lambda x: x in wmap.keys(), review)
    review = map(lambda x: wmap[x], review)
    review = list(filter(lambda x: x < inp_size, review))
    inp = torch.zeros(1,inp_size)
    for word in review:
        inp[:,word]+=1
    print('word indicies:',review)
    pred = model(inp.cuda()).argmax().item()
    if pred ==1:
        pred = 'good'
    else: 
        pred = 'bad'
    print('prediction:',pred)
    
# maps input vector back to word counts
def review_to_words(inp_vector, nmap):
    for index,count in enumerate(inp_vector):
        if count.item() != 0:
            print('{}:{}'.format(nmap[index],count.item()))
            
