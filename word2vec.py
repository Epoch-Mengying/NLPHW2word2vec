#### Mengying Zhang

import os,sys,re,csv
import pickle
from collections import Counter, defaultdict
import numpy as np
import scipy
import math
import random
import nltk
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize # added
from numba import jit
import matplotlib.pyplot as plt



#... (1) First load in the data source and tokenize into one-hot vectors.
#... Since one-hot vectors are 0 everywhere except for one index, we only need to know that index.


#... (2) Prepare a negative sampling distribution table to draw negative samples from.
#... Consistent with the original word2vec paper, this distribution should be exponentiated.


#... (3) Run a training function for a number of epochs to learn the weights of the hidden layer.
#... This training will occur through backpropagation from the context words down to the source word.


#... (4) Re-train the algorithm using different context windows. See what effect this has on your results.


#... (5) Test your model. Compare cosine similarities between learned word vectors.










#.................................................................................
#... global variables
#.................................................................................


random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10


vocab_size = 0
hidden_size = 100
uniqueWords = [""]                      #... list of all unique tokens
wordcodes = {}                          #... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()                  #... how many times each token occurs
samplingTable = []                      #... table to draw negative samples from






#.................................................................................
#... load in the data and convert tokens to one-hot indices
#.................................................................................



def loadData(filename):
    global uniqueWords, wordcodes, wordcounts
    override = True
    if override:
        #... for debugging purposes, reloading input file and tokenizing is quite slow
        #...  >> simply reload the completed objects. Instantaneous.
        fullrec = pickle.load(open("w2v_fullrec.p","rb"))
        wordcodes = pickle.load( open("w2v_wordcodes.p","rb"))
        uniqueWords= pickle.load(open("w2v_uniqueWords.p","rb"))
        wordcounts = pickle.load(open("w2v_wordcounts.p","rb"))
        return fullrec


    #... load in the unlabeled data file. You can load in a subset for debugging purposes.
    handle = open(filename, "r", encoding="utf8")
    fullconts =handle.read().split("\n")
    fullconts = [entry.split("\t")[1].replace("<br />", "") for entry in fullconts[1:(len(fullconts)-1)]]

    #... apply simple tokenization (whitespace and lowercase)
    [fullconts] = [" ".join(fullconts).lower()]



    print ("Generating token stream...")
    #... (TASK) populate fullrec as one-dimension array of all tokens in the order they appear.
    #... ignore stopwords in this process
    #... for simplicity, you may use nltk.word_tokenize() to split fullconts.
    #... keep track of the frequency counts of tokens in origcounts.
    fullrec = np.array(word_tokenize(fullconts))
    min_count = 50
    origcounts = Counter(fullrec)

    # stopwords removed
    stopWords = set(stopwords.words('english'))
    fullrec_nostopwords = [w for w in fullrec if not w in stopWords]


    print ("Performing minimum thresholding..")
    #... (TASK) populate array fullrec_filtered to include terms as-is that appeared at least min_count times
    #... replace other terms with <UNK> token.
    #... update frequency count of each token in dict wordcounts where: wordcounts[token] = freq(token)
    fullrec_filtered = ['<UNK>' if origcounts[w] <= min_count else w for w in fullrec_nostopwords]
    wordcounts = dict(Counter(fullrec_filtered))

    #... after filling in fullrec_filtered, replace the original fullrec with this one.
    fullrec = fullrec_filtered





    print ("Producing one-hot indicies")
    #... (TASK) sort the unique tokens into array uniqueWords
    #... produce their one-hot indices in dict wordcodes where wordcodes[token] = onehot_index(token)
    #... replace all word tokens in fullrec with their corresponding one-hot indices.
    uniqueWords = []
    n = 0
    for key in sorted(wordcounts.keys()):
        uniqueWords.append(key)
        wordcodes[key] = n
        n += 1

    for index in range(len(fullrec)):
        word = fullrec[index]
        fullrec[index] = int(wordcodes[word])


     


    #... close input file handle
    handle.close()



    #... store these objects for later.
    #... for debugging, don't keep re-tokenizing same data in same way.
    #... just reload the already-processed input data with pickles.
    #... NOTE: you have to reload data from scratch if you change the min_count, tokenization or number of input rows
    
    pickle.dump(fullrec, open("w2v_fullrec.p","wb+"))
    pickle.dump(wordcodes, open("w2v_wordcodes.p","wb+"))
    pickle.dump(uniqueWords, open("w2v_uniqueWords.p","wb+"))
    pickle.dump(dict(wordcounts), open("w2v_wordcounts.p","wb+"))


    #... output fullrec should be sequence of tokens, each represented as their one-hot index from wordcodes.
    return fullrec







#.................................................................................
#... compute sigmoid value
#.................................................................................
@jit(nopython=True)
def sigmoid(x):
    return float(1)/(1+np.exp(-x))









#.................................................................................
#... generate a table of cumulative distribution of words
#.................................................................................


def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power=0.75):
    # Added
    override = True
    if override:
        cumulative_dict = pickle.load(open("w2v_negativeSampleTable.p","rb"))
        print ("negativeSampleTable loaded")
        return cumulative_dict
                  

    #global wordcounts
    #... stores the normalizing denominator (count of all tokens, each count raised to exp_power)
    max_exp_count = 0


    print ("Generating exponentiated count vectors")
    #... (TASK) for each uniqueWord, compute the frequency of that word to the power of exp_power
    #... store results in exp_count_array.
    exp_count_array = np.array([wordcounts[word] ** exp_power for word in uniqueWords]) # stores each count in an array
    max_exp_count = sum(exp_count_array)



    print ("Generating distribution")

    #... (TASK) compute the normalized probabilities of each term.
    #... using exp_count_array, normalize each value by the total value max_exp_count so that
    #... they all add up to 1. Store this corresponding array in prob_dist
    prob_dist = exp_count_array/max_exp_count # a list





    print ("Filling up sampling table")
    #... (TASK) create a dict of size table_size where each key is a sequential number and its value is a one-hot index
    #... the number of sequential keys containing the same one-hot index should be proportional to its prob_dist value
    #... multiplied by table_size. This table should be stored in cumulative_dict.
    #... we do this for much faster lookup later on when sampling from this table.

    cumulative_dict = {}
    table_size = 1e8
    c = 0
    for i in range(len(uniqueWords)):
        number_of_keys = int(prob_dist[i] * table_size)
        for j in range(number_of_keys):
            cumulative_dict[c] = i
            c += 1

    # Added: save the negative sampling table
    pickle.dump(cumulative_dict, open("w2v_negativeSampleTable.p","wb+"))

    return cumulative_dict






#.................................................................................
#... generate a specific number of negative samples
#.................................................................................

def generateSamples(context_idx, num_samples):
    global samplingTable, uniqueWords, randcounter
    results = [context_idx]
    #... (TASK) randomly sample num_samples token indices from samplingTable.
    #... don't allow the chosen token to be context_idx.
    #... append the chosen indices to results
    

    len_table = len(samplingTable)
    while context_idx in results:
        results = []
        #sample =  random.sample(list(samplingTable),1)[0] # a dict key
        for i in range(num_samples):
            sample =  np.random.randint(len_table,size=1)[0]     ## way way faster than random.sample
            results.append(samplingTable[sample]) # want the corresponding dict value
    
    return results









@jit(nopython=True)
def performDescent(num_samples, learning_rate, center_token, sequence_chars,W1,W2,negative_indices):
    # sequence chars was generated from the mapped sequence in the core code
    nll_new = 0
    d = W1.shape[1] # dimension for embedding
    

    for k in range(0, len(sequence_chars)):

        #... (TASK) implement gradient descent. Find the current context token from sequence_chars
        #... and the associated negative samples from negative_indices. Run gradient descent on both
        #... weight matrices W1 and W2.
        #... compute the total negative log-likelihood and store this in nll_new.
        
        ############### for context word k ###############
    ### Temp 
        v = int(sequence_chars[k]) # current context token's index
        h = W1[v,]
        summation = 0
        ng_logsummation = 0 # for later use in calculating nll

    
        v_j_prime = W2[v, ]
        sigmoid_err = sigmoid(np.dot(v_j_prime,h)) - 1
        summation += sigmoid_err
        v_j_new =  v_j_prime - learning_rate * sigmoid_err * h   # for context word, a (d, 1) column vector
        


        ############### for negative samples ###############
    # summation, for later use in updating W1
        sigmoid_err_list = []
        for i in range(num_samples):
            ng_index = negative_indices[num_samples*k + i] # context token's corresponding negative sample i
            v_j_ng = W2[ng_index,] # current embedding for v this negative sample i: a (d, 1) column vector
            dot_product = np.dot(v_j_ng, h)
            sigmoid_err = sigmoid(dot_product)
            summation += sigmoid_err # for later use in updating W1 matrix
            sigmoid_err_list.append(sigmoid_err)
            
    
        for i in range(num_samples):
    # update W1 for negative sample
            v_j_ng = W2[ng_index,] # current embedding for v this negative sample i: a (d, 1) column vector
            v_I_ng = W1[ng_index, ] # current embedding for this negative sample i in W1: a (d, 1) column vector
            v_I_ngnew = (v_I_ng - learning_rate * summation * v_j_ng) # a(d, ) column vector

    # update W2 for negative sample
            v_j_ngnew = (v_j_ng - learning_rate *  sigmoid_err_list[i] * h)  # a (d, ) column vector
            
            W2[ng_index,] = v_j_ngnew # update the W2
            W1[ng_index,] = v_I_ngnew # update the W1



        ############## for context word k ###################
    ### update W1 
        v_Inew = h - learning_rate * summation * v_j_prime # a (d, 1) column vector
        W1[v,] = v_Inew # update the W1
    ### update W2
        W2[v,] = v_j_new # update the W2
         

        ############### for negative log likelihoods ###############
        for i in range(num_samples):
            ng_index = negative_indices[num_samples*k + i] # context token's corresponding negative sample i
            ng_logsummation += math.log(sigmoid(-np.dot(W2[ng_index,],v_Inew)))

        nll_new += -math.log(sigmoid(np.dot(v_j_new,v_Inew))) - ng_logsummation 

    return [nll_new]
##======================================================================================
    

  






#.................................................................................
#... learn the weights for the input-hidden and hidden-output matrices
#.................................................................................


def trainer(curW1 = None, curW2=None):
    global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size,np_randcounter, randcounter
    vocab_size = len(uniqueWords)           #... unique characters
    hidden_size = 100                       #... number of hidden neurons
    context_window = [-2,-1,1,2]            #... specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
    nll_results = []                        #... keep array of negative log-likelihood after every 1000 iterations


    #... determine how much of the full sequence we can use while still accommodating the context window
    start_point = int(math.fabs(min(context_window))) # where the center token index in mapped_sequence starts
    end_point = len(fullsequence)-(max(max(context_window),0)) # where the center token index will end(exclude)
    mapped_sequence = fullsequence




    #... initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output.
    if curW1==None:
        np_randcounter += 1
        W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
        W2 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
    else:
        #... initialized from pre-loaded file
        W1 = curW1
        W2 = curW2



    #... set the training parameters
    epochs = 5
    num_samples = 2
    learning_rate = 0.05
    nll = 0
    iternum = 0
    iternum_list = []



    print (end_point-start_point)

    #... Begin actual training
    for j in range(0,epochs):
        print ("Epoch: ", j)
        prevmark = 0

        
        #... For each epoch, redo the whole sequence...
        for i in range(start_point,end_point):


            if (float(i)/len(mapped_sequence))>=(prevmark+0.1):
                print ("Progress: ", round(prevmark+0.1,1))
                prevmark += 0.1

            if iternum%10000==0:
                print ("Negative likelihood: ", nll)                
                nll_results.append(nll)
                iternum_list.append(iternum) # Added
                nll = 0
                
                


            #... (TASK) determine which token is our current input. Remember that we're looping through mapped_sequence
            center_token = mapped_sequence[i] # it's the word token index
            #... (TASK) don't allow the center_token to be <UNK>. move to next iteration if you found <UNK>.
            if center_token == '<UNK>':
                continue


            center_token = int(center_token)
            iternum += 1
            #... now propagate to each of the context outputs
            mapped_context = [mapped_sequence[i+ctx] for ctx in context_window] # context window's corresponding sequence word token index in mapped_sequence
            negative_indices = []
         
            #print ("Generate 8 negative samples: ")
            
            for q in mapped_context:
                negative_indices += generateSamples(q, num_samples) # you have 8 elements in our case
            #print ("BAng! Iter:", i," finished!")

            #... implement gradient descent
            [nll_new] = performDescent(num_samples, learning_rate, center_token, mapped_context, W1,W2, negative_indices)
            nll += nll_new # update the nll so that we can print in the beginning of the next loop

        break




    for nll_res in nll_results:
        print (nll_res)

    # plotting here
    print ("======== negative log likelihood plot ====")
    plt.plot(iternum_list, nll_results)
    plt.ylabel('negative log likelihoods')
    plt.xlabel('iternums')
    plt.show()

    return [W1,W2]



#.................................................................................
#... Load in a previously-saved model. Loaded model's hidden and vocab size must match current model.
#.................................................................................

def load_model():
    handle = open("saved_W1.data(4)","rb")
    W1 = np.load(handle)
    handle.close()
    handle = open("saved_W2.data(4)","rb")
    W2 = np.load(handle)
    handle.close()
    return [W1,W2]






#.................................................................................
#... Save the current results to an output file. Useful when computation is taking a long time.
#.................................................................................

def save_model(W1,W2):
    handle = open("saved_W1.data","wb+")
    np.save(handle, W1, allow_pickle=False)
    handle.close()

    handle = open("saved_W2.data","wb+")
    np.save(handle, W2, allow_pickle=False)
    handle.close()






#... so in the word2vec network, there are actually TWO weight matrices that we are keeping track of. One of them represents the embedding
#... of a one-hot vector to a hidden layer lower-dimensional embedding. The second represents the reversal: the weights that help an embedded
#... vector predict similarity to a context word.






#.................................................................................
#... code to start up the training function.
#.................................................................................
word_embeddings = []
proj_embeddings = []
def train_vectors(preload=False):
    global word_embeddings, proj_embeddings
    if preload:
        [curW1, curW2] = load_model()
    else:
        curW1 = None
        curW2 = None
    [word_embeddings, proj_embeddings] = trainer(curW1,curW2)
    save_model(word_embeddings, proj_embeddings)









#.................................................................................
#... for the averaged morphological vector combo, estimate the new form of the target word
#.................................................................................

def morphology(word_seq):
    global word_embeddings, proj_embeddings, uniqueWords, wordcodes
    embeddings = word_embeddings
    vectors = [word_seq[0], # suffix averaged
    embeddings[wordcodes[word_seq[1]]]]
    vector_math = vectors[0]+vectors[1]
    #... find whichever vector is closest to vector_math
    #... (TASK) Use the same approach you used in function prediction() to construct a list
    #... of top 10 most similar words to vector_math. Return this list.







#.................................................................................
#... for the triplet (A,B,C) find D such that the analogy A is to B as C is to D is most likely
#.................................................................................

def analogy(word_seq):
    global word_embeddings, proj_embeddings, uniqueWords, wordcodes
    embeddings = word_embeddings
    vectors = [embeddings[wordcodes[word_seq[0]]],
    embeddings[wordcodes[word_seq[1]]],
    embeddings[wordcodes[word_seq[2]]]]
    vector_math = -vectors[0] + vectors[1] - vectors[2] # + vectors[3] = 0
    #... find whichever vector is closest to vector_math
    #... (TASK) Use the same approach you used in function prediction() to construct a list
    #... of top 10 most similar words to vector_math. Return this list.







#.................................................................................
#... find top 10 most similar words to a target word
#.................................................................................


def prediction(target_word):
    global word_embeddings, uniqueWords, wordcodes
    targets = [target_word]
    outputs = []
    #... (TASK) search through all uniqueWords and for each token, compute its similarity to target_word.
    #... you will compute this using the absolute cosine similarity of the word_embeddings for the word pairs.
    #... Note that the cosine() function from scipy.spatial.distance computes a DISTANCE so you need to convert that to a similarity.
    #... return a list of top 10 most similar words in the form of dicts,
    #... each dict having format: {"word":<token_name>, "score":<cosine_similarity>}











if __name__ == '__main__':
    if len(sys.argv)==2:
        filename = sys.argv[1]
        #... load in the file, tokenize it and assign each token an index.
        #... the full sequence of characters is encoded in terms of their one-hot positions

        fullsequence= loadData(filename)
        print ("Full sequence loaded...")
        #print(uniqueWords)
        #print (len(uniqueWords))



        #... now generate the negative sampling table
        print ("Total unique words: ", len(uniqueWords))
        print("Preparing negative sampling table")
        samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)


        #... we've got the word indices and the sampling table. Begin the training.
        #... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
        #... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
        #... ... and uncomment the load_model() line

        train_vectors(preload=False)
        [word_embeddings, proj_embeddings] = load_model()




        #... we've got the trained weight matrices. Now we can do some predictions
        targets = ["good", "bad", "scary", "funny"]
        for targ in targets:
            print("Target: ", targ)
            bestpreds= (prediction(targ))
            for pred in bestpreds:
                print (pred["word"],":",pred["score"])
            print ("\n")



        #... try an analogy task. The array should have three entries, A,B,C of the format: A is to B as C is to ?
        print (analogy(["son", "daughter", "man"]))
        print (analogy(["thousand", "thousands", "hundred"]))
        print (analogy(["amusing", "fun", "scary"]))
        print (analogy(["terrible", "bad", "amazing"]))



        #... try morphological task. Input is averages of vector combinations that use some morphological change.
        #... see how well it predicts the expected target word when using word_embeddings vs proj_embeddings in
        #... the morphology() function.

        s_suffix = [word_embeddings[wordcodes["stars"]] - word_embeddings[wordcodes["star"]]]
        others = [["types", "type"],
        ["ships", "ship"],
        ["values", "value"],
        ["walls", "wall"],
        ["spoilers", "spoiler"]]
        for rec in others:
            s_suffix.append(word_embeddings[wordcodes[rec[0]]] - word_embeddings[wordcodes[rec[1]]])
        s_suffix = np.mean(s_suffix, axis=0)
        print (morphology([s_suffix, "techniques"]))
        print (morphology([s_suffix, "sons"]))
        print (morphology([s_suffix, "secrets"]))






    else:
        print ("Please provide a valid input filename")
        sys.exit()


