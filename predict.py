import copy
import sys
import math
import pickle

class Node():
    __slots__ = 'parent', 'left', 'right'
    def __init__(self, parent, left=None, right=None):
        self.parent = parent
        self.left = left
        self.right = right

def main(test_file_name):
    content = ''
    with open(test_file_name,'r') as fp:
        content = content + str(fp.readlines())
    #feature_test_map, num_features, num_samples, init_entropy = extract_features(content)
    
    strippped_content = content.strip(" ")
    
    mode='ab'
    
    if mode == 'dt':
        with open('DTree','rb') as saved_model3:
            saved_model = pickle.load(saved_model3)
        num_samples = 1
        feature_test_map,num_features = extract_features(content)
        #print("features extracted")
        prediction = predict_language(feature_test_map, saved_model)
        if prediction == 'nl':
            prediction = 'Dutch'
        elif prediction == 'en':
            prediction == "English"
        print()
        print("Text is in: ",prediction)
    elif mode == 'ab':
        num_classifiers = 15
        '''PUT IN FAVORITE MODEL'S NAME HERE'''
        with open('AdaBoostWeightsDump_fav','rb') as saved_model2:
            saved_model = pickle.load(saved_model2)
        weights_of_classifiers = saved_model[0]
        print("weights_of_weak_classifiers: ", weights_of_classifiers)
        stump_ids = saved_model[1]
        print("stump_ids: ", stump_ids)
        prediction = predict_language_simple_adaboost(content, weights_of_classifiers, stump_ids)
        if prediction == 'nl':
            prediction = 'Dutch'
        elif prediction == 'en':
            prediction == "English"
        print()
        print("Text is in: ",prediction)
            
    
def wt_normalize(weights_of_classifiers):
    sum = 0
    for i in range(0, len(weights_of_classifiers)):
        sum = sum + weights_of_classifiers[i]
    
    for i in range(0, len(weights_of_classifiers)):    
        weights_of_classifiers[i] = weights_of_classifiers[i] / sum
    
    return weights_of_classifiers 
    
def predict_language_simple_adaboost(content, weights_of_classifiers, stump_ids):
    weights_of_classifiers = wt_normalize(weights_of_classifiers)
    test_data = content 
    overall_pred = 0
    for i in range(len(weights_of_classifiers)):
        pred = get_stump_prediction(stump_ids[i], test_data)
        if pred == 'en':
            pred = 1
        else: 
            pred = -1
        overall_pred = overall_pred + pred*weights_of_classifiers[i]
    if overall_pred > 0:
        return 'en'
    else:
        return 'nl'


def get_stump_prediction(stump_id, input_sample):
    if stump_id == 1:
        if get_feature1(stump_id, input_sample) == True:
            #print("stump1 en")
            return 'en'
        #print("stump1 nl")
        return 'nl'
    elif stump_id == 2:
        if get_feature2(stump_id, input_sample) == True:
            #print("stump2 en")
            return 'en'
        #print("stump2 nl")
        return 'nl'
    elif stump_id == 3:
        if get_feature3(stump_id, input_sample) == True:
            #print("stump3 en")
            return 'en'
        #print("stump3 nl")
        return 'nl'
    elif stump_id == 4:
        if get_feature4(stump_id, input_sample) == True:
            #print("stump4 en")
            return 'en'
        #print("stump4 nl")
        return 'nl'
    elif stump_id == 5:
        if get_feature5(stump_id, input_sample) == True:
            #print("stump5 en")
            return 'en'
        #print("stump5 nl")
        return 'nl'
    elif stump_id == 6:
        if get_feature6(stump_id, input_sample) == True:
            #print("stump6 en")
            return 'en'
        #print("stump6 nl")
        return 'nl'
    elif stump_id == 7:
        if get_feature7(stump_id, input_sample) == True:
            #print("stump7 en")
            return 'en'
        #print("stump7 nl")
        return 'nl'
    elif stump_id == 8:
        if get_feature8(stump_id, input_sample) == True:
            #print("stump8 en")
            return 'en'
        #print("stump8 nl")
        return 'nl'
    elif stump_id == 9:
        if get_feature9(stump_id, input_sample) == True:
            #print("stump9 en")
            return 'en'
        #print("stump9 nl")
        return 'nl'
    elif stump_id == 10:
        if get_feature10(stump_id, input_sample) == True:
            #print("stump10 en")
            return 'en'
        #print("stump10 nl")
        return 'nl'

       
    
def predict_language(input_feature_map, trained_tree):
    #print("input_feature_map: ", input_feature_map)
    if(input_feature_map[0][1] == 1):
        input_feature_map = input_feature_map[1:][0]
    #print("after: ", input_feature_map)
    root_node = trained_tree
    print("feature used: ")
    while(True):
        if(root_node.parent == 'nl'):
            return 'nl'
        if(root_node.parent == 'en'):
            return 'en'
        feature = root_node.parent
        print(str(feature)+" ",end="")
        if(input_feature_map[feature] == True):
            root_node = root_node.left
        if(input_feature_map[feature] == False):
            root_node = root_node.right
    print()

def extract_features(sample):
    data = []
    feature_map = []
    #num_samples = len(sample)
    num_samples = 1
    for i in range(0, num_samples):
        feature_map.append(['NA'])
            
    for i in range(0, num_samples):
        feature_map[i].append(get_feature1(i, sample))
        feature_map[i].append(get_feature2(i, sample))
        feature_map[i].append(get_feature3(i, sample))
        feature_map[i].append(get_feature4(i, sample))
        feature_map[i].append(get_feature5(i, sample))    
        feature_map[i].append(get_feature6(i, sample))    
        feature_map[i].append(get_feature7(i, sample))    
        feature_map[i].append(get_feature8(i, sample))    
        feature_map[i].append(get_feature9(i, sample))    
        feature_map[i].append(get_feature10(i, sample))    
    num_features = len(feature_map[0]) - 1
    
    'information of node to come in handy to maintain and use tree'
    signature = ['lang',1,2,3,4,5,6,7,8,9,10]
    feature_map.insert(0, signature)
    
    #for i in range(len(feature_map)):
    #    for j in range(len(feature_map[0])):
    #        print(str(feature_map[i][j])+" ",end="")
    #    print()
    #init_entropy = get_entropy(nl_count, en_count, num_samples)
    return feature_map, num_features


def get_entropy(class_0, class_1, total):
    if class_0 != 0 and class_1 != 0:
        P_0 = class_0/total
        P_1 = class_1/total
        entropy = -1*((P_0*math.log(P_0,2)) + (P_1*math.log(P_1,2)))
    else:
        entropy = 0
    return entropy
'''Each stump returns true if english, false otherwise''' 

#if English articles are present
def get_feature1(i, curr_data):
    #ind_words = curr_data.split('|')[1].split()
    ind_words = curr_data
    for single_word in ind_words.lower().split(" "):
        single_word = single_word.lower().replace(',','')
        if single_word == 'a' or single_word =='an' or single_word =='the':
            return True
    return False

'''if Dutch article is present'''
dutch_articles = ['een', 'de', 'het', 'groene', 'groen', 'hij', 'zij', 'haar', 'hem', 'zijn', 'dit', 'deze', 'die', 'dat', 'wie', 'wat']
def get_feature2(i, curr_data):
    #ind_words = curr_data.split('|')[1].split()
    ind_words = curr_data
    for single_word in ind_words.split(" "):
        single_word = single_word.lower().replace(',','')
        if single_word in dutch_articles:
            return False
    return True

'''if word and is present'''
'''
def get_feature3(i, curr_data):
    #ind_words = curr_data.split('|')[1].split()
    ind_words = curr_data
    for single_word in ind_words:
        single_word = single_word.lower().replace(',','')
        if single_word == 'and':
            return True
    return False
'''
def get_feature3(i, curr_data):
    #ind_words = curr_data.split('|')[1].split()
    ind_words = curr_data
    tot_len = 0
    for single_word in ind_words.split(" "):
        single_word = single_word.lower().replace(',','')
        tot_len = tot_len + len(single_word)
    avg_word_len = tot_len / len(curr_data)
    if(avg_word_len > 9):
        return False
    return True


'''if common dutch words, that are not in english, are present'''
common_dutch = ['niet','wat','ze', 'zijn', 'maar','die', 'heb','voor', 'ben','mijn','dit','hem','hebben','heeft','nu',
                'hoe','kom','gaan','bent','haar','doen','ook', 
                'daar','al','ons','gaat','hebt','waarom','deze','laat','moeten','wie','alles',    
                'kunnen','nooit','komt','misschien','iemand','veel','worden','onze','leven','weer',    
                'nodig','twee','tegen','maken','wordt','mag','altijd','wacht','geef','dag','zeker',    
                'allemaal','gedaan','huis','zij','jaar','vader','doet','vrouw','geld','hun','anders',    
                'zitten','niemand','binnen','spijt','maak','staat','werk','moeder','gezien','waren','wilde',    
                'praten','genoeg','meneer','klaar','ziet','elkaar','uur','zegt','helpen','helemaal',    
                'graag','krijgen','werd','zonder','naam','vriend','beetje','jongen','snel','geven','achter',
                'wanneer','kinderen','onder']

def get_feature4(i, curr_data):
    #ind_words = curr_data.split('|')[1].split()
    ind_words = curr_data
    for single_word in ind_words.split(" "):
        single_word = single_word.lower().replace(',','')
        if single_word in common_dutch:
            return False
    return True

'''check other filler Dutch words'''
def get_feature5(i, curr_data):
    #ind_words = curr_data.split('|')[1].split()
    ind_words = curr_data
    for single_word in ind_words.split(" "):
        single_word = single_word.lower().replace(',','')
        if (single_word == 'omdat' or single_word == 'van'):
            return False
    return True

'''check Dutch substrings which are rare in english'''
def get_feature6(i, curr_data):
    #ind_words = curr_data.split('|')[1].split()
    ind_words = curr_data
    for single_word in ind_words.split(" "):
        single_word = single_word.lower().replace(',','')
        if (('sch' in single_word) or ('tsj' in single_word)):
            return False
    return True

'''check substrings that occur with higher frequency in dutch'''
def get_feature7(i, curr_data):
    #ind_words = curr_data.split('|')[1].split()
    ind_words = curr_data
    frequency = 0
    for single_word in ind_words.split(" "):
        single_word = single_word.lower().replace(',','')
        if (('zi' in single_word) or ('ji' in single_word) or ('iz' in single_word)):
            frequency = frequency + 1
    if frequency > 3:
        return False
    return True

'''check other filler English words'''
'''
def get_feature8(i, curr_data):
    #ind_words = curr_data.split('|')[1].split()
    ind_words = curr_data
    for single_word in ind_words:
        single_word = single_word.lower().replace(',','')
        if (single_word == 'the'):
            return True
    return False
'''
def get_feature8(i, curr_data):
    #ind_words = curr_data.split('|')[1].split()
    ind_words = curr_data
    long_count = 0
    for single_word in ind_words.split(" "):
        single_word = single_word.lower().replace(',','')
        if (len(single_word) > 8):
            long_count = long_count + 1
        if(long_count > 5):
            return False
    return True

'''words end rarely with this in english'''
def get_feature9(i, curr_data):
    #ind_words = curr_data.split('|')[1].split()
    ind_words = curr_data
    for single_word in ind_words.split(" "):
        single_word = single_word.lower().replace(',','')
        if (single_word.endswith('r') or single_word.endswith('i') or single_word.endswith('n') or single_word.endswith('e')):
            return False
    return True

'''dutch has multiple common words with occurrences of 'oo' '''
def get_feature10(i, curr_data):
    #ind_words = curr_data.split('|')[1].split()
    ind_words = curr_data
    count = 0
    for single_word in ind_words.split(" "):
        single_word = single_word.lower().replace(',','')
        if ('oo' in single_word):
            count = count + 1
    if count > 2:
        return False
    return True


'''Usage:  python  predict.py   mode=ab|dt   test_file_name.txt '''
'''test_file should have plain text. If test file is in format same as given sample train files, then uncomment currently commented
   first line in each get_feature functions'''
if __name__ == "__main__":
    #mode = sys.argv[2]
    test_file_name = sys.argv[1]
    #sample_length = sys.argv[3]
    #main(mode, test_file_name)
    main(test_file_name)
    