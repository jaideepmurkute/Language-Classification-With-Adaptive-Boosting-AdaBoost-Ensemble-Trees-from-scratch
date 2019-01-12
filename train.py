import sys
import math
import copy
import random
import pickle
import numpy as np

class Node():
    __slots__ = 'parent', 'left', 'right'
    def __init__(self, parent, left=None, right=None):
        self.parent = parent
        self.left = left
        self.right = right

def main(mode, train_file_name):
        count = 0
        #with open('fifty_words_sample.dat','r') as fp:
        with open(train_file_name,'r') as fp:
           count = count + 1
           content = fp.readlines()
        feature_map, num_features, num_samples, init_entropy = extract_features(content)
        
        '''for visual aid and identify columns'''
        signature = ['class',1,2,3,4,5,6,7,8,9,10]
        
        feature_map.insert(0, signature)
        backup_feature_map = copy.deepcopy(feature_map)
        
        if mode == 'dt':
            trained_tree = build_dtree(feature_map, feature_map)
            with open("DTree",'wb') as dump_file1:
                pickle.dump(trained_tree, dump_file1)
            print("Decision Tree trained and saved.")
            #my_predictions(trained_tree)
            
        elif mode == 'ab':
            weights_of_classifiers, stump_ids = new_simple_adaptive_boosting(backup_feature_map, 5, content)
            
            #predict_language_simple_adaboost(weights_of_classifiers, stump_ids)
            
            #with open("AdaBoost",'wb') as dump_file2:
            #    pickle.dump(weak_classifiers_list, dump_file2)
            with open("AdaBoostWeightsDump",'wb') as dump_file3:
                pickle.dump([weights_of_classifiers,stump_ids], dump_file3)
            print("AdaBoost Tree(s) trained and saved.")
        else:
            print("Incorrect mode entered. Choose between 'dt'(Decision Tree) and 'ab'(AdaBoost).")

def my_predictions(trained_tree):
    test_file = 'twenty_words_test.dat'
    with open(test_file,'r') as fp:
           content = fp.readlines()
    feature_map, num_features, num_samples, init_entropy = extract_features(content)
    correct = 0
    for i in range(1, len(feature_map)):
        prediction = predict_language(feature_map[i], trained_tree)
        if prediction == 'nl' and feature_map[i][0] == 'nl':
            correct = correct + 1
        elif prediction == 'en' and feature_map[i][0] == 'en':
            correct = correct + 1
    print()
    print("Accuracy: ",correct/(len(feature_map)-1))  
    
def wt_normalize(weights_of_classifiers):
    sum = 0
    for i in range(0, len(weights_of_classifiers)):
        sum = sum + weights_of_classifiers[i]
    
    for i in range(0, len(weights_of_classifiers)):    
        weights_of_classifiers[i] = weights_of_classifiers[i] / sum
    
    return weights_of_classifiers 
    
def predict_language_simple_adaboost(weights_of_classifiers, stump_ids):
    weights_of_classifiers = wt_normalize(weights_of_classifiers)
    vote_en= 0
    vote_nl = 0
    with open('fifty_words_test.dat','r') as fp:
           content = fp.readlines()
    for i, sentence in enumerate(content):
        content[i] = sentence.strip('\n')
        #print("content[i]:  ", content[i])
    #print("***************************Splitted: ", splitted)      
    test_data = ['nl | Mosasaurus, ook wel maashagedis genoemd, is een geslacht uit de ',
                 'nl|Limburg. Waarschijnlijk zullen ze niet om de afrastering heenlopen. In',
                 'en | Hello there, how are you people doing?',
                 'nl|het noorden zouden ze dan in het stedelijke gebied van',
                'en|the past as it is described in written documents. Events',
                'en|occurring before written record are considered prehistory. It is an',
                'en|umbrella term that relates to past events as well as',
                'en|the memory, discovery, collection, organization, presentation, and interpretation of information'
                 ]
    test_data = content 
    correct = 0
    for j in range(0, len(test_data)):
        overall_pred = 0
        for i in range(len(weights_of_classifiers)):
            pred = get_stump_prediction(stump_ids[i], test_data[j])
            if pred == 'en':
                pred = 1
            else: 
                pred = -1
            print("pred*weights_of_classifiers[i]: ", pred*weights_of_classifiers[i])
            overall_pred = overall_pred + pred*weights_of_classifiers[i]
        if overall_pred >= 0:
            print('en')
            if(test_data[j].split('|')[0]=='en'):
                correct = correct + 1
        else:
            print('nl')
            if(test_data[j].split('|')[0]=='nl'):
                correct = correct + 1
    print("Accuracy: ",correct/len(test_data))

def predict_language(input_feature_map, trained_tree):
    root_node = trained_tree
    while(True):
        if(root_node.parent == 'nl'):
            return 'nl'
        if(root_node.parent == 'en'):
            return 'en'
        feature = root_node.parent
        if(input_feature_map[feature] == True):
            root_node = root_node.left
        if(input_feature_map[feature] == False):
            root_node = root_node.right


def build_dtree(feature_map, parent):
    #print("inside build tree")
    '''No examples left'''
    if len(feature_map) == 1:
        nl_count = 0
        en_count = 0
        for data in parent:
            if data[0] == 'nl':
                nl_count = nl_count + 1
            if data[0] == 'en':
                en_count = en_count + 1
        if en_count >= nl_count:
            return Node("en")
        if nl_count > en_count:
            return Node("nl")
    elif len(feature_map[0]) == 2:
        '''single feature left'''
        nl_count = 0
        en_count = 0
        for data in feature_map:
            if data[0] == 'nl':
                nl_count = nl_count + 1
            if data[0] == 'en':
                en_count = en_count + 1
        if en_count >= nl_count:
            return Node("en")
        if nl_count > en_count:
            return Node("nl")
    else:
        '''make copy so that we can make changes without loosing data'''
        copied_feature_map = copy.deepcopy(feature_map)
        extra_backup = copy.deepcopy(feature_map)
        class_0, class_1, total = get_distribution(copied_feature_map)
        #print("Passed this: ")
        #print(copied_feature_map[0])
        best_split_index = get_best_split_node(copied_feature_map, get_entropy(class_0, class_1, total))
        #print("Got back this: ")
        #print(copied_feature_map[0])
        signature = copied_feature_map[0]
        signature.pop(best_split_index)
        
        left_child_content = []
        right_child_content = []
        
        left_child_content.insert(0, signature)
        right_child_content.insert(0, signature)
        
        for i in range(1, len(copied_feature_map)):
            if(copied_feature_map[i][best_split_index] == True):
                copied_feature_map[i].pop(best_split_index)
                left_child_content.append(copied_feature_map[i])
            elif(copied_feature_map[i][best_split_index] == False):
                copied_feature_map[i].pop(best_split_index)
                right_child_content.append(copied_feature_map[i])
        
        current_node = Node(extra_backup[0][best_split_index])
        '''extend left branch'''
        left_child = build_dtree(left_child_content, parent)
        
        '''extend right branch'''
        right_child = build_dtree(right_child_content, parent)
        
        current_node.left = left_child
        current_node.right = right_child
        
        '''for non-base-case recursive calls'''
        return current_node
   
             
def get_distribution(copied_feature_map):
    class_0 = 0
    class_1 = 0
    for i in range(len(copied_feature_map)):
        if copied_feature_map[i][0] == 'nl':
            class_0 = class_0 + 1
        elif copied_feature_map[i][0] == 'en':
            class_1 = class_1 + 1
    tot = class_0 + class_1
    return class_0, class_1, tot

def extract_features(content):
    data = []
    feature_map = []
    num_samples = 0
    for item in content:
        stripped_line = item.strip()
        if not (stripped_line == "" or stripped_line is None):
            data.append(stripped_line)
            num_samples = num_samples + 1
    
    nl_count = 0
    en_count = 0
    for i in range(0, num_samples):
        if(data[i].split('|')[0] == 'nl'):
            feature_map.append(['nl'])
            nl_count = nl_count + 1
        elif(data[i].split('|')[0] == 'en'):
            feature_map.append(['en'])
            en_count = en_count + 1
        else:
            continue
    
    
    for i in range(0, num_samples):
        feature_map[i].append(get_feature1(i, data[i]))
        feature_map[i].append(get_feature2(i, data[i]))
        feature_map[i].append(get_feature3(i, data[i]))
        feature_map[i].append(get_feature4(i, data[i]))
        feature_map[i].append(get_feature5(i, data[i]))    
        feature_map[i].append(get_feature6(i, data[i]))    
        feature_map[i].append(get_feature7(i, data[i]))
        feature_map[i].append(get_feature8(i, data[i]))
        feature_map[i].append(get_feature9(i, data[i]))
        feature_map[i].append(get_feature10(i, data[i]))    
    num_features = len(feature_map[0]) - 1
    
    'information of node to come in handy to maintain and use tree'
    signature = ['class',1,2,3,4,5,6,7,8,9,10]
    feature_map.insert(0, signature)
    init_entropy = get_entropy(nl_count, en_count, num_samples)
    
    return feature_map, num_features, num_samples, init_entropy


'''Each stump returns true if english, false otherwise''' 

#if English articles are present
def get_feature1(i, curr_data):
    ind_words = curr_data.split('|')[1].split()
    for single_word in ind_words:
        single_word = single_word.lower().replace(',','')
        if single_word == 'a' or single_word =='an' or single_word =='the':
            #print("single_word: ", single_word)
            return True
    return False

'''if Dutch article is present'''
dutch_articles = ['een', 'de', 'het', 'groene', 'groen', 'hij', 'zij', 'haar', 'hem', 'zijn', 'dit', 'deze', 'die', 'dat', 'wie', 'wat']
def get_feature2(i, curr_data):
    ind_words = curr_data.split('|')[1].split()
    for single_word in ind_words:
        single_word = single_word.lower().replace(',','')
        if single_word in dutch_articles:
            return False
    return True

'''if word and is present'''
'''
def get_feature3(i, curr_data):
    ind_words = curr_data.split('|')[1].split()
    for single_word in ind_words:
        single_word = single_word.lower().replace(',','')
        if single_word == 'and':
            return True
    return False
'''
def get_feature3(i, curr_data):
    ind_words = curr_data.split('|')[1].split()
    tot_len = 0
    for single_word in ind_words:
        single_word = single_word.lower().replace(',','')
        tot_len = tot_len + len(single_word)
    avg_word_len = tot_len / len(curr_data)
    if(avg_word_len > 9):
        return False
    return True


'''if common dutch words, that are not in english, are present'''
common_dutch = ['niet','wat','ze', 'zijn', 'maar', 'die', 'heb','voor', 'ben','mijn','dit','hem','hebben','heeft','nu',
                'hoe', 'kom',    'gaan',    'bent',    'haar',    'doen',    'ook', 
                'daar',    'al',    'ons',    'gaat',    'hebt',    'waarom',    'deze',    'laat', 'moeten',    'wie',    'alles',    
                'kunnen',    'nooit',    'komt',    'misschien',    'iemand',    'veel',    'worden',    'onze',    'leven',    'weer',    
                'nodig',    'twee',    'tegen',    'maken', 'wordt',    'mag',    'altijd',    'wacht',    'geef',    'dag',    'zeker',    
                'allemaal',    'gedaan',    'huis',    'zij',    'jaar',    'vader',    'doet',    'vrouw',    'geld',    'hun',    'anders',    
                'zitten',    'niemand',    'binnen','spijt',    'maak',    'staat',    'werk',    'moeder',    'gezien',    'waren',    'wilde',    
                'praten',    'genoeg',    'meneer',    'klaar',    'ziet',    'elkaar',    'uur',    'zegt',    'helpen',    'helemaal',    
                'graag',    'krijgen',    'werd',    'zonder',    'naam',    'vriend',    'beetje',    'jongen',    'snel',    'geven',    'achter',
                'wanneer',    'kinderen',    'onder']
def get_feature4(i, curr_data):
    ind_words = curr_data.split('|')[1].split()
    for single_word in ind_words:
        single_word = single_word.lower().replace(',','')
        if single_word in common_dutch:
            return False
    return True

'''check other filler dutch words'''
def get_feature5(i, curr_data):
    ind_words = curr_data.split('|')[1].split()
    for single_word in ind_words:
        single_word = single_word.lower().replace(',','')
        if (single_word == 'omdat' or single_word == 'van'):
            return False
    return True

'''check dutch substrings which are rare in english'''
def get_feature6(i, curr_data):
    ind_words = curr_data.split('|')[1].split()
    for single_word in ind_words:
        single_word = single_word.lower().replace(',','')
        if (('sch' in single_word) or ('tsj' in single_word)):
            return False
    return True

'''check substrings that occur with higher frequency in dutch'''
def get_feature7(i, curr_data):
    ind_words = curr_data.split('|')[1].split()
    frequency = 0
    for single_word in ind_words:
        single_word = single_word.lower().replace(',','')
        if (('zi' in single_word) or ('ji' in single_word) or ('iz' in single_word)):
            frequency = frequency + 1
    if frequency > 3:
        return False
    return True

'''check other filler English words'''
'''
def get_feature8(i, curr_data):
    ind_words = curr_data.split('|')[1].split()
    for single_word in ind_words:
        single_word = single_word.lower().replace(',','')
        if (single_word == 'the'):
            return True
    return False
'''
def get_feature8(i, curr_data):
    ind_words = curr_data.split('|')[1].split()
    long_count = 0
    for single_word in ind_words:
        single_word = single_word.lower().replace(',','')
        if (len(single_word) > 8):
            long_count = long_count + 1
        if(long_count > 5):
            return False
    return True

'''words end rarely with this in english'''
def get_feature9(i, curr_data):
    ind_words = curr_data.split('|')[1].split()
    for single_word in ind_words:
        single_word = single_word.lower().replace(',','')
        if (single_word.endswith('r') or single_word.endswith('i') or single_word.endswith('n') or single_word.endswith('e')):
            return False
    return True

'''dutch has multiple common words with occurrences of 'oo' '''
def get_feature10(i, curr_data):
    ind_words = curr_data.split('|')[1].split()
    count = 0
    for single_word in ind_words:
        single_word = single_word.lower().replace(',','')
        if ('oo' in single_word):
            count = count + 1
    if count > 2:
        return False
    return True


def get_best_split_node(passed_feature_map, curr_node_entropy):
    all_info_gains = []
    '''watch outn for -1 in line below, it was introduced to avoid exception, first col is language'''
    for i in range(1, len(passed_feature_map[0])):
        en_t = 0
        nl_t = 0
        en_f = 0
        nl_f = 0
        for sample in passed_feature_map:
            label = sample[0]
            if(label == 'nl'):
                if(sample[i] == True):
                    nl_t = nl_t + 1
                else:
                    nl_f = nl_f + 1
            if(label == 'en'):
                if(sample[i] == True):
                    en_t = en_t + 1
                else:
                    en_f = en_f + 1
        
        tot_t = nl_t + en_t
        tot_f = nl_f + en_f
        tot = tot_t + tot_f
        true_child_entropy = get_entropy(nl_t, en_t, tot_t) 
        false_child_entropy = get_entropy(nl_f, en_f, tot_f)
        
        true_child_wt_entropy = (tot_t / tot)*true_child_entropy
        false_child_wt_entropy = (tot_f / tot)*false_child_entropy
        
        entropy_after_split = true_child_wt_entropy + false_child_wt_entropy
        
        curr_info_gain = curr_node_entropy - entropy_after_split
        all_info_gains.append(curr_info_gain)
    
    '''+1 to accomodate negative offset caused by label'''
    best_split_col_index = all_info_gains.index(max(all_info_gains)) + 1
    
    return best_split_col_index


def get_entropy(class_0, class_1, total):
    if class_0 != 0 and class_1 != 0:
        P_0 = class_0/total
        P_1 = class_1/total
        entropy = -1*((P_0*math.log(P_0,2)) + (P_1*math.log(P_1,2)))
    else:
        entropy = 0
    return entropy

def normalize_data(sample_weights):
    sum = 0
    for weight in sample_weights:
        sum = sum + weight
    for i in range(0, len(sample_weights)):
        sample_weights[i] = sample_weights[i] / sum
    return sample_weights 
    

def get_stump_prediction(stump_id, input_sample):
    if stump_id == 1:
        if get_feature1(stump_id, input_sample) == True:
            return 'en'
        return 'nl'
    elif stump_id == 2:
        if get_feature2(stump_id, input_sample) == True:
            return 'en'
        return 'nl'
    elif stump_id == 3:
        if get_feature3(stump_id, input_sample) == True:
            return 'en'
        return 'nl'
    elif stump_id == 4:
        if get_feature4(stump_id, input_sample) == True:
            return 'en'
        return 'nl'
    elif stump_id == 5:
        if get_feature5(stump_id, input_sample) == True:
            return 'en'
        return 'nl'
    elif stump_id == 6:
        if get_feature6(stump_id, input_sample) == True:
            return 'en'
        return 'nl'
    elif stump_id == 7:
        if get_feature7(stump_id, input_sample) == True:
            return 'en'
        return 'nl'
    elif stump_id == 8:
        if get_feature8(stump_id, input_sample) == True:
            return 'en'
        return 'nl'
    elif stump_id == 9:
        if get_feature9(stump_id, input_sample) == True:
            return 'en'
        return 'nl'
    elif stump_id == 10:
        if get_feature10(stump_id, input_sample) == True:
            return 'en'
        return 'nl'

def get_best_split_node_boost(passed_feature_map, curr_node_entropy, sample_weights):
    all_info_gains = []
    for i in range(1, len(passed_feature_map[0])):
        en_t = 0
        nl_t = 0
        en_f = 0
        nl_f = 0
        for sample in passed_feature_map:
            label = sample[0]
            if(label == 'nl'):
                if(sample[i] == True):
                    nl_t = nl_t + 1
                else:
                    nl_f = nl_f + 1
            if(label == 'en'):
                if(sample[i] == True):
                    en_t = en_t + 1
                else:
                    en_f = en_f + 1
        
        tot_t = nl_t + en_t
        tot_f = nl_f + en_f
        tot = tot_t + tot_f
        true_child_entropy = get_simple_entropy_boost(passed_feature_map, sample_weights) 
        false_child_entropy = get_simple_entropy_boost(passed_feature_map, sample_weights)
        
        true_child_wt_entropy = (tot_t / tot)*true_child_entropy
        false_child_wt_entropy = (tot_f / tot)*false_child_entropy
        
        entropy_after_split = true_child_wt_entropy + false_child_wt_entropy
        
        curr_info_gain = curr_node_entropy - entropy_after_split
        all_info_gains.append(curr_info_gain)
    
    '''+1 to accomodate negative offset caused by label'''
    best_split_col_index = all_info_gains.index(max(all_info_gains)) + 1
    
    return best_split_col_index

def get_simple_entropy_boost(weighted_examples, sample_weights):
    relevant_weights = []
    total_weights_sum = 0
    good_weighted_examples = weighted_examples[1:]
    P_0_weights_sum = 0
    P_1_weights_sum = 0
    for i in range(len(good_weighted_examples)-1):
        if(good_weighted_examples[i][0] == 'en'):
            P_0_weights_sum = P_0_weights_sum + sample_weights[i]
        elif(good_weighted_examples[i][0] == 'nl'):
            P_1_weights_sum = P_1_weights_sum + sample_weights[i]
    total = P_0_weights_sum + P_1_weights_sum
    entropy = -1*((P_0_weights_sum*math.log(P_0_weights_sum,2)) + (P_1_weights_sum*math.log(P_1_weights_sum,2)))
    
    return entropy

def new_simple_adaptive_boosting(input_feature_map, num_weak_classifiers, content):
    extra_backup = copy.deepcopy(input_feature_map)
    num_samples = len(input_feature_map)-1
    sample_weights = []
    weak_classifiers_list = []
    num_features = len(input_feature_map[0])-1
    error = 0
    count = 0
    data = []
    feature_map = []
    num_samples = 0
    max_epochs = 3
    error_log = []
    min_error = 9999999
    best_weights = []
    error_per_features = [0]*num_features
    
    num_classifiers = 50
    weights_of_classifiers = [1]*num_classifiers
    best_stump_ids = [1]*num_classifiers
    
    '''data pre-processing'''
    for item in content:
        stripped_line = item.strip()
        if not (stripped_line == "" or stripped_line is None):
            data.append(stripped_line)
            num_samples = num_samples + 1
    
    '''assign equal weights to all data samples'''
    for i in range(2,len(input_feature_map)):
        count = count + 1
        sample_weights.append(1/(len(input_feature_map)-2))
    
    
    for k in range(0, max_epochs): 
        for i in range(num_classifiers):
            error = 0
            curr_entropy = get_simple_entropy_boost(input_feature_map, sample_weights)
            best_split_index = get_best_split_node_boost(input_feature_map, curr_entropy, sample_weights)
            #best_stump_ids.append(best_split_index)
            best_stump_ids[i] = best_split_index
            all_predictions = []
         
            for j in range(0,len(input_feature_map)-2):
                curr_pred = get_stump_prediction(best_split_index, data[j])
                if(input_feature_map[j+1][0]==curr_pred):
                    continue
                else:
                    error = error + sample_weights[j]
                all_predictions.append(curr_pred)
            
            correct = 0
            for j in range (len(all_predictions)):
                if all_predictions[j] == input_feature_map[j+1][0]:
                    correct = correct + 1
                    sample_weights[j] = sample_weights[j] * error/(1-error)
            
            '''normalize weights'''
            sum = 0
            for j in range(0, len(input_feature_map)-2):
                sum = sum + sample_weights[j]
            for j in range(0, len(input_feature_map)-2):
                sample_weights[j] = sample_weights[j] / sum
         
            '''update clf weights i.e. voting power'''
            error_log.append(error)
            if error < min_error:
                min_error = error
            
            if error > 0:
                weights_of_classifiers[i] = math.log(((1-error) / error),2)
            else:
                print("CONVERGED!")
                break
    return weights_of_classifiers, best_stump_ids
  
'''Usage: python   train.py    dt|ab   train_file_name'''      
'''train samples should be in format: language_name(nl|en) sentence''' 
if __name__ == "__main__":
    mode = sys.argv[1]
    train_file_name = sys.argv[2]
    main(mode, train_file_name)