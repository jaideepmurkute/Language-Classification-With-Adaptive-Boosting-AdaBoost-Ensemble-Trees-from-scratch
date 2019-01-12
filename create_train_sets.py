def main():
    with open('train.dat','r') as fp:
           content = fp.readlines()
    good_content = ''
    for word in content:
        if ((word == "") or (word is None) or (word=='\n')):
            continue
        else:
            good_content = good_content + word
    #print(good_content)
    
    splitted_content = good_content.split(" ")
    
    max_words = len(splitted_content)
    words_read = 0
    formatted_sample = []
    num_samples = 0
    
    sentence_length = 50
    start = 0
    end = start + sentence_length
    samples = 0
    while(end+sentence_length < max_words):
        sample = ""
        end = start + sentence_length
        samples = samples + 1
        for i in range(start,end):
            if(i < max_words):
                if(splitted_content[i] == '\n'):
                    print("look at me")
                    break
                sample = sample + (splitted_content[i]+" ")
        start = end
        formatted_sample.append('nl|'+sample+'\n')
    
    print(formatted_sample)
    fp = open("fifty_words_sample.dat",'w')
    for item in formatted_sample:
        fp.write(item)
    print("samples: ", samples)
    
    
    
    
    
    with open('train_eng.dat','r') as fp:
           content = fp.readlines()
    good_content = ''
    for word in content:
        if ((word == "") or (word is None) or (word=='\n')):
            continue
        else:
            good_content = good_content + word
    #print(good_content)
    
    splitted_content = good_content.split(" ")
    
    max_words = len(splitted_content)
    words_read = 0
    formatted_sample = []
    num_samples = 0
    
    sentence_length = 50
    start = 0
    end = start + sentence_length
    samples = 0
    while(end+sentence_length < max_words):
        sample = ""
        end = start + sentence_length
        samples = samples + 1
        for i in range(start,end):
            if(i < max_words):
                sample = sample + (splitted_content[i]+" ")
        start = end
        formatted_sample.append('en|'+sample+'\n')
    
    print(formatted_sample)
    fp = open("fifty_words_sample.dat",'a')
    for item in formatted_sample:
        fp.write(item)
    print("samples: ", samples)
    
    
    
    
    
        
if __name__ == "__main__":
    
    main()