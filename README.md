

```python
import nltk
```


```python
TRAINING_FOLDER = 'data/train-mails/'
TESTING_FOLDER = 'data/test-mails/'
```


```python
import os
from collections import Counter

def read_data(folder):
    emails = [os.path.join(folder,f) for f in os.listdir(folder)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:  #Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)
    
    # Xóa stopwords, dấu câu
    list_to_remove = dictionary.keys()
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary
```


```python
train_dict = read_data(TRAINING_FOLDER)
print train_dict[:10]
```

    [('order', 1414), ('address', 1293), ('report', 1216), ('mail', 1127), ('send', 1079), ('language', 1072), ('email', 1051), ('program', 1001), ('our', 987), ('list', 935)]



```python

```
