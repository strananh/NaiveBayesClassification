import re
import string
import numpy as np
import pandas as pd
from nltk import sent_tokenize
from nltk import word_tokenize
from os import listdir
from os import path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from prettytable import PrettyTable
# from os import getcwd

# global variables
path_2_congnghe = "dataset\\CongNghe"
path_2_others = "dataset\\Others"

dfDataSet = pd.DataFrame(columns = list(['filenames', 'contents', 'labels']))
dfDataSet = dfDataSet.astype(dtype = {'filenames':'object', 'contents':'object', 'labels':'int'})

dfTrainData = pd.DataFrame(columns = list(['filenames', 'contents', 'labels']))
dfTrainData = dfTrainData.astype(dtype = {'filenames':'object', 'contents':'object', 'labels':'int'})

dfTestData = pd.DataFrame(columns = list(['filenames', 'contents', 'labels']))
dfTestData = dfTestData.astype(dtype = {'filenames':'object', 'contents':'object', 'labels':'int'})

dictOfTrain = dict()

def log2File(log, filename = 'log.txt'):
    with open(filename, mode = 'a', encoding = 'utf-8') as f:
        f.write(str(log))
        f.write("\n")
    return 0

def readData(path_2_news, label_of_news):
    global dfDataSet

    filenames = []
    contents = []
    labels = []

    files = listdir(path_2_news)
    for f in files:
        with open(path_2_news + "\\" + f, mode = 'r', encoding = 'utf-8') as reader:
            content_of_news = ""
            for line in reader:
                content_of_news += line
        
        filenames.append(f)
        contents.append(content_of_news)
        labels.append(label_of_news)
    
    df = pd.DataFrame(columns = list(['filenames', 'contents', 'labels']))
    df = df.astype(dtype = {'filenames':'object', 'contents':'object', 'labels':'int'})
    df['filenames'] = filenames
    df['contents'] = contents
    df['labels'] = labels

    dfDataSet = dfDataSet.append(df)

def extractIdentifiers(text):
    text = text.strip()
    
    firstSpace = text.find(" ")
    vnUpperChars = "ẮẰẲẴẶĂẤẦẨẪẬÂÁÀÃẢẠĐẾỀỂỄỆÊÉÈẺẼẸÍÌỈĨỊỐỒỔỖỘÔỚỜỞỠỢƠÓÒÕỎỌỨỪỬỮỰƯÚÙỦŨỤÝỲỶỸỴ"
    vnLowerChars = "ắằẳẵặăấầẩẫậâáàãảạđếềểễệêéèẻẽẹíìỉĩịốồổỗộôớờởỡợơóòõỏọứừửữựưúùủũụýỳỷỹỵ"
    vnChars = vnUpperChars + vnLowerChars
    midPuncts = r"-'\."
    startPuncts = r"[\"\'\s\(]+"
    endPuncts = r"[\.\,\"\'\s\)]+"
    nounComponent = r"[A-Z" + vnUpperChars + r"]+[a-z" + midPuncts + vnChars + r"0-9\+]*"
    # identifiers have length of 2-7 words
    maxLen = 8
    minLen = 2

    sentences = sent_tokenize(text)
    for i in range(0, len(sentences)):
        s = sentences[i]
        s = s.strip()
        if(len(s) > 20):
            for lenOfRegex in reversed(range(minLen + 1, maxLen)):
                noun = nounComponent
                for count in range(minLen, lenOfRegex):
                    noun = noun + r"[-\s+]" + nounComponent
                reg = startPuncts + noun + endPuncts
                pattern = re.compile(pattern = reg)
                matches = pattern.findall(string = s, pos = firstSpace)
            
                if (len(matches) > 0):
                    for j in range(0, len(matches)):
                        m = matches[j]
                        s = s.replace(m, " ")
                        m = re.sub(r"^" + startPuncts, "", m)
                        m = re.sub(endPuncts + r"$", "", m)
                        m = m.strip()
                        matches[j] = m
                    # add to dictionary
                    add2DictOfTrain(matches)
        sentences[i] = s
    # return text after extract identifiers
    return " ".join(sentences)


def cleanText(text):
    result = ""

    # punctuations
    punctuations = "“”‘’…"
    punctuations = punctuations + string.punctuation
    tranlator = str.maketrans("", "", punctuations)
    
    # stop-words
    stopwords = []
    with open("vnstopword.txt", mode = 'r', encoding = 'utf-8') as f:
        for word in f:
            stopwords.append(word.strip())

    sentences = sent_tokenize(text)
    for s in sentences:
        if(len(s) > 20):
            # http links must be removed first, before punctuations removed
            s = re.sub(pattern = r"http:\S+", repl = " ", string = s)
            # remove punctuations
            s = s.translate(tranlator)
            # remove number or quantities
            s = re.sub(pattern = r"\s?\d+\w*\s?", repl = " ", string = s)
            # remove stop words
            if (len(s) > 20):
                words = word_tokenize(s.lower())
                notStopWords = [w for w in words if w not in stopwords]
                result += " " + " ".join(notStopWords)
    # return text afer clean
    return (result)

def add2DictOfTrain(keys):
    # add array of keys to dictionary
    global dictOfTrain
    pattern = re.compile(pattern = r"[\r\n]")

    value = len(dictOfTrain)
    for k in keys:
        k = pattern.sub(repl = " ", string = k)
        if (k not in dictOfTrain):
            dictOfTrain[k] = value
            value += 1

def buildDictOfTrain():
    global dfTrainData
    dfLocal = dfTrainData

    for i, row in dfLocal.iterrows():
        contents = row["contents"]
        contents = extractIdentifiers(contents)
        contents = cleanText(contents)
        dfLocal.loc[i, 'contents'] = contents

    try:
        vctz_1gram = CountVectorizer(encoding = "utf-8", ngram_range = (1, 1), max_df = 0.8, min_df = 50, max_features = 10000)
        vctz_1gram.fit(dfLocal["contents"])
        add2DictOfTrain(vctz_1gram.vocabulary_.keys())
    except:
        pass
    try:
        vctz_2gram = CountVectorizer(encoding = "utf-8", ngram_range = (2, 2), max_df = 0.8, min_df = 50, max_features = 10000)
        vctz_2gram.fit(dfLocal["contents"])
        add2DictOfTrain(vctz_2gram.vocabulary_.keys())
    except:
        pass
    try:
        vctz_3gram = CountVectorizer(encoding = "utf-8", ngram_range = (3, 3), max_df = 0.8, min_df = 50, max_features = 10000)
        vctz_3gram.fit(dfLocal["contents"])
        add2DictOfTrain(vctz_3gram.vocabulary_.keys())
    except:
        pass
    try:
        vctz_4gram = CountVectorizer(encoding = "utf-8", ngram_range = (4, 4), max_df = 0.8, min_df = 50, max_features = 10000)
        vctz_4gram.fit(dfLocal["contents"])
        add2DictOfTrain(vctz_4gram.vocabulary_.keys())
    except:
        pass

    # print(dictOfTrain)

def saveDictOfTrain2File():
    with open("output\\dictionary", mode = "a", encoding = "utf-8") as f:
        for key, value in dictOfTrain.items():
            f.write("%s:%d\n" % (key, value))

def readDictOfTrainFromFile():
    global dictOfTrain
    dictOfTrain = dict()

    with open("output\\dictionary", mode = "r", encoding = "utf-8") as f:
        for line in f:
            key = line.split(":")[0]
            value = int(line.split(":")[1])
            dictOfTrain[key] = value

def main():
    global dfDataSet
    global dfTrainData
    global dfTestData

    print("read data ...")
    readData(path_2_congnghe, label_of_news = 0)
    readData(path_2_others, label_of_news = 1)

    dfTrainData, dfTestData = train_test_split(dfDataSet, shuffle = True, test_size = 0.3)
    dfTrainData = dfTrainData.reset_index(drop = True)
    dfTestData = dfDataSet.reset_index(drop = True)

    if(path.isfile("output\\dictionary")):
        print("read dictionary from file ...")
        readDictOfTrainFromFile()
    else:
        print("build dictionary ...")
        buildDictOfTrain()
        saveDictOfTrain2File()
    
    print("transform training dataset to doc-term matrix ...")
    countVctz = CountVectorizer(encoding = 'utf-8', vocabulary = dictOfTrain, min_df = 0.1)
    countVctz._validate_vocabulary()
    countDocTerm = countVctz.transform(raw_documents = dfTrainData['contents'])
    
    print("use Mutual Information to select k-best features ...")
    if(len(dictOfTrain.items()) < 10000):
        selector = SelectKBest(score_func = mutual_info_classif, k = 'all')
    else:
        selector = SelectKBest(score_func = mutual_info_classif, k = 10000)
    selector.fit(X = countDocTerm, y = dfTrainData['labels'])
    kbestDocTerm = selector.transform(X = countDocTerm)

    newFeature = []
    scoreFeature = []
    for choose, feature, score in zip(selector.get_support(), countVctz.get_feature_names(), selector.scores_):
        if(choose == True):
            newFeature.append(feature)
            scoreFeature.append(score)

    for score, feature in sorted(zip(scoreFeature, newFeature), reverse = True):
        log2File("%s:\t\t\t%s" % (str(feature), str(score)), "output\\kbest-features")

    print("fit Multinomial Naive Bayes ...")
    nbClf = MultinomialNB()
    nbModel = nbClf.fit(X = kbestDocTerm, y = dfTrainData['labels'])
    
    print("predict test dataset")
    testDocTerm = countVctz.transform(raw_documents = dfTestData['contents'])
    kbestTestDocTerm = selector.transform(X = testDocTerm)
    predictResult = nbModel.predict(X = kbestTestDocTerm)

    clfReport = classification_report(y_true = dfTestData['labels'], y_pred = predictResult)
    refuMatrix = confusion_matrix(y_true = dfTestData['labels'], y_pred = predictResult)

    print(clfReport)
    log2File("Classification Report:\r", "output\\report")
    log2File(clfReport, "output\\report")

    print(refuMatrix)
    
# Entry point of the program
main()