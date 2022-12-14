import string
import random
import nltk
import pandas as pd
import datetime
import numpy as np
from nltk.stem import WordNetLemmatizer  # It has the ability to lemmatize.
import tensorflow as tensorF  # A multidimensional array of elements is represented by this symbol.
from tensorflow.keras import (
    Sequential,
)  # Sequential groups a linear stack of layers into a tf.keras.Model
from tensorflow.keras.layers import Dense, Dropout

nltk.download("omw-1.4")
nltk.download("punkt")  # required package for tokenization
nltk.download("wordnet")  # word database

# Get schedule
df = pd.read_excel("Schedule_Cleaned.xlsm")

data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey"],
            "responses": ["Hi there", "Hello", "Hi :)"],
        },
        {
            "tag": "goodbye",
            "patterns": ["bye", "later"],
            "responses": ["Bye", "take care"],
        },
        {
            "tag": "coach",
            "patterns": ["Who is our data coach?", "Who is our Associate coach"],
            "responses": ["Data Coach, Associate coach"],
        },
        {
            "tag": "task",
            "patterns": ["What is my task?", "show me a task"],
            "responses": ["Data Coach, Associate coach"],
        },
        {
            "tag": "tomorrow",
            "patterns": [
                "What is my task tomorrow?",
                "show me a task next day",
                "What are we covering in class tomorrow?",
                "tomorrow",
            ],
            "responses": ["Data Coach, Associate coach"],
        },
        {
            "tag": "schedule",
            "patterns": [
                "What is my task tomorrow?",
                "show me a task next day",
                "What are we covering in class tomorrow?",
            ],
            "responses": ["Data Coach, Associate coach"],
        },
        {
            "tag": "monday",
            "patterns": [
                "what am i doing on monday?",
                "What is on monday",
                "Tasks I need to complete on monday",
                "Tasks for monday",
                "Schedule for monday",
                "Upcoming monday tasks",
            ],
            "responses": ["Data Coach, Associate coach"],
        },
        {
              "tag": "thursday",
            "patterns": [
                "what am i doing on thursday?",
                "What is on Thursday",
                "Tasks I need to complete on Thursday",
                "Tasks for thursday",
                "Schedule for thursday",
                "Upcoming thursday task",
                "next Thursday"
            ],
            "responses": ["Data Coach, Associate coach"],
        }
    ]
}

lm = WordNetLemmatizer()  # for getting words
# lists
ourClasses = []
newWords = []
documentX = []
documentY = []
# Each intent is tokenized into words and the patterns and their associated tags are added to their respective lists.
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        ournewTkns = nltk.word_tokenize(pattern)  # tokenize the patterns
        newWords.extend(ournewTkns)  # extends the tokens
        documentX.append(pattern)
        documentY.append(intent["tag"])

    if (
        intent["tag"] not in ourClasses
    ):  # add unexisting tags to their respective classes
        ourClasses.append(intent["tag"])

newWords = [
    lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation
]  # set words to lowercase if not in punctuation
newWords = sorted(set(newWords))  # sorting words
ourClasses = sorted(set(ourClasses))  # sorting classes

trainingData = []  # training list array
outEmpty = [0] * len(ourClasses)
# bow model
for idx, doc in enumerate(documentX):
    bagOfwords = []
    text = lm.lemmatize(doc.lower())
    for word in newWords:
        bagOfwords.append(1) if word in text else bagOfwords.append(0)

    outputRow = list(outEmpty)
    outputRow[ourClasses.index(documentY[idx])] = 1
    trainingData.append([bagOfwords, outputRow])

random.shuffle(trainingData)
trainingData = np.array(
    trainingData, dtype=object
)  # coverting our data into an array afterv shuffling

x = np.array(list(trainingData[:, 0]))  # first trainig phase
y = np.array(list(trainingData[:, 1]))  # second training phase

iShape = (len(x[0]),)
oShape = len(y[0])
# parameter definition
ourNewModel = Sequential()
# In the case of a simple stack of layers, a Sequential model is appropriate

# Dense function adds an output layer
ourNewModel.add(Dense(128, input_shape=iShape, activation="relu"))
# The activation function in a neural network is in charge of converting the node's summed weighted input into activation of the node or output for the input in question
ourNewModel.add(Dropout(0.5))
# Dropout is used to enhance visual perception of input neurons
ourNewModel.add(Dense(64, activation="relu"))
ourNewModel.add(Dropout(0.3))
ourNewModel.add(Dense(oShape, activation="softmax"))
# below is a callable that returns the value to be used with no arguments
md = tensorF.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)
# Below line improves the numerical stability and pushes the computation of the probability distribution into the categorical crossentropy loss function.
ourNewModel.compile(loss="categorical_crossentropy", optimizer=md, metrics=["accuracy"])
# Output the model in summary
# print(ourNewModel.summary())
# Whilst training your Nural Network, you have the option of making the output verbose or simple.
ourNewModel.fit(x, y, epochs=200, verbose=0)
# By epochs, we mean the number of times you repeat a training set.


def ourText(text):
    newtkns = nltk.word_tokenize(text)
    newtkns = [lm.lemmatize(word) for word in newtkns]
    return newtkns


def wordBag(text, vocab):
    newtkns = ourText(text)
    bagOwords = [0] * len(vocab)
    for w in newtkns:
        for idx, word in enumerate(vocab):
            if word == w:
                bagOwords[idx] = 1
    return np.array(bagOwords)


def Pclass(text, vocab, labels):
    bagOwords = wordBag(text, vocab)
    ourResult = ourNewModel.predict(np.array([bagOwords]), verbose=0)[0]
    newThresh = 0.2
    yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

    yp.sort(key=lambda x: x[1], reverse=True)
    newList = []
    for r in yp:
        newList.append(labels[r[0]])
    return newList


# TODO use functiona as part of user interface
def get_res(intents, df):
    match intents[0]:
        case "coach":
            return df[["Coach/Associate", "Name"]]
        case "greeting":
            return random.choice(["Hi there", "Hello", "Hi :)"])
        case "goodbye":
            return random.choice(["Bye", "take care"])
        case "tomorrow":
            date1 = datetime.date.today() - datetime.timedelta(days=13)
            Task1 = df.query("Date == @date1")["Task"].to_list()
            CoachAssociate = df.query("Date == @date1")["Coach/Associate"].to_list()
            Name1 = df.query("Date == @date1")["Name"].to_list()
            while True:
                Name = input("Please input the Coach or Associates Coachs Name.")
                if Name not in ("Nathan", "NaN"):
                    print("Enter either this names exactly, Nathan or NaN")
                else:
                    if Name == "Nathan":
                        return "Tomorrow, {}, {}, will be teaching {}.".format(
                            CoachAssociate[1], Name1[1], Task1[1]
                        )
                    if Name == "NaN":
                        return "Tomorrow, {}, {}, will be teaching {}.".format(
                            CoachAssociate[0], Name1[0], Task1[0]
                        )
                    break
        case "monday":
            date2 = (
                datetime.date.today()
                + datetime.timedelta(days=-datetime.date.today().weekday(), weeks=1)
                - datetime.timedelta(days=21)
            )
            Task1 = df.query("Date == @date2")["Task"].to_list()
            CoachAssociate = df.query("Date == @date2")["Coach/Associate"].to_list()
            Name1 = df.query("Date == @date2")["Name"].to_list()
            am1 = df.query("Date == @date2")["AM"].to_list()
            pm1 = df.query("Date == @date2")["PM"].to_list()
            eod = df.query("Date == @date2")["EOD"].to_list()
            while True:
                Name = input("Please input the Coach or Associates Coachs Name.")
                if Name not in ("Nathan", "NaN"):
                    print("Enter either this names exactly, Nathan or NaN")
                else:
                    if Name == "Nathan":
                        return "On Monday {}, {}, {}, will be teaching {} in the morning, {} in the afternoon and {} at the end of the day.".format(
                            date2,
                            CoachAssociate[1],
                            Name1[1],
                            am1[1],
                            pm1[1],
                            eod[1],
                        )

                    if Name == "NaN":
                        return "On Monday, {}, {}, will be teaching {}.".format(
                            CoachAssociate[0], Name1[0], Task1[0]
                        )
                    break
        case "thursday":
            date2= (
                datetime.date.today()
                + datetime.timedelta(days=-datetime.date.today().weekday(), weeks=1)
                - datetime.timedelta(days=18)
            )
            Task1 = df.query("Date == @date2")["Task"].to_list()
            CoachAssociate = df.query("Date == @date2")["Coach/Associate"].to_list()
            Name1 = df.query("Date == @date2")["Name"].to_list()
            am1 = df.query("Date == @date2")["AM"].to_list()
            pm1 = df.query("Date == @date2")["PM"].to_list()
            eod = df.query("Date == @date2")["EOD"].to_list()
            while True:
                Name = input("Please input the Coach or Associates Coachs Name.")
                if Name not in ("Nathan", "NaN"):
                    print("Enter either this names exactly, Nathan or NaN")
                else:
                    if Name == "Nathan":
                        return "On Thursday {}, {}, {}, will be teaching {} in the morning, {} in the afternoon and {} at the end of the day.".format(
                            date2,
                            CoachAssociate[1],
                            Name1[1],
                            am1[1],
                            pm1[1],
                            eod[1],
                        )

                    if Name == "NaN":
                        return "Tomorrow, {}, {}, will be teaching {}.".format(
                            CoachAssociate[0], Name1[0], Task1[0]
                        )
                    break
            return               
        case _:
            return "I do not understand please ask a different question"


if __name__ == "__main__":
    print("Hi, how can I help you today?")
    while True:
        newMessage = input("")
        intents = Pclass(newMessage, newWords, ourClasses)
        ourResult = get_res(intents, df=df)
        print(ourResult)
