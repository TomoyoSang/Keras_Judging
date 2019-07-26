import numpy as np
import jieba 
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout

vectorizer=CountVectorizer()

label=[]
text=[]
line_counter=0
#read in 
with open('train_shuffle.txt','r',encoding='utf-8') as file1:
    for line in file1:
        label.append(int(line[0]))
        text.append(" ".join(jieba.cut(line[2:],cut_all=False)))
        line_counter+=1
with open('jieba_result.txt','w',encoding='utf-8') as file2:
    for i in range(0,line_counter):
        result=str(text[i])
        file2.write(result)
with open('label_data.txt','w',encoding='utf-8') as file3:
    for i in range(0,line_counter):
        result=str(label[i])+'\n'
        file3.write(result)

print(line_counter)

#create dictionary
word_dict={}
word_counter=0
for line in text:
    for word in line.split():
        if word not in word_dict:
            word_counter+=1
            word_dict[word]=word_counter
            

#change text into list
text_to_data=[]

with open('jieba_result.txt','r',encoding='utf-8') as file3:
    for line in file3:
        #print(line)
        cur_list=np.zeros(word_counter)
        for word in line.split():            
            cur_list[word_dict[word]-1]=1
        text_to_data.append(cur_list)

with open('text_to_data.txt','w',encoding='utf-8') as file4:
     for i in range(0,line_counter):
        result=str(text_to_data[i])+'\n'
        file4.write(result)


#print(np.shape(text_to_data))
#print(np.shape(label))

#现在已有的数据：
#label，text_to_data
#输入问题

batch_size=32

model=Sequential()
model.add(Dense(64,input_dim=word_counter))
model.add(Activation('relu'))
#model.add(Dropout(0.1))
model.add(Dense(32))
model.add(Activation('sigmoid'))
#model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


div=line_counter*3//4
x_train=np.array(text_to_data[0:div])
y_train=np.array(label[0:div])
x_test=np.array(text_to_data[div+1:])
y_test=np.array(label[div+1:])

#Rounds=5000
#for i in range(0,Rounds):
#    start=(batch_size*i)%div
#    end=min(start+batch_size,div)

#    cur_x=np.array(x_train[start:end])
#    cur_y=np.array(y_train[start:end])

#print(x_train[1])
#print(y_train[1])

#while True:
model.fit(x_train,y_train,batch_size=256, epochs=3)
score1 = model.evaluate(x_train, y_train, batch_size=32)
#score = model.evaluate(x_batch, y_batch, batch_size=1)
#for i in range(0,5000):
#    x_batch=np.array(text_to_data[i])
#    x_batch=x_batch.reshape(1,word_counter+1)
#    y_batch=np.array(label[i])
#    y_batch=y_batch.reshape(1,1)
#    model.fit(x_batch,y_batch,batch_size=1)
#    score = model.evaluate(x_batch, y_batch, batch_size=1)
score = model.evaluate(x_test, y_test, batch_size=32)
print(score)
