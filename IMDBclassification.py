
# coding: utf-8

# In[2]:


#Geoffrey Rozo
#MSDS 458 - Assignment #3

#Chollet (2018)
#Chollet, F. 2018. Deep Learning with Python. 
#Shelter Island, N.Y.: Manning. [ISBN-13: 978-1617294433] 

#chapter 6, pages 186...

from keras.layers import Embedding

import matplotlib.pyplot as plt

import time

embedding_layer = Embedding(1000, 64)


# In[3]:


from keras.datasets import imdb
from keras import preprocessing

max_features = 10000      #number of words as features
maxlen = 20               #only looking at this many (20) words/cut off in text...
 
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features)       #loading imdb data as lists of integers

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
#turned the lists of integers into 2D integer tensor (samples, maxlen)


# In[4]:


x_train[0]  #list of reviews, each review a list of word indices(encoding a sequence of words) #pg 68


# In[5]:


y_train[0]   #y train and y test are lists of 0s and 1s, 0 = negative review, 1 = positive review


# In[6]:


from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))   #max input length specified in embed. layer
#after embedding layer, activations have shape (samples, maxlen, 8)

model.add(Flatten())    #flattens 3D tensor of embeddings to 2D tensor (samples, maxlen*8)

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)


# In[23]:


history_dict = history.history

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[24]:


plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[26]:


model2 = Sequential()
model2.add(Embedding(10000, 8, input_length=maxlen))   #max input length specified in embed. layer
#after embedding layer, activations have shape (samples, maxlen, 8)

model2.add(Flatten())    #flattens 3D tensor of embeddings to 2D tensor (samples, maxlen*8)

model2.add(Dense(16, activation='relu'))

model2.add(Dense(1, activation='sigmoid'))
model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model2.summary()

history2 = model2.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)


# In[27]:


history2_dict = history2.history

acc = history2_dict['acc']
val_acc = history2_dict['val_acc']
loss_values = history2_dict['loss']
val_loss_values = history2_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[28]:


plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[7]:


#so above is our simple example of only looking at 1 Dense layer, this model DOES NOT consider
#order/sequence of words....

#Let's study this question for imdb dataset: is order of words in the text important?

#Using a dense layer, we get up to validation accuracy = ~75%

#NEXT, we compare to RNN or LSTM structures to see whether ORDER makes a difference?

#Chollet chapter 6.2 (pages 196...)

from keras.layers import SimpleRNN


# In[8]:


#....... now to use these model RNN structures on IMDB movie review classification problem.
#pages 200...on


# In[9]:


from keras.preprocessing import sequence

max_features = 10000     #numbers of words as features
maxlen = 500             #cuts off at 500 words in the text
batch_size = 32

(input_train, y_train), (input_test, y_test) = imdb.load_data(
    num_words=max_features)

print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

print('input_train shape: ', input_train.shape)
print('input_test shape: ', input_test.shape)


# In[10]:


model3 = Sequential()
model3.add(Embedding(max_features, 32))
model3.add(SimpleRNN(32))
model3.add(Dense(1, activation='sigmoid'))

model3.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

processing_time = []
    
start_time = time.clock()

history3 = model3.fit(input_train, y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[11]:


history3_dict = history3.history

acc = history3_dict['acc']
val_acc = history3_dict['val_acc']
loss_values = history3_dict['loss']
val_loss_values = history3_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss (1 SimpleRNN(32))')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('1RNN_32_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy (1 SimpleRNN(32))')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('1RNN_32_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[12]:


#recommended to stack several RNN layers to increase representational power of the network...
#NEED to get all the intermed. layers to return full sequence of outputs... (page 199)


# In[13]:


model4 = Sequential()
model4.add(Embedding(10000, 32))
model4.add(SimpleRNN(32, return_sequences=True))
model4.add(SimpleRNN(32, return_sequences=True))
model4.add(SimpleRNN(32, return_sequences=True))
model4.add(SimpleRNN(32))
model4.add(Dense(1, activation='sigmoid'))

model4.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

processing_time = []
    
start_time = time.clock()

history4 = model4.fit(input_train, y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[14]:


history4_dict = history4.history

acc = history4_dict['acc']
val_acc = history4_dict['val_acc']
loss_values = history4_dict['loss']
val_loss_values = history4_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss (4 SimpleRNN(32))')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('4RNN_32_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy (4 SimpleRNN(32))')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('4RNN_32_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[15]:


#so, I built SimpleRNN models before, with 1 and 4 SimpleRNN layers (32 nodes)...
#but page 202-206 describes using LSTM or GRU layers to combat against the Vanishing gradient problem.

#page 204 ==> "you don't need to understand anything about the specific architecture of an LSTM
#cell; as a human, it shouldn't be your job to understand it. Just keep in mind what the
#LSTM cell is meant to do: allow past information to be reinjected at a later time, thus fighting
#the vanishing-gradient problem."


# In[16]:


#"...only specify the output dimensionality of the LSTM layer; leave every other argument...at the
#Keras defaults. Keras has good defaults, and things will almost always "just work" w/o having
#to tune parameters by hand." - page 204-205 Chollet (2018)

from keras.layers import LSTM


# In[17]:


model5 = Sequential()
model5.add(Embedding(max_features, 32))
model5.add(LSTM(32))
model5.add(Dense(1, activation='sigmoid'))

model5.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

processing_time = []
    
start_time = time.clock()

history5 = model5.fit(input_train, y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


#weird... just plugging in the battery into my laptop made training way faster.

#from 100-110 seconds, down to ~40 seconds...


# In[18]:


history5_dict = history5.history

acc = history5_dict['acc']
val_acc = history5_dict['val_acc']
loss_values = history5_dict['loss']
val_loss_values = history5_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss (1 LSTM(32))')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('1LSTM_32_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy (1 LSTM(32))')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('1LSTM_32_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[19]:


model6 = Sequential()
model6.add(Embedding(max_features, 32))
model6.add(LSTM(32, return_sequences=True))
model6.add(LSTM(32, return_sequences=True))
model6.add(LSTM(32, return_sequences=True))
model6.add(LSTM(32))
model6.add(Dense(1, activation='sigmoid'))

model6.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

processing_time = []
    
start_time = time.clock()

history6 = model6.fit(input_train, y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[20]:


history6_dict = history6.history

acc = history6_dict['acc']
val_acc = history6_dict['val_acc']
loss_values = history6_dict['loss']
val_loss_values = history6_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss (4 LSTM(32))')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('4LSTM_32_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy (4 LSTM(32))')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('4LSTM_32_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[21]:


#model 3, 4, 5, 6...

test_loss3, test_acc3 = model3.evaluate(input_test, y_test)
test_loss4, test_acc4 = model4.evaluate(input_test, y_test)
test_loss5, test_acc5 = model5.evaluate(input_test, y_test)
test_loss6, test_acc6 = model6.evaluate(input_test, y_test)


# In[25]:


#test_acc3   #0.78184
#test_acc4   #0.7816
#test_acc5   #0.85816
#test_acc6   #0.80564


# In[ ]:


#try one last experiment... same setup of comparing 1 and 4 simpleRNN, then 1 and 4 LSTM,
#but let's change the max_features = 20000, increase the number words to see if that matters...


# In[26]:


max_features2 = 20000     #numbers of words as features
maxlen = 500             #cuts off at 500 words in the text
batch_size = 32

(input_train, y_train), (input_test, y_test) = imdb.load_data(
    num_words=max_features2)

print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

print('input_train shape: ', input_train.shape)
print('input_test shape: ', input_test.shape)


# In[27]:


model10 = Sequential()
model10.add(Embedding(max_features2, 32))
model10.add(SimpleRNN(32))
model10.add(Dense(1, activation='sigmoid'))

model10.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

processing_time = []
    
start_time = time.clock()

history10 = model10.fit(input_train, y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[28]:


history10_dict = history10.history

acc = history10_dict['acc']
val_acc = history10_dict['val_acc']
loss_values = history10_dict['loss']
val_loss_values = history10_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss (1 SimpleRNN(32), max_feat=20k)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('1RNN_32_20k_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy (1 SimpleRNN(32), max_feat=20k)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('1RNN_32_20k_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[61]:


#compare, see if the node size in the RNN matters when we changed/doubled the max_features to 20,000


# In[62]:


model11 = Sequential()
model11.add(Embedding(max_features2, 32))
model11.add(SimpleRNN(64))
model11.add(Dense(1, activation='sigmoid'))

model11.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

processing_time = []
    
start_time = time.clock()

history11 = model11.fit(input_train, y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[63]:


history11_dict = history11.history

acc = history11_dict['acc']
val_acc = history11_dict['val_acc']
loss_values = history11_dict['loss']
val_loss_values = history11_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss (1 SimpleRNN(64), max_feat=20k)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('1RNN_64_20k_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy (1 SimpleRNN(64), max_feat=20k)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('1RNN_64_20k_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[ ]:


#just stick to the 32 nodes, better than 64


# In[29]:


model12 = Sequential()
model12.add(Embedding(max_features2, 32))
model12.add(SimpleRNN(32, return_sequences=True))
model12.add(SimpleRNN(32, return_sequences=True))
model12.add(SimpleRNN(32, return_sequences=True))
model12.add(SimpleRNN(32))
model12.add(Dense(1, activation='sigmoid'))

model12.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

processing_time = []
    
start_time = time.clock()

history12 = model12.fit(input_train, y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[30]:


history12_dict = history12.history

acc = history12_dict['acc']
val_acc = history12_dict['val_acc']
loss_values = history12_dict['loss']
val_loss_values = history12_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss (4 SimpleRNN(32), max_feat=20k)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('4RNN_32_20k_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy (4 SimpleRNN(32), max_feat=20k)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('4RNN_32_20k_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[31]:


model13 = Sequential()
model13.add(Embedding(max_features2, 32))
model13.add(LSTM(32))
model13.add(Dense(1, activation='sigmoid'))

model13.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

processing_time = []
    
start_time = time.clock()

history13 = model13.fit(input_train, y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[32]:


history13_dict = history13.history

acc = history13_dict['acc']
val_acc = history13_dict['val_acc']
loss_values = history13_dict['loss']
val_loss_values = history13_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss (1 LSTM(32), max_feat=20k)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('1LSTM_32_20k_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy (1 LSTM(32), max_feat=20k)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('1LSTM_32_20k_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[33]:


model14 = Sequential()
model14.add(Embedding(max_features2, 32))
model14.add(LSTM(32, return_sequences=True))
model14.add(LSTM(32, return_sequences=True))
model14.add(LSTM(32, return_sequences=True))
model14.add(LSTM(32))
model14.add(Dense(1, activation='sigmoid'))

model14.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

processing_time = []
    
start_time = time.clock()

history14 = model14.fit(input_train, y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[34]:


history14_dict = history14.history

acc = history14_dict['acc']
val_acc = history14_dict['val_acc']
loss_values = history14_dict['loss']
val_loss_values = history14_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss (4 LSTM(32), max_feat=20k)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('4LSTM_32_20k_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy (4 LSTM(32), max_feat=20k)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('4LSTM_32_20k_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[35]:


#models 10, 12, 13, 14...

test_loss10, test_acc10 = model10.evaluate(input_test, y_test)
test_loss12, test_acc12 = model12.evaluate(input_test, y_test)
test_loss13, test_acc13 = model13.evaluate(input_test, y_test)
test_loss14, test_acc14 = model14.evaluate(input_test, y_test)


# In[39]:


#test_acc10   #0.79488
#test_acc12   #0.78932
#test_acc13   #0.8552
#test_acc14   #0.82968


# In[ ]:


#just want to see, if 2 LSTMs is any better...


# In[40]:


model15 = Sequential()
model15.add(Embedding(max_features2, 32))
model15.add(LSTM(32, return_sequences=True))
model15.add(LSTM(32))
model15.add(Dense(1, activation='sigmoid'))

model15.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

processing_time = []
    
start_time = time.clock()

history15 = model15.fit(input_train, y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[41]:


history15_dict = history15.history

acc = history15_dict['acc']
val_acc = history15_dict['val_acc']
loss_values = history15_dict['loss']
val_loss_values = history15_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss (2 LSTM(32), max_feat=20k)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('2LSTM_32_20k_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy (2 LSTM(32), max_feat=20k)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('2LSTM_32_20k_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[42]:


test_loss15, test_acc15 = model15.evaluate(input_test, y_test)

test_acc15   #


# In[ ]:


#overall:

#LSTM layers achieved highest test accuracies (1 LSTM layer with 10k max_Features == 85.8% accuracy)
#LSTM 1 layer, 20k max_features were least processing time, 1/4 less than 4 LSTM layers, and
#better test accuracy than any of the simpleRNN layers...

