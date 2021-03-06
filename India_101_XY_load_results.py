######################################################
##   Code for traceless routing using NNs (Classification mode)
######################################################
import matplotlib.pyplot as plt
import numpy as np
from math import floor
import csv, time, utm, sys, pickle , math
from os import listdir
from os.path import isfile, join
import datetime
from termcolor import colored
from keras.callbacks import Callback, EarlyStopping
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation , Flatten
from keras.layers.convolutional import  Convolution2D , MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adadelta , Adam , RMSprop
from keras.regularizers import l1, l2, l1l2, activity_l1, activity_l2, activity_l1l2
from keras.utils import np_utils

Approach = 'XY' 

def sopen( p ):
    a = open( p )
    reader = csv.reader( a , delimiter='\t'  )
    return list(reader)
    
# function to format the input for the NN
def form_inp( x , hh , vv ) :
    xx = []    
    for i in x :
        u = np.zeros((  hh+1 ))   
        u   [ i[0] ] = i[2]
        v = np.zeros((  vv+1 ))   
        v   [ i[1] ] = i[2]
        s = np.hstack(( u,v ))        
        u = np.zeros((  hh+1 ))   
        u   [ i[3] ] = i[5]
        v = np.zeros((  vv+1 ))   
        v   [ i[4] ] = i[5]
        f = np.hstack(( u,v ))  
        xx.append(  np.hstack(( s,f ))   )  
    xx = np.array(xx)
    return xx  

mypath = "/home/local/ANT/nikosc/Documents/codes/Routing_graph/India/data/"  

data = sopen( mypath + 'India_Delivery_data' )   
   
TIME_index = data[0].index('change_date') ;
LAT_index = data[0].index('latitude') ;
LON_index = data[0].index('longitude') ;
EMP_index = data[0].index('holder_employee_id') ;
data = sorted( data[1:] , key=lambda fail: fail[TIME_index] )

# Separate into training and testing datasets
trains = data[:150000]
tests = data[150000:]

# List of employee IDs
employees = [ tup[EMP_index] for tup in data ] 
employees = list(set( employees ))

# Training dataset
TRAINS = []
for i in employees :
    hmm = [ [ tup[LAT_index] , tup[LON_index] , tup[TIME_index] ] for tup in trains if tup[EMP_index] == i ]
    hmm = sorted( hmm , key=lambda fail: fail[2] )
    TRAINS.append([ i , hmm ])

train_edges = []
for i in TRAINS :
    for j in range( len(i[1])-1 ) :
        k = j+1
        a = i[1][j][2]
        b = i[1][k][2]
        c = datetime.datetime( int(a[:4]) , int(a[5:7]) , int(a[8:10]) , \
                               int(a[11:13]) , int(a[14:16]) , int(a[17:19]) )
        d = datetime.datetime( int(b[:4]) , int(b[5:7]) , int(b[8:10]) , \
                               int(b[11:13]) , int(b[14:16]) , int(b[17:19]) )  
        diff = d-c                     
        if diff.days == 0 and diff.seconds <= 1200   :
            aa = list( utm.from_latlon( float( i[1][j][0]  ) , float( i[1][j][1] )  ) )[0:2] 
            bb = list( utm.from_latlon( float( i[1][k][0]  ) , float( i[1][k][1] )  ) )[0:2]  
            train_edges.append([ aa , bb , diff.seconds ])


# Testing dataset
TESTS = []
for i in employees :
    hmm = [ [ tup[LAT_index] , tup[LON_index] , tup[TIME_index] ] for tup in tests if tup[EMP_index] == i ]
    hmm = sorted( hmm , key=lambda fail: fail[2] )
    TESTS.append([ i , hmm ]) 
    
test_edges = []
for i in TESTS :
    for j in range( len(i[1])-1 ) :
        k = j+1
        a = i[1][j][2]
        b = i[1][k][2]
        c = datetime.datetime( int(a[:4]) , int(a[5:7]) , int(a[8:10]) , \
                               int(a[11:13]) , int(a[14:16]) , int(a[17:19]) )
        d = datetime.datetime( int(b[:4]) , int(b[5:7]) , int(b[8:10]) , \
                               int(b[11:13]) , int(b[14:16]) , int(b[17:19]) )  
        diff = d-c                     
        if diff.days == 0 and diff.seconds <= 1200   :
            aa = list( utm.from_latlon( float( i[1][j][0]  ) , float( i[1][j][1] )  ) )[0:2] 
            bb = list( utm.from_latlon( float( i[1][k][0]  ) , float( i[1][k][1] )  ) )[0:2]  
            test_edges.append([ aa , bb , diff.seconds ])


##################################
# TESSELATE THE AREA             
##################################  
# Find the area
P = [] 
for i in train_edges :
    P.append( i[0] )
    P.append( i[1] )
for i in test_edges :
    P.append( i[0] )
    P.append( i[1] )
P = np.asarray( P )
x_min, x_max = P[:, 0].min() - 1000, P[:, 0].max() + 1000
y_min, y_max = P[:, 1].min() - 1000, P[:, 1].max() + 1000

limits = [  x_min-2000 , x_max+2000 , y_min-2000 , y_max+2000 ]
limits = [  212000 , 234000 , 1907000 , y_max ]
min_lats= limits[0]
max_lats= limits[1]  
min_longs= limits[2]  
max_longs= limits[3] 

# Plot the data points
plt.figure(1)
plt.close()
plt.plot(P[:, 0], P[:, 1], 'k.', markersize = 5)     
plt.xlim( limits[0], limits[1] )
plt.ylim( limits[2], limits[3] )
#plt.xlim(x_min, x_max)
#plt.ylim(y_min, y_max)
plt.show()

train_edges = [ tup for tup in train_edges if \
          tup[0][0] >= min_lats and tup[0][0] <= max_lats and\
          tup[0][1] >= min_longs and tup[0][1] <= max_longs and\
          tup[1][0] >= min_lats and tup[1][0] <= max_lats and\
          tup[1][1] >= min_longs and tup[1][1] <= max_longs ]
test_edges = [ tup for tup in test_edges if \
          tup[0][0] >= min_lats and tup[0][0] <= max_lats and\
          tup[0][1] >= min_longs and tup[0][1] <= max_longs and\
          tup[1][0] >= min_lats and tup[1][0] <= max_lats and\
          tup[1][1] >= min_longs and tup[1][1] <= max_longs ] 



# Format input-output
x = np.array( [ [ tup[0][0] , tup[1][0] ] for tup in train_edges ] ).ravel()
x = np.hstack( (  x  , np.array( [ [ tup[0][0] , tup[1][0] ] for tup in test_edges ] ).ravel()   )  )
mx = np.mean(x)
sx = np.std(x)
y = np.array( [ [ tup[0][1] , tup[1][1] ] for tup in train_edges ] ).ravel()
y = np.hstack( (  y  , np.array( [ [ tup[0][1] , tup[1][1] ] for tup in test_edges ] ).ravel()   )  )
my = np.mean(y)
sy = np.std(y)
 
trainers_in = np.array([ [ (tup[0][0]-mx)/sx , (tup[0][1]-my)/sy ,  (tup[1][0]-mx)/sx , (tup[1][1]-my)/sy ]   \
                        for tup in train_edges ] )
trainers_out =  np.array([ tup[-1] for tup in train_edges ])
mt = np.mean(trainers_out)
st = np.std(trainers_out)
                        
testers_in = np.array([ [ (tup[0][0]-mx)/sx , (tup[0][1]-my)/sy ,  (tup[1][0]-mx)/sx , (tup[1][1]-my)/sy ]   \
                        for tup in test_edges   ] )


##################################################################
#######   NN training   ###################################
################################################################## 
Method =  'Class'  
Nodes =  [ 500 , 1000 ] 
Layers = [ 1 , 2 , 3 ] 
Iterations = [ 30 , 60 , 100 ]
Batch_sizes = [ 100 , 250 , 500 ]
Activations =  'relu' 

interval = 50  # Divide the range of delivery times into time intervals of width 'interval'
y =  np.array([  int( tup/interval ) for tup in trainers_out  ])
y = np_utils.to_categorical(  y , max(y)+1 ) # Convert to vector with all zeros and a single 1 
x =  trainers_in 

for nod in Nodes [ : ] :
  for lay in Layers [ : ]:
    for iters in Iterations [ : ] :
      for batch_size in Batch_sizes [ : ] :      
            model = Sequential()    
            for i in range(lay): 
                model.add( Dense( nod , input_dim=x[0].shape[0] , init='uniform' , activation = Activations ))
                model.add(Dropout(0.5))
            model.add( Dense( len(y[0]) ,   activation='softmax' )   )
            learning_rate = .1 
            opt = SGD( lr = learning_rate ) 
            model.compile( loss='categorical_crossentropy' ,  optimizer=opt )                
            for i in range( iters ):
                #print i
                hist = model.fit( x , y , verbose=0 , validation_split = 0  , nb_epoch=1 ,  batch_size = 500 )                                          
 

            ##################################################################
            #######   TESTING   ###################################
            ################################################################## 
            predd =  model.predict( testers_in )
            pred = []
            for i in predd:
                pp=0
                for j in range(len(predd[0])):
                    pp +=   (j*interval+interval/2)*i[j]
                pred.append( pp )
            
            actuals = [ tup[-1] for tup in test_edges  ]  # actual times 
            devs = []
            abs_err = 0
            rel_err = 0
            for i in range(len(pred)) :
                devs.append(  pred[i] - actuals[i]  )
                abs_err +=  ( pred[i]  - actuals[i] )**2
                rel_err += pred[i]  - actuals[i]
            abs_err = abs_err/float(len(pred))
            rel_err = rel_err/float(len(pred))

            print colored( 'Method: ' + Method +  ' //  Nodes: '   + str(nod) + ' // Layers: '   + str(lay)   \
                    + ' // Activations: '   + str(Activations)  + ' // Iterations: '   + str(iters) \
                    + ' // Batch size: '   + str(batch_size)  ,'magenta' ) 
            print colored( 'Average abs deviation from NN prediction: '   + str(abs_err) , 'blue' )  
            print colored( 'Average deviation from NN prediction: '       + str(rel_err) , 'blue' )  
            
            
            # save NN
            json_string = model.to_json()
            open( mypath + 'Hyderabad' +'_' +  Approach + '_' +  Method + '_' + str(lay) + '_' + str(nod) + '_' + Activations  \
                 + '_' + str(iters) + '_' + str(batch_size) +'.json' , 'w' ).write(json_string)
            model.save_weights( mypath + 'Hyderabad' + Approach + '_' +  Method + '_' + str(lay) + '_' + str(nod) + '_' + Activations  \
                 + '_' + str(iters) +'.h5' , overwrite=True )






# Plot results
devs = sorted(devs)
aa = range(-500,501,20)
devz = []
for i in range(len(aa)-1) : 
    a = 0
    b = 0
    for j in range(len(devs)) :
        if devs[j] < aa[i+1] and devs[j] >= aa[i]:
            a += 1
    devz.append( a )
                                               
plt.figure(2)
plt.clf()
#plt.close()
plt.plot( aa[:-1] , devz , label = 'NN')
plt.title( 'Hyderabad // Method: ' + Method +  ' //  Nodes: '   + str(Nodes) + ' // Layers: '   + str(Layers)   \
                        + ' // Activations: '   + str(Activations)  , fontsize=  13)
plt.xlabel('Deviation from actual times (sec)' , fontsize=20 )
plt.ylabel('Number of location pairs' , fontsize=20  )
plt.legend( fontsize=20 )
plt.show()    
sys.stdout.flush()        
time.sleep( .1 )  
