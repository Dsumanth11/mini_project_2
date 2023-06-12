# from crypt import methods
from flask import Flask,render_template,request

app=Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/whatpdis")
def hello1():
    return render_template("whatpdis.html")

@app.route("/Diseases")
def hello2():
    return render_template("Diseases.html")

@app.route("/feedback")
def hello3():
    return render_template("feedback.html")

@app.route("/Diabetes")
def hello4():
    return render_template("diabetes.html")

@app.route("/sub",methods=['POST'])
def submit():
    if request.method == "POST":
        # name = request.form["username"]
        a=float(request.form["age"])
        b=float(request.form["sex"])
        c=float(request.form["cp"])
        d=float(request.form["trestbps"])
        e=float(request.form["chol"])
        f=float(request.form["fbs"])
        g=float(request.form["restecg"])
        h=float(request.form["thalach"])
        i=float(request.form["exang"])
        j=float(request.form["oldpeak"])
        k=float(request.form["slope"])
        l=float(request.form["ca"])
        m=float(request.form["thal"])
        # import pandas as pd
        # import numpy as np
        # df=pd.read_csv("diabetes.csv")

        # x=df.iloc[:,0:8].values
        # y=df.iloc[:,8].values

        # from sklearn.model_selection import train_test_split
        # x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        # from sklearn.preprocessing import StandardScaler
        # sc=StandardScaler()
        # x_train=sc.fit_transform(x_train)
        # x_test=sc.transform(x_test)
        # from sklearn.linear_model import LogisticRegression
        # Log= LogisticRegression(random_state=0)
        # Log.fit(x_train,y_train)
        # y_pred=Log.predict(x_test)
        # z=Log.predict(sc.transform([[a,b,c,d,e,f,g,h]]))
        # if z==1:
        #     name="YES"
        # else:
        #     name="NO"
        
        import sys
        import pandas as pd
        import numpy as np
        import sklearn
        import matplotlib
        import keras
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        import matplotlib.pyplot as plt
        from pandas.plotting import scatter_matrix 
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

        # the names will be the names of each column in our pandas DataFrame
        names = ['age',
                'sex',
                'cp',
                'trestbps',
                'chol',
                'fbs',
                'restecg',
                'thalach',
                'exang',
                'oldpeak',
                'slope',
                'ca',
                'thal',
                'class']

        # read the csv
        cleveland = pd.read_csv(url, names=names)        

        data = cleveland[~cleveland.isin(['?'])]

        # drop rows with NaN values from DataFrame
        data = data.dropna(axis=0)
        data = data.apply(pd.to_numeric)
        from sklearn import model_selection

        X = np.array(data.drop(['class'], 1))
        y = np.array(data['class'])

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

        from keras.utils.np_utils import to_categorical

        Y_train = to_categorical(y_train, num_classes=None)
        Y_test = to_categorical(y_test, num_classes=None)

        from keras.models import Sequential
        from keras.layers import Dense
        from keras.optimizers import Adam

        # define a function to build the keras model
        def create_model():
            # create model
            model = Sequential()
            model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
            model.add(Dense(4, kernel_initializer='normal', activation='relu'))
            model.add(Dense(5, activation='softmax'))

            # compile model
            adam = Adam(learning_rate=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            return model

        model = create_model()

        # print(model.summary())

        # fit the model to the training data
        model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose = 1)

        # ---------------------- duplicate can comment below till main
        Y_train_binary = y_train.copy()
        Y_test_binary = y_test.copy()

        Y_train_binary[Y_train_binary > 0] = 1
        Y_test_binary[Y_test_binary > 0] = 1

        def create_binary_model():
                # create model
            model = Sequential()
            model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
            model.add(Dense(4, kernel_initializer='normal', activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            # Compile model
            adam = Adam(lr=0.001)
            model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
            return model

        binary_model = create_binary_model()

        # fit the binary model on the training data
        binary_model.fit(X_train, Y_train_binary, epochs=100, batch_size=10, verbose = 1)

        from sklearn.metrics import classification_report, accuracy_score

        categorical_pred = np.argmax(model.predict(X_test), axis=1)

        binary_pred = np.round(binary_model.predict(X_test)).astype(int)

        # ------------------------main----------------------


        abc=np.array([[a,b,c,d,e,f,g,h,i,j,k,l,m]])
        # from sklearn.metrics import classification_report, accuracy_score
        categorical_pred = np.argmax(model.predict(abc), axis=1)
        print(categorical_pred[0])  # this is the output to be sent to front end page

        # print('Results for Categorical Model')
        # print(accuracy_score(y_test, categorical_pred))
        # print(classification_report(y_test, categorical_pred))
        # ////#
        
    return render_template("sub.html",n=categorical_pred[0],a1=a,b1=b,c1=c,d1=d,e1=e,f1=f,g1=g,h1=h,i1=i,j1=j,k1=k,l1=l,m1=m) 

if __name__=="__main__":
    app.run(debug=True)