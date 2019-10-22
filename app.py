from flask import Flask, request, url_for, redirect, render_template
import image
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn import linear_model
import matplotlib.pyplot as xlt
import matplotlib.pyplot as plt

global k 
dict = {'none':0,'nosugar':0,'banana':58,'apple':34,'dates':103,'mango':60,'milk':31,'coconut juice':41,'coke':68,'orange':42,'brownrice':72,'basmatirice':58,'noodles':52,'oats':50,}; 



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/first', methods=['GET', 'POST'])
def first():
    if request.method == 'POST':
         a = request.form.get('first')
         b = request.form.get('second')
         c = request.form.get('third')
         d = request.form.get('fourth')
         e = request.form.get('fifth')
         f = request.form.get('sixth')
         g = request.form.get('seventh')
         h = request.form.get('eight')
         i = request.form.get('nineth')
         j = request.form.get('tenth')
        #statement to check the proper copying of form values  
        # x=int(a)+int(b)+int(c)+int(d)+int(e)+int(f)+int(g)+int(h)+int(i)+int(j)+int(k)
         
        #SVM MODULE
        # path1='maindata.csv'
        # df = pd.read_csv(path1)
        # X = df[['AGE','SEX','BMI','BP','TC','LDL','HDL','TG','TCH(FORMULA)','LTG(EYE)']]    
        # y = df['GLUCOSE']
         #print (df.corr())
         #normalized_X = preprocessing.normalize(X)
         #normalized_Y = y.reshape(-1,1)
         #train_X = normalized_X[:300]
         #train_Y = normalized_Y[:300]
         #test_X = normalized_X[-100:]
         #test_Y = normalized_Y[-100:]
         #clf = svm.SVR(C=1.0, cache_size=200, epsilon=0.1, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
         #clf.fit(train_X, train_Y)
         #test_Y_pred = clf.predict(test_X)
         #train_Y_pred = clf.predict(train_X)
         #print((test_Y_pred).reshape(-1,1))
         #X_lately=[[int(a),int(b),int(c),int(d),int(e),int(f),int(g),int(h),int(i),int(j)]]
         #value=clf.predict(X_lately)
         #print(value)
         #plt.scatter(test_Y,test_Y,color='darkorange')#actual glucose values
         #plt.scatter(test_Y_pred,test_Y_pred,color='navy')#predicted glucose values
         #plt.xlabel('Actual Glucose')
         #plt.ylabel('Predicted Glucose')
         #plt.title('Support Vector Regression')
         #plt.legend()
         #plt.savefig('static/glucosepredictor.png')   





         #LARS MODULE
         path1='maindata.csv'
         df = pd.read_csv(path1)
         X = df[['AGE','SEX','BMI','BP','TC','LDL','HDL','TG','TCH(FORMULA)','LTG(EYE)']]    
         y = df['GLUCOSE']
         print(df.corr())
         train_X = X[:300]
         train_Y = y[:300]
         test_X = X[-100:]
         test_Y = y[-100:]
         clf = linear_model.Lars()
         clf.fit(train_X, train_Y)
         test_Y_pred = clf.predict(test_X)
         train_Y_pred = clf.predict(train_X)
         print((test_Y_pred).reshape(-1,1))
         X_lately=[[float(a),float(b),float(c),float(d),float(e),float(f),float(g),float(h),float(i),float(j)]]
         value=clf.predict(X_lately)
         print(value)
         xlt.scatter(test_Y,test_Y,color='darkorange')
         xlt.scatter(test_Y_pred,test_Y_pred,color='navy')
         xlt.xlabel('Actual Glucose')
         xlt.ylabel('Predicted Glucose')
         xlt.title('Least Angle Regression')
         xlt.legend()
         xlt.savefig('static/glucosepredictor.png')
         dict['predgluc']=value;

         return render_template('first.html', variable = value)
    
    return render_template('first.html', variable = value)

@app.route('/second', methods=['GET', 'POST'])
def second():
    return render_template('second.html',)


@app.route('/third', methods=['GET', 'POST'])
def third():
    if request.method == 'POST':
         n=0
         v1=0
         bglvl = dict.get('predgluc') 
         print bglvl
         global predbg
         predbg=0
         p = request.form.get('plan')
         t = request.form.get('tired')
         plan=int(p)
         tired=int(t)
         food1 = request.form.get('food1')
         q1 = request.form.get('q1')

         v1=v1+int(food1)*int(q1)
         n=n+int(q1)

         food2 = request.form.get('food2')
         q2 = request.form.get('q2')
         v1=v1+int(food2)*int(q2)
         n=n+int(q2)

         food3 = request.form.get('food3')
         q3 = request.form.get('q3')
         v1=v1+int(food3)*int(q3)
         n=n+int(q3)
         total=v1

         if(bglvl<100):
            gi=total/n;
            if(gi!=0):
                spike=((0.38*gi) + 22);
            else:
                spike=0;
            if(tired == 1):
                bglvl-=20;
            predbg=bglvl+spike
            print predbg," is your predicted blood glucose \n"
            print spike
            if (spike<=35):
                if (plan==1):
                    print "It is ok to eat \n";
                    x='It is ok to eat.'
                if (plan==2):
                    print "This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables \n";
                    x='This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables'
                if (plan==3):
                    print "This food is insufficient to keep up with your activities, Please eat more food \n";
                    x='This food is insufficient to keep up with your activities, Please eat more food'
                plt.axis([0, 4.0, 0, 300])
                plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/4), bglvl+(spike/2),bglvl+(3*spike/4),predbg, predbg-(spike/4),predbg-(spike/2),predbg-(3*spike/4),predbg-spike])
                plt.xlabel('Time(hr)')
                plt.ylabel('Blood Glucose level(mg/DL)')
                plt.title('Glycemic Excursions for the next 4 hours !')
                plt.savefig('static/foodpredictor.png')
            if (spike>35 and spike<=48):
                if (plan==2):
                    print "It is ok to eat \n";
                    x='It is ok to eat.'
                if (plan==3):
                    x='This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables '
                    print "This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables \n";
                if (plan==1):
                    x='You are consuming excessive food , please reduce the quantities to keep up with your activities.'
                    print "You are consuming excessive food , please reduce the quantities to keep up with your activities. \n";
            
                plt.axis([0, 4.0, 0, 300])
                plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/3), bglvl+(2*spike/3),predbg,predbg-(spike/5), predbg-(2*spike/5),predbg-(3*spike/5),predbg-(4*spike/5),predbg-spike])
                plt.xlabel('Time(hr)')
                plt.ylabel('Blood Glucose level(mg/DL)')
                plt.title('Glycemic Excursions for the next 4 hours !')
                plt.savefig('static/foodpredictor.png')
            if (spike>48):
                if (plan==3):
                    x='It is ok to eat.'
                    print "It is ok to eat \n";
                if (plan==2):
                    x='You are consuming excessive food , please reduce your quantity or eat something less carbohydrate-rich'
                    print" You are consuming excessive food , please reduce your quantity or eat something less carbohydrate-rich \n";
                if (plan==1):
                    x='You are consuming excessive food , please reduce your quantity extensively.'
                    print "You are consuming excessive food , please reduce your quantity extensively \n";
                plt.axis([0, 4.0, 0, 300])
                plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/2), predbg,predbg-(1*spike/6),predbg-(2*spike/6), predbg-(3*spike/6),predbg-(4*spike/6),predbg-(5*spike/6),predbg-spike])
                plt.xlabel('Time(hr)')
                plt.ylabel('Blood Glucose level(mg/DL)')
                plt.title('Glycemic Excursions for the next 4 hours !')
                plt.savefig('static/foodpredictor.png')



         if(bglvl>=100 and bglvl<140):
            gi=total/n;
            if(tired == 1):
                bglvl-=30;  
            if (gi !=0):
                spike=((0.53*gi) + 67);
            else:
                spike=0;
            predbg=bglvl+spike;
            print predbg,"is your predicted blood glucose \n"
            print spike
            if (spike<=85):
                if (plan==1):
                    x='It is ok to eat.'
                    print "It is ok to eat \n"
                if (plan==2):
                    x='This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables.'
                    print "This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables \n"
                if (plan==3):
                    x='This food is insufficient to keep up with your activities, Please eat more food.'
                    print "This food is insufficient to keep up with your activities, Please eat more food \n"
        
                plt.axis([0, 4.0, 0, 300])
                plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/4), bglvl+(spike/2),bglvl+(3*spike/4),predbg, predbg-(spike/4),predbg-(spike/2),predbg-(3*spike/4),predbg-spike])
                plt.xlabel('Time(hr)')
                plt.ylabel('Blood Glucose level(mg/DL)')
                plt.title('Glycemic Excursions for the next 4 hours !')
                plt.savefig('static/foodpredictor.png')
            if (spike>85 and spike<=103):
                if (plan==2):
                    x='It is ok to eat.'
                    print "It is ok to eat \n"
                if (plan==3):
                    x='This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables .'
                    print "This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables \n"
                if (plan==1):
                    x='You are consuming excessive food , please reduce the quantities to keep up with your activities.'
                    print "You are consuming excessive food , please reduce the quantities to keep up with your activities. \n"
                
                plt.axis([0, 4.0, 0, 300])
                plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/3), bglvl+(2*spike/3),predbg,predbg-(spike/5), predbg-(2*spike/5),predbg-(3*spike/5),predbg-(4*spike/5),predbg-spike])
                plt.xlabel('Time(hr)')
                plt.ylabel('Blood Glucose level(mg/DL)')
                plt.title('Glycemic Excursions for the next 4 hours !')
                plt.savefig('static/foodpredictor.png')
            if (spike>103):
                if (plan==3):
                    x='It is ok to eat.'
                    print "It is ok to eat \n"
                if (plan==2):
                    x='You are consuming excessive food , please reduce your quantity or eat something less carbohydrate-rich.'
                    print" You are consuming excessive food , please reduce your quantity or eat something less carbohydrate-rich \n"
                if (plan==1):
                    x='You are consuming excessive food , please reduce your quantity extensively.'
                    print "You are consuming excessive food , please reduce your quantity extensively \n"
            
                plt.axis([0, 4.0, 0, 300])
                plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/2), predbg,predbg-(1*spike/6),predbg-(2*spike/6), predbg-(3*spike/6),predbg-(4*spike/6),predbg-(5*spike/6),predbg-spike])
                plt.xlabel('Time(hr)')
                plt.ylabel('Blood Glucose level(mg/DL)')
                plt.title('Glycemic Excursions for the next 4 hours !')
                plt.savefig('static/foodpredictor.png')



         if(bglvl>=140):
                gi=total/n;
                if(gi!=0):
                    spike=((0.8*gi) + 120);
                else:
                    spike=0;
                if(tired == 1):
                    bglvl-=40;  
                predbg=bglvl+spike;
                print predbg,"for uncontrolled diabetes\n";
                print spike
                if (spike<=147):
                    if (plan==1):
                        x='It is ok to eat.'
                        print "It is ok to eat \n"
                    if (plan == 2):
                        x='This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables.'
                        print "This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables \n"
                    if (plan == 3):
                        x='This food is insufficient to keep up with your activities, Please eat more food.'
                        print "This food is insufficient to keep up with your activities, Please eat more food \n"
                    plt.axis([0, 4.0, 0, 300])
                    plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/4), bglvl+(spike/2),bglvl+(3*spike/4),predbg, predbg-(spike/4),predbg-(spike/2),predbg-(3*spike/4),predbg-spike])   
                    plt.xlabel('Time(hr)')
                    plt.ylabel('Blood Glucose level(mg/DL)')
                    plt.title('Glycemic Excursions for the next 4 hours !')
                    plt.savefig('static/foodpredictor.png')
                if (spike > 147 and spike <= 174):
                    if (plan == 2):
                        x='It is ok to eat.'
                        print "It is ok to eat \n"
                    if (plan == 3):
                        x='This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables.'
                        print "This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables \n"
                    if (plan == 1):
                        x='You are consuming excessive food , please reduce the quantities to keep up with your activities.'
                        print "You are consuming excessive food , please reduce the quantities to keep up with your activities. \n"
                
                    plt.axis([0, 4.0, 0, 300])
                    plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/3), bglvl+(2*spike/3),predbg,predbg-(spike/5), predbg-(2*spike/5),predbg-(3*spike/5),predbg-(4*spike/5),predbg-spike])  
                    plt.xlabel('Time(hr)')
                    plt.ylabel('Blood Glucose level(mg/DL)')
                    plt.title('Glycemic Excursions for the next 4 hours !')
                    plt.savefig('static/foodpredictor.png')
                if (spike > 174):
                    if (plan == 3):
                        x='It is ok to eat.'
                        print "It is ok to eat \n"
                    if (plan == 2):
                        x='You are consuming excessive food , please reduce your quantity or eat something less carbohydrate-rich.'
                        print" You are consuming excessive food , please reduce your quantity or eat something less carbohydrate-rich \n"
                    if (plan == 1):
                        x='You are consuming excessive food , please reduce your quantity extensively.'
                        print "You are consuming excessive food , please reduce your quantity extensively \n"
                    plt.axis([0, 4.0, 0, 300])
                    plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/2), predbg,predbg-(1*spike/6),predbg-(2*spike/6), predbg-(3*spike/6),predbg-(4*spike/6),predbg-(5*spike/6),predbg-spike])   
                    plt.xlabel('Time(hr)')
                    plt.ylabel('Blood Glucose level(mg/DL)')
                    plt.title('Glycemic Excursions for the next 4 hours !')
                    plt.savefig('static/foodpredictor.png')

         return render_template('third.html', variable = predbg,variable1=x)
    return render_template('third.html', variable = predbg, variable1=x)


@app.route('/fourth', methods=['GET', 'POST'])
def fourth():
    return render_template('fourth.html',)

@app.route('/fifth', methods=['GET', 'POST'])
def fifth():
    if request.method == 'POST':
         glucose = request.form.get('glucose')
         bglvl=float(glucose)
         p = request.form.get('plan')
         t = request.form.get('tired')
         food1 = request.form.get('food1')
         q1 = request.form.get('q1')
         food2 = request.form.get('food2')
         q2 = request.form.get('q2')
         food3 = request.form.get('food3')
         q3 = request.form.get('q3')
         plan=int(p)
         tired=int(t)
         food1 = request.form.get('food1')
         q1 = request.form.get('q1')
         v1=0
         n=0
         v1=v1+int(food1)*int(q1)
         n=n+int(q1)

         food2 = request.form.get('food2')
         q2 = request.form.get('q2')
         v1=v1+int(food2)*int(q2)
         n=n+int(q2)

         food3 = request.form.get('food3')
         q3 = request.form.get('q3')
         v1=v1+int(food3)*int(q3)
         n=n+int(q3)
         total=v1

         if(bglvl<100):
            gi=total/n;
            if(gi!=0):
                spike=((0.38*gi) + 22);
            else:
                spike=0;
            if(tired == 1):
                bglvl-=20;
            predbg=bglvl+spike
            print predbg," is your predicted blood glucose \n"
            print spike
            if (spike<=35):
                if (plan==1):
                    print "It is ok to eat \n";
                    x='It is ok to eat'
                if (plan==2):
                    print "This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables \n";
                    x='This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables'
                if (plan==3):
                    print "This food is insufficient to keep up with your activities, Please eat more food \n";
                    x='This food is insufficient to keep up with your activities, Please eat more food'
                plt.axis([0, 4.0, 0, 300])
                plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/4), bglvl+(spike/2),bglvl+(3*spike/4),predbg, predbg-(spike/4),predbg-(spike/2),predbg-(3*spike/4),predbg-spike])
                plt.xlabel('Time(hr)')
                plt.ylabel('Blood Glucose level(mg/DL)')
                plt.title('Glycemic Excursions for the next 4 hours !')
                plt.savefig('static/foodpredictor.png')
            if (spike>35 and spike<=48):
                if (plan==2):
                    print "It is ok to eat \n";
                    x='It is ok to eat'
                if (plan==3):
                    print "This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables \n";
                    x='This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables'
                if (plan==1):
                    print "You are consuming excessive food , please reduce the quantities to keep up with your activities. \n";
                    x='You are consuming excessive food , please reduce the quantities to keep up with your activities.'
            
                plt.axis([0, 4.0, 0, 300])
                plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/3), bglvl+(2*spike/3),predbg,predbg-(spike/5), predbg-(2*spike/5),predbg-(3*spike/5),predbg-(4*spike/5),predbg-spike])
                plt.xlabel('Time(hr)')
                plt.ylabel('Blood Glucose level(mg/DL)')
                plt.title('Glycemic Excursions for the next 4 hours !')
                plt.savefig('static/foodpredictor.png')
            if (spike>48):
                if (plan==3):
                    print "It is ok to eat \n";
                    x='It is ok to eat'
                if (plan==2):
                    print" You are consuming excessive food , please reduce your quantity or eat something less carbohydrate-rich \n";
                    x='You are consuming excessive food , please reduce your quantity or eat something less carbohydrate-rich'
                if (plan==1):
                    print "You are consuming excessive food , please reduce your quantity extensively \n";
                    x='You are consuming excessive food , please reduce your quantity extensively'
                plt.axis([0, 4.0, 0, 300])
                plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/2), predbg,predbg-(1*spike/6),predbg-(2*spike/6), predbg-(3*spike/6),predbg-(4*spike/6),predbg-(5*spike/6),predbg-spike])
                plt.xlabel('Time(hr)')
                plt.ylabel('Blood Glucose level(mg/DL)')
                plt.title('Glycemic Excursions for the next 4 hours !')
                plt.savefig('static/foodpredictor.png')



         if(bglvl>=100 and bglvl<140):
            gi=total/n;
            if(tired == 1):
                bglvl-=30;  
            if (gi !=0):
                spike=((0.53*gi) + 67);
            else:
                spike=0;
            predbg=bglvl+spike;
            print predbg,"is your predicted blood glucose \n"
            print spike
            if (spike<=85):
                if (plan==1):
                    print "It is ok to eat \n"
                    x='It is ok to eat'
                if (plan==2):
                    print "This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables \n"
                    x='This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables'
                if (plan==3):
                    print "This food is insufficient to keep up with your activities, Please eat more food \n"
                    x='This food is insufficient to keep up with your activities, Please eat more food'
        
                plt.axis([0, 4.0, 0, 300])
                plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/4), bglvl+(spike/2),bglvl+(3*spike/4),predbg, predbg-(spike/4),predbg-(spike/2),predbg-(3*spike/4),predbg-spike])
                plt.xlabel('Time(hr)')
                plt.ylabel('Blood Glucose level(mg/DL)')
                plt.title('Glycemic Excursions for the next 4 hours !')
                plt.savefig('static/foodpredictor.png')
            if (spike>85 and spike<=103):
                if (plan==2):
                    print "It is ok to eat \n"
                    x='It is ok to eat'
                if (plan==3):
                    print "This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables \n"
                    x='This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables'
                if (plan==1):
                    print "You are consuming excessive food , please reduce the quantities to keep up with your activities. \n"
                    x='You are consuming excessive food , please reduce the quantities to keep up with your activities.'
                
                plt.axis([0, 4.0, 0, 300])
                plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/3), bglvl+(2*spike/3),predbg,predbg-(spike/5), predbg-(2*spike/5),predbg-(3*spike/5),predbg-(4*spike/5),predbg-spike])
                plt.xlabel('Time(hr)')
                plt.ylabel('Blood Glucose level(mg/DL)')
                plt.title('Glycemic Excursions for the next 4 hours !')
                plt.savefig('static/foodpredictor.png')
            if (spike>103):
                if (plan==3):
                    print "It is ok to eat \n"
                    x='It is ok to eat'
                if (plan==2):
                    print" You are consuming excessive food , please reduce your quantity or eat something less carbohydrate-rich \n"
                    x='You are consuming excessive food , please reduce your quantity or eat something less carbohydrate-rich '
                if (plan==1):
                    print "You are consuming excessive food , please reduce your quantity extensively \n"
                    x='You are consuming excessive food , please reduce your quantity extensively'
            
                plt.axis([0, 4.0, 0, 300])
                plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/2), predbg,predbg-(1*spike/6),predbg-(2*spike/6), predbg-(3*spike/6),predbg-(4*spike/6),predbg-(5*spike/6),predbg-spike])
                plt.xlabel('Time(hr)')
                plt.ylabel('Blood Glucose level(mg/DL)')
                plt.title('Glycemic Excursions for the next 4 hours !')
                plt.savefig('static/foodpredictor.png')



         if(bglvl>=140):
                gi=total/n;
                if(gi!=0):
                    spike=((0.8*gi) + 120);
                else:
                    spike=0;
                if(tired == 1):
                    bglvl-=40;  
                predbg=bglvl+spike;
                print predbg,"for uncontrolled diabetes\n";
                print spike
                if (spike<=147):
                    if (plan==1):
                        print "It is ok to eat \n"
                        x='It is ok to eat'
                    if (plan == 2):
                        print "This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables \n"
                        x='This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables'
                    if (plan == 3):
                        print "This food is insufficient to keep up with your activities, Please eat more food \n"
                        x='This food is insufficient to keep up with your activities, Please eat more food'
                    plt.axis([0, 4.0, 0, 300])
                    plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/4), bglvl+(spike/2),bglvl+(3*spike/4),predbg, predbg-(spike/4),predbg-(spike/2),predbg-(3*spike/4),predbg-spike])   
                    plt.xlabel('Time(hr)')
                    plt.ylabel('Blood Glucose level(mg/DL)')
                    plt.title('Glycemic Excursions for the next 4 hours !')
                    plt.savefig('static/foodpredictor.png')
                if (spike > 147 and spike <= 174):
                    if (plan == 2):
                        print "It is ok to eat \n"
                        x='It is ok to eat'
                    if (plan == 3):
                        print "This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables \n"
                        x='This food is insufficient to keep up with your activities , Please eat little more of less carbohydrate-rich food like sprouts,vegetables '

                    if (plan == 1):
                        print "You are consuming excessive food , please reduce the quantities to keep up with your activities. \n"
                        x='You are consuming excessive food , please reduce the quantities to keep up with your activities.'

                    plt.axis([0, 4.0, 0, 300])
                    plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/3), bglvl+(2*spike/3),predbg,predbg-(spike/5), predbg-(2*spike/5),predbg-(3*spike/5),predbg-(4*spike/5),predbg-spike])  
                    plt.xlabel('Time(hr)')
                    plt.ylabel('Blood Glucose level(mg/DL)')
                    plt.title('Glycemic Excursions for the next 4 hours !')
                    plt.savefig('static/foodpredictor.png')
                if (spike > 174):
                    if (plan == 3):
                        print "It is ok to eat \n"
                        x='It is ok to eat.'
                    if (plan == 2):
                        print" You are consuming excessive food , please reduce your quantity or eat something less carbohydrate-rich \n"
                        x='You are consuming excessive food , please reduce your quantity or eat something less carbohydrate-rich.'
                    if (plan == 1):
                        print "You are consuming excessive food , please reduce your quantity extensively \n"
                        x='You are consuming excessive food , please reduce your quantity extensively .'
                    plt.axis([0, 4.0, 0, 300])
                    plt.plot([0, 0.5, 1, 1.5,2.0,2.5,3.0,3.5,4.0], [bglvl, bglvl+(spike/2), predbg,predbg-(1*spike/6),predbg-(2*spike/6), predbg-(3*spike/6),predbg-(4*spike/6),predbg-(5*spike/6),predbg-spike])   
                    plt.xlabel('Time(hr)')
                    plt.ylabel('Blood Glucose level(mg/DL)')
                    plt.title('Glycemic Excursions for the next 4 hours !')
                    plt.savefig('static/foodpredictor.png')






         return render_template('third.html', variable = predbg, variable1=x)
       
    return render_template('third.html', variable = predbg,variable1=x)    


if __name__ == '__main__':
    app.run(debug=True)
