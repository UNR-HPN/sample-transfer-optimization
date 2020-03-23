#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: hem
"""

import numpy as np
import pandas as pd
import math
from read_data import ReadData
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import os


class DNNModel:
    def __init__(self, file_name, type_to_run, run_removed):
        print(file_name)
        self.type_to_run = type_to_run
        a = ReadData(file_name)
        self.fl = a.fl
        self.is_freq = False
        self.run_removed = run_removed
        self.clf_dir = "clfs"
        if self.run_removed:
            a.total_1_std(0.5)
            self.clf_dir = "removed_clfs"
        self.logs = a.run_code()
    def accuracy(self, expected, actual):
        return 100*math.fabs(expected-actual)/actual
    def make_percentage_for_acc(self, logs_passed, number_of_input_layer, classifier=False):
        start_from = 5
        X_actual = {}
        X = []
        y = []
        default_value = 0
        for log in logs_passed:
            average = log[1]
            
            prev = log[2]
            new_log = [log[0]]
            new_y = [log[0]]
            success = True
            actual_log = log[start_from+1:min(len(log), start_from+1+15)]
            actual_throughputs = []
            for i in range(start_from, number_of_input_layer+start_from+1):
                if len(log) <= i:
                    new_log.append(default_value)
                    continue
                if i in [start_from, start_from+1] and (prev == 0 or log[i] ==0):
                    actual_throughputs.append(log[i])
                    prev = log[i]
                else:
                    if prev == 0:
                        new_log.append(default_value)
                    else:
                        perc = log[i] - prev
                        if(perc/(prev+1.5) >1000):
                            pass
                        new_log.append(perc/(prev+1.5))
                        actual_throughputs.append(log[i])
                        prev = log[i]
            if success:
                X.append(new_log)
                X_actual[log[0]] = actual_log
                new_y.append(average)
                
                average_this = np.mean(np.array(actual_throughputs))
                
                average_percentage = 100*(average-average_this)/average
                if not classifier:
                    new_y += [average_percentage]
                else:
                    new_y += self.get_output_category(average_percentage)
                y.append(new_y)
        return pd.DataFrame(X), X_actual, pd.DataFrame(y)
    def get_output_category(self, value):
        output = [0]*31
        if value>15:
            value=15
        if value<-15:
            value=-15
        value = int(value+0.5)
        index = value+15
        output[index] = 1
        return output
    def make_percentage(self, logs_passed, number_of_input_layer):
        start_from = 2
        X_actual = {}
        X = []
        y = []
        default_value = 0
        for log in logs_passed:
            average = log[1]
            optimal_time = log[2]
            prev = log[start_from+1]/2
            new_log = [log[0]]
            new_y = [log[0]]
            success = True
            actual_log = log[start_from+1:min(len(log), start_from+1+15)]
            for i in range(start_from+1, number_of_input_layer+start_from+1):
                if len(log) <= i:
                    new_log.append(default_value)
                    continue
                if i in [start_from, start_from+1] and (prev == 0 or log[i] ==0):
                    prev = log[i]
                else:
                    if prev == 0:
                        new_log.append(default_value)
                    else:
                        perc = log[i] - prev
                        if prev <= 1:
                            prev += 1.5
                        new_log.append(perc/(prev))
                        prev = log[i]
            if success:
                X.append(new_log)
                X_actual[log[0]] = actual_log
                new_y.append(average)
                optimal_classify = [0]*13
                try:
                    optimal_classify[optimal_time-3] = 1
                except:
                    optimal_classify[-1] = 1
                new_y += optimal_classify
                y.append(new_y)
        return pd.DataFrame(X), X_actual, pd.DataFrame(y)
    def create_clfs(self):
        start_at = 3
        run_till = 16
        type_to_run = self.type_to_run
        random_id = 42
        all_clfs = {}
        for i in range(start_at, run_till):
            number_of_input_layer = i
            joblib_url = './'+self.clf_dir+'/'+type_to_run+"-10-"+str(number_of_input_layer)+"-"+str(random_id)+"-percentage-optimal.pkl"
            if os.path.exists(joblib_url):
                print("[+] File already created.")
                continue
            self.solvers = [10]
            from_ = 1
            X, X_actual, y = self.make_percentage(self.logs, number_of_input_layer)
            
            hidden_layer_sizes = tuple([100 for i in range(3)])
            
            X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.30, random_state=random_id)
            y_train_optimal = y_train.loc[:,2:]
            X_train = X_train.loc[:,1:]
            X_test = X_test.loc[:,1:]
            X_test_actual = []
            
            data = {'hidden_layer_length': len(hidden_layer_sizes), 
                    'hidden_layer_sizes': hidden_layer_sizes, 
                    "from": from_, "till": number_of_input_layer, "solver": "lbfgs", 'alpha':0.1, "max_iter": 1000,
                    "activation": "relu", "learning_rate": 0.001, "learning_rate_init": 0.001,
                    "train_data": len(X_train), "test_data": len(X)-len(X_train), "is_random": 1}

            clf = MLPClassifier(solver=data["solver"], alpha=data['alpha'], 
                                   hidden_layer_sizes=data["hidden_layer_sizes"], max_iter=data["max_iter"], activation=data["activation"], 
                                   learning_rate_init=data["learning_rate_init"])
            clf.fit(X_train, y_train_optimal)
        
            y_pred = clf.predict_proba(X_test)
            if not number_of_input_layer in all_clfs:
                all_clfs[number_of_input_layer] = {}
            all_clfs[number_of_input_layer]["clf"] = clf
            all_clfs[number_of_input_layer]["y_pred"] = y_pred
            all_clfs[number_of_input_layer]["X_actual"] = X_test_actual
            if not os.path.exists(joblib_url):
                joblib.dump(clf, joblib_url)
            print("CLF for i="+str(number_of_input_layer)+" is done.")
        print("CLfs are created.")
        
    def run_models(self):
        y = None
        start_at = 3
        run_till = 16
        type_to_run = self.type_to_run
        random_id = 42
        X = None
        X_actual = None
        test_timestamp = None
        self.X_check = None
        self.y_test = None
        y_train = None
        X_test, X_train = None, None
        all_clfs = {}
        
        for i in range(start_at, run_till):
            number_of_input_layer = i
            joblib_url = './'+self.clf_dir+'/'+type_to_run+"-10-"+str(number_of_input_layer)+"-"+str(random_id)+"-percentage-optimal.pkl"
            self.solvers = [10]
            X, X_actual, y = self.make_percentage(self.logs, number_of_input_layer)
            X_train, X_test, y_train , self.y_test = train_test_split(X, y, test_size=0.30, random_state=random_id)
            y_test_optimal = self.y_test.loc[:,:]
            X_train = X_train.loc[:,1:]
            self.X_check = self.y_test.loc[:,0:0]
            test_timestamp = X_test.loc[:,0:0].values.ravel()
            X_test = X_test.loc[:,1:]
            X_test_actual = {}
            
            
            for time_stamp in test_timestamp:
                X_test_actual[time_stamp] = np.array(X_actual[time_stamp][max(i-4, 0):i+1]).mean()
            if not number_of_input_layer in all_clfs:
                all_clfs[number_of_input_layer] = {}
            all_clfs[number_of_input_layer]["clf"] = joblib.load(joblib_url)
            all_clfs[number_of_input_layer]["y_pred"] = all_clfs[number_of_input_layer]["clf"].predict_proba(X_test)
            all_clfs[number_of_input_layer]["y_test_optimal"] = y_test_optimal
            all_clfs[number_of_input_layer]["X_actual"] = X_test_actual
            
        self.all_clfs = all_clfs
    def test_two(self, predicted, actual):
        actual = list(actual)
        predicted = list(predicted)
        max_actual = max(actual)
        actual_1 = actual.index(max_actual)
        max_predicted = max(predicted)
        predicted_1 = predicted.index(max_predicted)
        return actual_1 - predicted_1, max_predicted, predicted_1
    def run_test(self, perc, decreaser=0.05):
        start_at = 3
        run_till = 16
        type_to_run = self.type_to_run
        time_acc = {}
        type_to_run = self.type_to_run
        threshhold = {}
        for i in range(start_at, run_till+1):
            threshhold[i] = perc - decreaser*(i-3)
        ind = 0
        solutions_ = {}
        total = 0
        total_1 = 0
        accs = []
        all_clfs = self.all_clfs
        for i in range(-16, 16):
            solutions_[i] = 0
        
        solutions_for_index = set([])
        max_prediction = {}
        convergence_time = {}
        actual_convergence = {}
        time_difference = {}
        for i in range(0, run_till+1):
            time_difference[i] = 0
        for i in range(start_at, run_till+1):
            convergence_time[i] = 0
            actual_convergence[i] = 0
        
        
        
        for number_of_input in all_clfs:
            if number_of_input > 10:
                pass
            ind = 0
            indexes = self.y_test.index.values.tolist()
            for index in indexes:
                row = all_clfs[number_of_input]["y_test_optimal"].loc[index:index,:]
                timestamp = row.loc[index,0]
                
                actual_solutions = row.loc[index, 2:].values.ravel()
                average_throughput = row.loc[index, 1]
                predicted_throguhput = all_clfs[number_of_input]["X_actual"][timestamp]
                
                average_throughput = average_throughput if average_throughput != 0 else 1
                sol, max_predicted, predicted_time = self.test_two(all_clfs[number_of_input]["y_pred"][ind], actual_solutions)
                acc = abs((average_throughput-predicted_throguhput)/average_throughput)
                if ind in max_prediction:
                    if max_prediction[ind][0] < max_predicted:
                        max_prediction[ind] = [max_predicted, acc, predicted_time+3, sol, all_clfs[number_of_input]["y_pred"][ind]]
                else:
                    max_prediction[ind] = [max_predicted, acc, predicted_time+3, sol, all_clfs[number_of_input]["y_pred"][ind]]
                
                if max_predicted > threshhold[number_of_input] and ind not in solutions_for_index:
                    if predicted_time+3<=number_of_input+1:
                        accs.append(acc)
                        solutions_for_index.add(ind)
                        solutions_[sol] += 1
                        total += 1
                        
                        convergence_time[number_of_input+1] += 1
                        actual_convergence[predicted_time+3] += 1
                    time_acc[timestamp] = acc
                ind += 1
        for i in max_prediction:
            if i not in solutions_for_index:
                accs.append(max_prediction[i][1])
                convergence_time[15 if type_to_run != "freq" else 20] += 1
                actual_convergence[max_prediction[i][2]] += 1
                solutions_[max_prediction[i][3]] += 1
        total_1 = len(self.y_test)
        accs = np.array(accs)
        print(actual_convergence)
        print(convergence_time)
        sum_time = 0
        sum_can_be_time = 0
        for i in convergence_time:
            sum_time += (i*convergence_time[i])
            sum_can_be_time += (i*actual_convergence[i])
#        print("For type_to_run = "+type_to_run+", Perc = " +str(round(perc, 2)) + " the ER is "\
#              +str(round(100*accs[~np.isnan(accs)].mean(), 8)) + " % and CT is "\
#              +str(round(sum_time*1.0/total_1, 8))+" s and can be CT is "+str(round(sum_can_be_time*1.0/total_1, 2))+"s "+" for perc="+str(perc)+" and decreaser="+str(decreaser))
        return accs[~np.isnan(accs)].mean(), sum_can_be_time*1.0/total_1


def run_for_all_data(to_points, acc):
    data = {
        "esnet": ["esnet_regular.csv"], 
        "xsede": ["xsede_regular.csv"], 
        "pronghorn": ["pronghorn_regular.csv"], 
        "ph_freq": ["pronghorn_frequency.csv"],
        "esnet_freq": ["esnet_frequency.csv"],
        "dtns_freq" : ["dtns_frequency.csv"], 
        "dtns": ["dtns_regular.csv"]
        }
    for i in data:
        for j in data[i]:
            print("For data %s" % j)
            network = "dtns"
            if "esnet" in j:
                network = "esnet"
            elif "xsede" in j:
                network = "xsede"
            elif "pronghorn" in j:
                network = "pronghorn"
            a = DNNModel("./data/%s/%s" % (network, j), i, False)
            a.create_clfs()
            divide_by = 1
            if "freq" in j:
                divide_by = 10
            a.run_models()
            accs, ct = a.run_test(0.7, 0.05)
            print("##Error rate is %f" % (100*accs))
            print("##Average Convergence time is %f" % (ct/divide_by))
            print("##Total transfers after removing %d"%len(a.logs))
run_for_all_data(15, 10)








