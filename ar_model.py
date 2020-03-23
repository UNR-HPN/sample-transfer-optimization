#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: hem
"""
import numpy as np
import pandas as pd
import math
from read_data import ReadData

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.ar_model import AR

class ARModel:
    def __init__(self, file_name):
        print(file_name)
        a = ReadData(file_name)
        self.fl = a.fl
        self.is_freq = False
        self.remove_files = True
        a.total_1_std(0.5)
    def accuracy(self, expected, actual):
        return 100*math.fabs(expected-actual)/actual
    def model_run(self, to_points, accuracy_bound, model_name="arima"):
        accs = {}
        self.removed = 0
        acc = []
        self.tt_values = -10000
        r_squared_calculation = {to_points+1: 0}
        for j in self.fl:
            s_p = to_points+1
            if self.fl[j]['removed'] or min(to_points, len(self.fl[j]["logs"])) < 3:
                self.removed += 1
                self.fl[j]["stopping_point"] = s_p
                continue
            logs = np.array(self.fl[j]['logs'])
            is_conve = False
            lgs = logs[:,[0,2]]
            df = pd.DataFrame({'1': lgs[:,0], '2': lgs[:,1]})
            for i in range(3, min(to_points, len(self.fl[j]["logs"])-5)):
                mat = df['2'].as_matrix()[:i]
                
                predictions = []
                if model_name == "arima":
                    model = ARIMA(mat, order=(0, 1, 0))
                    model_fit = model.fit(disp=0)
                    predictions.append(model_fit.predict(i, i, typ="levels")[0])
                elif model_name == "ar":
                    model = AR(mat)
                    start_params = [0, 0, 1]
                    try:
                        model_fit = model.fit(maxlag=1, start_params=start_params, disp=-1)
                        predictions.append(model_fit.predict(i, i)[0])
                    except:
                        pass
                elif model_name == "adaptive":
                    predictions.append(self.fl[j]["logs"][i-1,2])
                elif model_name == "arma":
                    model = ARMA(mat, order=(0, 0))
                    model_fit = model.fit()
                    predictions.append(model_fit.predict(i, i)[0])
                else:
                    break
                
                if len(predictions) and self.accuracy(predictions[0], lgs[i,1]) < accuracy_bound:
                    ac1 = self.accuracy(np.mean(predictions), self.fl[j]["mean"]) 
                    is_conve = True
                    if i not in r_squared_calculation:
                        r_squared_calculation[i] = 0
                    r_squared_calculation[i] += 1
                    if not np.isnan(ac1):
                        if i not in accs:
                            accs[i] = []
                        accs[i].append(ac1)
                        acc.append(ac1)
                    s_p = i
                    break
                elif i + 1 == min(to_points, len(self.fl[j]["logs"])-5):
                    if len(predictions)==0:
                        predictions.append(lgs[i-1,1])
                    ac1 = self.accuracy(np.mean(predictions), self.fl[j]["mean"])
                    is_conve = True
                    if i not in r_squared_calculation:
                        r_squared_calculation[i] = 0
                    r_squared_calculation[i] += 1
                    if not np.isnan(ac1):
                        if i not in accs:
                            accs[i] = []
                        accs[i].append(ac1)
                        acc.append(ac1)
                    s_p = i
                    break
            if not is_conve:
                r_squared_calculation[to_points+1] += 1
            self.fl[j]["stopping_point"] = s_p
        x = sorted(list(r_squared_calculation.keys()))
        total = sum(list(r_squared_calculation.values()))
        y = []
        prev = 0.0
        for i in x:
            prev += r_squared_calculation[i]
            if not prev:
                y.append(0)
            else:
                y.append(prev/total)
        relative_ = {}
        for i in range(len(x)):
            relative_[x[i]] = round(y[i]*100, 2)
            
        if self.is_freq>0:
            for i in r_squared_calculation.keys():
                val = r_squared_calculation.pop(i)
                r_squared_calculation[i/10.] = val
            for i in relative_.keys():
                val = relative_.pop(i)
                relative_[i/10.] = val
        
        for i in accs.keys():
            if not accs[i]:
                accs.pop(i)
            else:
                accs[i] = np.mean(accs[i])
        self.r_squared_calculation = r_squared_calculation
        self.relative_ = relative_
        self.rr_accuracy = np.mean(acc)
        self.i_val_acc = accs
        print("##Error rate is %.5f" % self.rr_accuracy)


def run_for_all_data(to_points, acc):
    data = {
        "esnet": ["esnet_regular2.csv", "esnet_regular.csv"], 
        "xsede": ["xsede_regular.csv", "xsede_regular2.csv"], 
        "pronghorn": ["pronghorn_regular2.csv", "pronghorn_regular.csv"], 
        "freq": ["pronghorn_frequency.csv", "pronghorn_frequency2.csv", "esnet_frequency.csv", "esnet_frequency2.csv", "dtns_frequency.csv", "dtns_frequency2.csv"], 
        "dtns": ["dtns_regular2.csv", "dtns_regular.csv"]
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
            divide_by = 1
            if "freq" in j:
                divide_by = 10
            a = ARModel("./data/%s/%s" % (network, j))
            a.model_run(15, 10, model_name="ar")
            times = []
            count = 0
            for i in a.fl:
                if not a.fl[i]["removed"]:
                    times.append(a.fl[i]["stopping_point"])
                    count += 1
            print("##Average Convergence time is %.5f" % (np.mean(times)/divide_by))
            print("##Total transfers after removing %d" % count)
run_for_all_data(15, 10)
