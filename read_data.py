#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: hem
"""
import numpy as np

class ReadData:
    def __init__(self, file_path):
        self.add_zero = False
        self.read_file(file_path)
    def read_file(self, file_path):
        self.fl = {}
        with open(file_path, 'r') as file_:
            index_id = 0
            reader = file_
            line = reader.readline()
            header = self.parse_header(line)
            line = reader.readline()
            while(line):
                self.fl[index_id] = self.parse_line(line.strip(), header)
                index_id += 1
                line = reader.readline()
        return self.fl
    def total_1_std(self, greater_than):
        h = []
        files = []
        self.remove_files = True
        to_be_used = []
        for i in self.fl:
            if np.isnan(self.fl[i]["mean"]):
                self.fl[i]["removed"] = 1
                
            if "removed" in self.fl[i] and self.fl[i]["removed"]:
                continue
            value = self.fl[i]
            h.append((value["std"]*1.0)/value["mean"])
            mean_over_std = (1.0 * value["std"])/value["mean"]
            if (self.remove_files and mean_over_std > greater_than) or value["mean"] == 0.0 or value["mean"] <= 0.1:
                files.append(i)
                value["removed"] = 1
                
            elif "removed" not in value:
                to_be_used.append(i)
                value["removed"] = 0
        return files
    def parse_header(self, line):
        header = {"time": 0}
        fields = line.strip().split(",")
        for i in range(len(fields)):
            if fields[i] in ['t%d'%j for j in range(1, 21)]:
                header["time"] = i
            elif fields[i] == "mean_throughput":
                header["mean"] = i
            elif fields[i] == "stdv_throughput":
                header["std"] = i
            else:
                header[fields[i]] = i
        return header
    def parse_line(self, line, header):
        data = {"logs": [], "removed": 0, "model_2": {}}
        time_index = 1
        if(self.add_zero):
            data["logs"].append([time_index, 0, 0.])
            time_index += 1
        fields = line.split(",")
        for i in range(len(fields)):
            if i <= header["time"]:
                if fields[i]:
                    data["logs"].append([time_index, 0, float(fields[i])])
                else:
                    data["logs"].append([time_index, 0, 0.])
                if i == 1:
                    if data["logs"][time_index-1][2] == data["logs"][time_index-2][2]:
                        data["logs"][i-1][2] -= 1
                time_index += 1
            elif i == header["mean"]:
                data["mean"] = float(fields[i])
            elif i == header["std"]:
                data["std"] = float(fields[i])
            elif "optimal_point" in header and i == header["optimal_point"]:
                data["model_2"]["required_seconds"] = int(fields[i])
            elif "optimal_error" in header and i == header["optimal_error"]:
                data["model_2"]["error_rate"] = float(fields[i])
        return data
    def run_code(self):
        logs = []
        till = 10
        for i in sorted(list(self.fl.keys())):
            if not self.fl[i]["removed"]:
                isNan = False
                thpts = np.array(self.fl[i]["logs"])[:30,2]
                for t in thpts:
                    if np.isnan(np.array(t)):
                        isNan = True
                if not isNan and len(thpts)>(till+1):
                    try:
                        logs.append([i, self.fl[i]["mean"], int(float(self.fl[i]["model_2"]["required_seconds"]))]+list(thpts))
                    except:
                        print("here", self.fl[i])
        return logs


            