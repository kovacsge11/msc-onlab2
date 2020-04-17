import sys
import numpy as np


f = open(sys.argv[1],"r")
f.readline()
f.readline()

s_vec = []
s_vec_line = f.readline()
s_vec_line = s_vec_line.lstrip().rstrip()
s_vec_line = s_vec_line[16:-1]
s_vec = s_vec_line.split()
for i in range(len(s_vec)):
    s_vec[i] = float(s_vec[i]) 

t_vec = []
t_vec_line = f.readline()
t_vec_line = t_vec_line.lstrip().rstrip()
t_vec_line = t_vec_line[1:-2]
t_vec = t_vec_line.split()
for i in range(len(t_vec)):
    t_vec[i] = float(t_vec[i])

#Normalizing s_vec and t_vec
s_max = s_vec[-1]
t_max = t_vec[-1]
s_min = s_vec[0]
t_min = t_vec[0]
for i in range(len(s_vec)):
    s_vec[i] = (s_vec[i]-s_min)/(s_max-s_min)
for i in range(len(t_vec)):
    t_vec[i] = (t_vec[i]-t_min)/(t_max-t_min)

tsize=len(t_vec)-4
ssize=len(s_vec)-4
cps = np.zeros((tsize,ssize*3))
for j in range(ssize):
    for i in range(tsize):
        line = f.readline()
        line = line.lstrip().rstrip()
        if j==0 and i==0: line = line[16:-1]
        elif j==ssize-1 and i==tsize-1: line = line[1:-4]
        elif i == tsize-1: line = line[1:-2]
        elif i == 0: line = line[2:-1]
        else: line = line[1:-1]
        line = line.split()
        cps[tsize-i-1,(ssize-j-1)*3] = float(line[0])
        cps[tsize-i-1,(ssize-j-1)*3+1] = float(line[1])
        cps[tsize-i-1,(ssize-j-1)*3+2] = float(line[2])

f.close()

tspf = open(sys.argv[2],"w")
tspf.write("{} {}\n".format(tsize*ssize,tsize+1))
for i in range(tsize):
    for j in range(ssize):
        tspf.write("{} {} {}\n".format(cps[i,j*3], cps[i,j*3+1], cps[i,j*3+2]))
        tspf.write("{} {} {} {} {}\n".format(s_vec[j], s_vec[j+1],s_vec[j+2],s_vec[j+3],s_vec[j+4]))
        tspf.write("{} {} {} {} {}\n".format(t_vec[i], t_vec[i+1],t_vec[i+2],t_vec[i+3],t_vec[i+4]))
        tspf.write("1\n")
        tspf.write("{}\n".format(j))

for i in range(tsize+1):
    tspf.write("{}\n".format(i*ssize))

tspf.close()
