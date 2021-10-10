#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import codecs
import matplotlib.pyplot as plt
import numpy as np
import odf
import math
import pdfkit
from itertools import chain
from statistics import mean
from uncertainties import ufloat, ufloat_fromstr
from sklearn.linear_model import LinearRegression
from lmfit.models import LorentzianModel
from IPython.display import display, Latex
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

X = "Time [$s$]"
Y = "Voltage [$V$]"
ref = 1.63


# # ### Monitoring of Laser Stability

# # In[53]:


# f = codecs.open('1_HeNe.txt').read().replace('\t', ',')
# i = open('new.txt', "w")
# i.write(f)
# i.close()
# HeNe = pd.read_csv('new.txt', skiprows=5, usecols=[0, 1], names=[X, Y])


# # In[59]:


# plt.figure(figsize=(20,5))
# plt.plot(HeNe[X], HeNe[Y])
# plt.xlabel(X)
# plt.ylabel(Y)
# plt.grid()
# plt.show()


# # In[60]:


# f = codecs.open('1_green.txt').read().replace('\t', ',')
# i = open('new.txt', "w")
# i.write(f)
# i.close()
# p1green = pd.read_csv('new.txt', skiprows=5, usecols=[0, 1], names=[X, Y])


# # In[62]:


# plt.figure(figsize=(20,5))
# plt.plot(p1green[X], p1green[Y])
# plt.xlabel(X)
# plt.ylabel(Y)
# plt.grid()
# plt.show()


# In[64]:


f = codecs.open('1_background.txt').read().replace('\t', ',')
i = open('new.txt', "w")
i.write(f)
i.close()
background = pd.read_csv('new.txt', skiprows=5, usecols=[0, 1], names=[X, Y])


# ### Polarization direction of green laser and Malus Law

# In[70]:


X = "𝜃"
Y = "Voltage [$V$]"
Z = "Normalized Voltage [$V$]"
BackOff = background[Y].mean()
angles = np.array([i for i in range(-90, 91, 10)])
df = pd.DataFrame(data=angles, columns=[X])


# In[73]:


f = codecs.open('2_-90to90.txt').read().replace('\t', ',')
i = open('new.txt', "w")
i.write(f)
i.close()
p2a = pd.read_csv('new.txt', skiprows=5, usecols=[1], names=[Y])
p2a = pd.concat([df, p2a], axis=1)
p2a


# In[74]:


p2a[Z] = p2a[Y] - np.min(p2a[Y])
A = np.array(p2a[X]).reshape(-1, 1)
B = np.array(p2a[Z]).reshape(-1, 1)
p2a.style.set_caption("Photodiode Signal when illuminated by green laser")


# In[79]:


def linelikeexcel(x,y):
    coefs = np.polyfit(x,y,deg=8)
    p_obj = np.poly1d(coefs)
    return p_obj

p_objt = linelikeexcel(np.array(df[X]), np.array(p2a[Z]))

x_line = np.linspace(min(np.array(df[X])), max(np.array(df[X])), 100)
y_linet = p_objt(x_line)

plt.plot(np.array(df[X]), np.array(p2a[Z]), 'o')
plt.plot(x_line,y_linet, 'r--')
plt.xlabel(X)
plt.ylabel(Z)
plt.title('Normalized Photodiode Signal vs Orientation angle of the Analyzer in degrees.')

plt.grid()
plt.show()


# # In[9]:


# C = np.rad2deg((np.cos(A))**2)


# # In[ ]:


# plt.plot(C, B)
# plt.grid()
# plt.show()


# # In[11]:





# # ### Fresnel's reflection coefficients and Brewester angle

# # In[39]:


# Brewster_angle = np.rad2deg(np.arctan(ref))
# np.format_float_positional(Brewster_angle, precision=4, unique=False, fractional=False, trim='k')


# # In[44]:


# angles = np.array([i for i in range(15, 52, 5)] + [i for i in range(53, 64)] + [i for i in range(65, 90, 5)])
# df = pd.DataFrame(data=angles, columns=[X])


# # #### For s-polarization

# # In[44]:


# primarys = 0
# def calc_rsp(inci):
#     r_sp = - ((np.sqrt(ref**2 - (np.sin(np.deg2rad(inci)))**2) - np.sqrt(1 - (np.sin(np.deg2rad(inci)))**2))**2)/(ref**2 - 1)
#     return r_sp

# display(Latex('At $90^\circ$, r = {}'.format(calc_rsp(90))))
# display(Latex('At $0^\circ$, r = {}'.format('%.4f' %calc_rsp(0))))
# theoreticalr = np.array([calc_rsp(i) for i in angles])
# st = pd.DataFrame(data=theoreticalr, columns=['rt'])


# # In[37]:


# f = codecs.open('p3s.txt').read().replace('\t', ',')
# i = open('new.txt', "w")
# i.write(f)
# i.close()
# p3s1 = pd.read_csv('new.txt', skiprows=5, usecols=[2], names=['Ir'])


# # In[38]:


# p3s = pd.concat([df, p3s1], axis=0)
# p3s['r'] = np.sqrt(p3s['Ir']/primarys)


# # In[ ]:


# def linelikeexcel(x,y):
#     coefs = np.polyfit(x,y,deg=8)
#     p_obj = np.poly1d(coefs)
#     return p_obj

# p_objs = linelikeexcel(df[X], p3s['r'])
# p_objst = linelikeexcel(df[X], st['rt'])

# x_line = np.linspace(min(x), max(x), 100)
# y_lines = p_objs(x_line)
# y_linest = p_objst(x_line)

# plt.plot(df[X], p3s['r'], df[X], st['rt'], 'o')
# plt.plot(x_line,y_lines, x_line, y_linest, 'r--')

# plt.grid()
# plt.show()


# # #### For p-polarization

# # In[9]:


# primaryp = 0
# def calc_rpp(inci):
#     r_pp = (((ref**2)*(np.sqrt(1 - (np.sin(np.deg2rad(inci)))**2))) - (np.sqrt(ref**2 - (np.sin(np.deg2rad(inci)))**2)))/((ref**2)*(np.sqrt(1 - (np.sin(np.deg2rad(inci)))**2)) + (np.sqrt(ref**2 - (np.sin(np.deg2rad(inci)))**2)))
#     return r_pp

# display(Latex('At $90^\circ$, r = {}'.format(calc_rpp(90))))
# display(Latex('At $0^\circ$, r = {}'.format('%.4f' %calc_rpp(0))))
# theoreticalr = np.array([calc_rpp(i) for i in angles])
# pt = pd.DataFrame(data=theoreticalr, columns=['rt'])


# # In[ ]:


# f = codecs.open('p3p.txt').read().replace('\t', ',')
# i = open('new.txt', "w")
# i.write(f)
# i.close()
# p3p1 = pd.read_csv('new.txt', skiprows=5, usecols=[2], names=['Ir'])


# # In[ ]:


# p3p = pd.concat([df, p3p1], axis=0)
# p3p['r'] = np.sqrt(p3p['Ir']/primaryp)


# # In[ ]:


# def linelikeexcel(x,y):
#     coefs = np.polyfit(x,y,deg=8)
#     p_obj = np.poly1d(coefs)
#     return p_obj

# p_objp = linelikeexcel(df[X], p3p['r'])
# p_objpt = linelikeexcel(df[X], pt['rt'])

# x_line = np.linspace(min(x), max(x), 100)
# y_linep = p_objp(x_line)
# y_linept = p_objpt(x_line)

# plt.plot(df[X], p3p['r'], df[X], pt['rt'], 'o')
# plt.plot(x_line,y_linep, x_line, y_linespt, 'r--')

# plt.grid()
# plt.show()


# # ### Brewster angle and rotation of polarization vector

# # In[33]:


# delta = 45
# dic = {'a': [[15, 16], [20, 25]]}
# angle = pd.DataFrame(dic)
# angle['Average'] = [mean(list(chain.from_iterable(x))) for x in angle.values.tolist()]
# angle['diff'] = angle['Average'] - delta
# angle


# # In[45]:


# def brewster(inci):
#     ang = np.rad2deg(np.arctan(-(np.sqrt((1-(np.sin(np.deg2rad(inci)))**2) * (ref**2 - (np.sin(np.deg2rad(inci)))**2)))/(np.sin(np.deg2rad(inci)))**2))
#     return ang
# theoreticalang = np.array([brewster(df[X])])
# calculatedang = np.array([brewster(angle['diff'])])


# # In[ ]:


# def linelikeexcel(x,y):
#     coefs = np.polyfit(x,y,deg=8)
#     p_obj = np.poly1d(coefs)
#     return p_obj

# p_objt = linelikeexcel(df[X], theoreticalang)
# p_objc = linelikeexcel(df[X], calculatedang)

# x_line = np.linspace(min(x), max(x), 100)
# y_linet = p_objsp(x_line)
# y_linec = p_objpp(x_line)

# plt.plot(df[X], theoreticalang, df[X], calculatedang, 'o')
# plt.plot(x_line,y_linet, x_line, y_linec, 'r--')

# plt.grid()
# plt.show()


# # In[ ]:


# #if this does not work, use scipy.optimize.curvefit()


# # ### Polarization by quarterwave and halfwave plates

# # In[ ]:


# X = 'Voltage'
# f = codecs.open('0.txt').read().replace('\t', ',')
# i = open('new.txt', "w")
# i.write(f)
# i.close()
# p0 = pd.read_csv('new.txt', skiprows=5, usecols=[2], names=[X])


# f = codecs.open('30.txt').read().replace('\t', ',')
# i = open('new.txt', "w")
# i.write(f)
# i.close()
# p30 = pd.read_csv('new.txt', skiprows=5, usecols=[2], names=[X])

# f = codecs.open('45.txt').read().replace('\t', ',')
# i = open('new.txt', "w")
# i.write(f)
# i.close()
# p45 = pd.read_csv('new.txt', skiprows=5, usecols=[2], names=[X])

# f = codecs.open('60.txt').read().replace('\t', ',')
# i = open('new.txt', "w")
# i.write(f)
# i.close()
# p60 = pd.read_csv('new.txt', skiprows=5, usecols=[2], names=[X])

# f = codecs.open('90.txt').read().replace('\t', ',')
# i = open('new.txt', "w")
# i.write(f)
# i.close()
# p90 = pd.read_csv('new.txt', skiprows=5, usecols=[2], names=[X])

# p4 = pd.concat([p0, p30, p45, p60, p90], axis=0)


# # In[47]:


# angles = np.array([i for i in range(-90, 91, 10)])
# angles


# # In[ ]:


# plt.plot(angles, p0[X], angles, p30[X], angles, p45[X], angles, p60[X], angles, p90[X])
# plt.show()


# # In[ ]:





# # In[ ]:


# def linelikeexcel(x,y):
#     coefs = np.polyfit(x,y,deg=8)
#     p_obj = np.poly1d(coefs)
#     return p_obj

# p_objsp = linelikeexcel(p3ds[X], p3ds[Y])
# p_objpp = linelikeexcel(p3dp[X], p3dp[Y])

# x_line = np.linspace(min(x), max(x), 100)
# y_linesp = p_objsp(x_line)
# y_linepp = p_objpp(x_line)

# plt.plot(p3ds[X], p3ds[Y], p3ds[X], p3dp[Y], 'o')
# plt.plot(x_line,y_linesp, x_line, y_linepp, 'r--')

# plt.grid()
# plt.show()


# In[ ]:




