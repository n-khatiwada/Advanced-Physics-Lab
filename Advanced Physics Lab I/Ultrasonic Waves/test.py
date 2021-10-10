import pandas as pd
import codecs
import matplotlib.pyplot as plt
import numpy as np
import odf
import math
import pdfkit
from uncertainties import ufloat, ufloat_fromstr
from sklearn.linear_model import LinearRegression
from lmfit.models import LorentzianModel
from IPython.display import display, Latex
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

X = 'Frequency [$kHz$]'
Y= 'Amplitude [$mV$]'

f = codecs.open('data11.txt').read().replace('\t', ',')
i = open('new.txt', "w")
i.write(f)
i.close()
p1 = pd.read_csv('new.txt', names=[X, Y])

