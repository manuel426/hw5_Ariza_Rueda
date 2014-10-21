import numpy as np 
import scipy
import matplotlib.pyplot as plt
from pprint import pprint 
from scipy.fftpack import fft2, fftfreq, ifft, fftshift
from  math import sqrt,log 

# Se cargan los datos de los circulos con x y y cada coordenada
data = np.loadtxt('circulos.dat')
x = data[:,0]
y = data[:,1]

# Se hace una imagen en forma de matriz para relalizarle fft 
tam = int(len(x)/20.0)
imagen = np.zeros((tam,tam))
n = len(x)
for i in range (n):
    imagen[int(x[i]/6.0*tam), int(y[i]/6.0)*tam]=+1 

# Se efectua la fft a la imagen 
fft_ = fft2(imagen)
fft_ = fftshift(fft_*fft_.conjugate())
h = 6.0/tam
freq_ = fftshift(fftfreq(tam,h))

# Se efectua la correlacion
ro = tam/2
repe = np.zeros ( ro+1 )
fft_radio = np.zeros( ro+1, dtype=np.dtype(np.complex_) )
for i in range(tam):
    for j in range(tam):
        radio = int(sqrt( (i-ro)**2 + (j-ro)**2 ))
        if ( radio <= ro ):
            fft_radio[radio]=+fft_[i,j]
            repe[radio]=+1

# Se grafica la tranformada para los datos del radio 
fft_radio = fft_radio/ repe
plt.plot(fft_radio)
plt.show()

# Se seleccionan los  maximos del espectro para obtener el radio 
maximos = []
radios = [] 

for i in range(len(fft_radio)):
    d = fft_radio[i]
    if(d>fft_radio[i-1]):
        if(d>fft_radio[i+1]):
            radios.append(d)
            maximos.append(i)

# Se calcula el delta de frecuencia asociado al radio deseado 
delta = [] 
for i in range (len(maximos)-1):
    delta.append( maximos[i+1]-maximos[i])

maxfrecuencias = []
for i in range(len(freq_)):
    if(freq_[i]>=0):
        maxfrecuencias.append(freq_[i])
hm = (maxfrecuencias[1]-maxfrecuencias[0])
promedio= sum (delta)/len(delta)
fdiametro = hm*promedio
tdiametro = 1.0/fdiametro
radio = tdiametro/2.0

pprint('El radio de los circulos es :'+ str(radio))


