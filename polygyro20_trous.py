'''
PRECISION DU PERCEMENT D'UN TUNNEL

Ce code assez simple permet de réaliser une tâche qui semble d'abord très
complexe. Le traitement rigoureux d'un problème comportant de nombreuses
observations et conditions ne pose aucune difficulté particulière, malgré
la taille importante des matrices.

Exemple
Pour une polygonale à 50 côtés avec un azimut gyroscopique mesuré après
chaque section de 5 côtés, on peut poser 9 conditions (6e, 11e, ..., 46e
côté). Dans ce cas, le problème consiste à appliquer 9 conditions à 59
observations, dont 50 sont corrélées entre elles (les gisements
indirects).

Pour boucher les trous "???" du code, le plus simple est de suivre les
explications données au tableau lors du cours.

-------------------------------------------------------------------------

Polygonale lancée à N côtés, avec détermination gyroscopique de l'azimut
géographique à la fin de chaque section de n côtés. Pour n > N, on ne
considère aucune mesure gyroscopique.

Hypothèse 1: portail parfait
On néglige l'imprécision de la position à l'entrée du tunnel (écart-type
latéral), ainsi que celle du gisement de l'orientation initiale vers un
point extérieur.

Hypothèse 2: calibration parfaite du gyroscope
Grâce aux coordonnées approchées des points, on peut calculer la valeur
exacte de la convergence du méridien pour chaque côté et convertir les
azimuts géographiques en gisements sans perte de précision. Bref, c'est
comme si l'on mesurait directement des gisements avec le gyroscope.

Idée conductrice: le gisement d'un côté mesuré directement avec le
gyroscope doit correspondre au gisement obtenu indirectement par
cheminement polygonal.

1. On calcule les gisements de la polygonale lancée par propagation
   des mesures angulaires, supposées indépendantes. Toutefois, les
   gisements sont corrélés puisqu'une mesure angulaire influence tous les
   gisements suivants. En fin de compte, c'est comme si l'on avait
   observé des gisements corrélés, que l'on peut traiter comme des
   mesures indirectes.

2. Les mesures gyroscopiques sont considérées comme des mesures
   directes, indépendantes entre elles. Chaque mesure gyroscopique donne
   lieu à une condition: les gisements directs et indirectes doivent
   être égaux.

A partir des gisements, on calcule l'écart transversal par propagation.
Pour cette préanalyse, on ne dispose pas de mesures, mais seulement de
leur disposition (polygonale tendue, côtés égaux) et de leur écart-type a
priori. En appliquant la propagation de variance, on obtient l'écart-type
de l'écart transversal. Le procédé est identique, quelle que soit la
corrélation entre les gisements.

En l'absence de mesures gyroscopiques, ou avant la compensation, donc en
ne considérant que les gisements indirects obtenus à partir des mesures
angulaires, on obtient la même valeur qu'avec la formule du cours établie
pour la polygonale lancée.

Si on applique la propagation de variance après compensation, l'opération
est identique, mais avec d'autres variances et covariances pour les
gisements (matrice N*N). A ce stade, on bénéficie de l'apport des mesures
gyroscopiques.

En faisant varier la densité des mesures gyroscopiques, donc en modifiant
n, on peut chercher l'optimum entre le coût et la précision. C'est tout
l'intérêt d'une préanalyse!

-------------------------------------------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)


# données

#N = int(abs(np.floor(int(input('nombre de côtés de la polygonale tendue: ')))))
N = 20

#s = int(input('longueur d''un côté en [m]: '))
s = 5

#sigmalfa = int(input('écart-type d''une mesure d''angle en [mgon]: '))
sigmalfa = 200

r = 0

#n = int(input('nombre de côtés par section, entre deux mesures gyroscopiques: '))
n = 5

if n>0:

	r = int(np.floor(N/n-0.1))
    
   # Si N est un multiple de n, soustraire 0.1 fait passer à l'entier
   # inférieur. Ainsi on exclut une mesure gyro à l'extrémité du
   # cheminement, qui serait inutile.
	
	if r>0:
		#sigmagyro = int(input('écart-type d''une mesure d''azimut en [mgon]: '))
		sigmagyro = 1

'''
-------------------------------------------------------------------------

D'abord on calcule le cheminement sans mesures gyroscopiques. Donc il
s'agit d'une polygonale lancée. On réalise une propagation de variance,
et non une compensation. On en tire les valeurs maximales des écarts-types
pour fixer l'échelle des diagrammes. Ainsi, l'effet des mesures
gyroscopiques sera facile à visualiser.

Chaque gisement dépend de la somme des mesures angulaires précédentes.
Il faut construire la matrice de propagation F telle que: dfi=F*dalfa.
C'est une matrice N*N triangulaire inférieure avec des 1.

'''

F = np.zeros((N,N))  # réserver un espace contigu
for i in range(N):
   for j in range(N):
	   if i >= j:
		   F[i,j] = 1


# matrice de covariance des gisements de la polygonale lancée
# implicitement: Qalfa = np.identity(N)

Kfi = F @ F.transpose()*sigmalfa*sigmalfa   # en [mgon^2]

# écart-type de chaque gisement, en [mgon]

sigmafi = np.sqrt(np.diag(Kfi))

maxsigmafi = np.max(sigmafi)

# Propagation des erreurs de gisements sur l'erreur transversale le long du
# cheminement. On choisit un repère local tel que l'axe des x soit
# parallèle au cheminement, ainsi y est transversal.

# s en [m] et KFI converti en [mrad^2], idem [mm] et [rad^2]

KFI = Kfi * np.pi * 2 / 400
aux = F @ KFI @ F.T
Kyy = (s**2) * aux

sigmay = np.sqrt(np.diag(Kyy))

maxsigmay = np.max(sigmay)

# Les lignes suivantes construisent un dessin. La syntaxe Python ne
# fait pas partie du cours "Méthodes d'estimation". Si nécessaire,
# consultez la documentation. Surtout: conservez ce code pour l'adapter à
# d'autres usages dans d'autres cours!

#----------------

x = np.arange(0,(N+1)*s/1000,s/1000)
y = np.append(0,sigmafi)
yerr = np.append(0,sigmay)
y_0  = np.zeros(N+1)


#plt.subplot(2, 1, 1)
#plt.plot(x,y,'r',label='Sans gyroscope')
#plt.xlabel('')
#plt.ylabel('gisement [mgon]')
#plt.title('écarts-types')
#plt.legend()

#plt.subplot(2, 1, 2)
#plt.errorbar(x,y_0,yerr=yerr,fmt='r',ecolor='r',label='Sans gyroscope')
#plt.xlabel('longueur [km]')
#plt.ylabel('latéral [mm]')

#plt.legend()
#plt.show()

#----------------

# en cas de mesures gyroscopiques

if r>0:

	# Aux observations indirectes des gisements, il faut ajouter les
	# observations directes (via les azimuts gyroscopiques). Le vecteur des
	# observations passe de N à N+r composantes, et l'on construit une matrice
	# auxiliaire Kll de dimension (N+r)*(N+r).

	Kll = np.zeros((N+r, N+r))      # réserver un espace contigu
	Kll[:N,:N] = Kfi # gisements propagés
	a = np.zeros(r)+sigmagyro*sigmagyro
	print(a)
	Kll[N:,N:] = np.diag(a) # azimuths gyro

	# Expression des conditions, une par mesure gyroscopique. Ces conditions
	# sont linéaires, on peut construire directement B de dimension r*(N+r).

	B = np.zeros((r,N+r))    # réserver un espace contigu
	for i in range(r):
		B[i, i*n] = 1
		B[i, N+i] = -1

		
	# matrice de covariance des gisements compensés
	aux = np.linalg.inv(B @ Kll @ B.T)
	Klcomp = Kll - Kll @ B.T @ aux @ B @ Kll

	# Après compensation, les azimuths gyro sont égaux aux gisements propagés
	# correspondants et leurs variances aussi. Donc toute l'information
	# nécessaire est contenue dans la partie supérieure gauche de la matrice.

	Kficomp = Klcomp[0:N,0:N]

	# écart-type de chaque gisement, en [mgon]

	sigmaficomp = np.sqrt(np.diag(Kficomp))

	np.set_printoptions(precision=2, suppress=True)
	print("sigmaficomp =")
	print(sigmaficomp)

	# propagation des erreurs des gisements compensés sur l'erreur transversale
	# le long du cheminement
	KFIcomp = Kficomp * np.pi * 2 / 400

	Kyycomp = (s ** 2) * F @ KFIcomp @ F.T

	sigmaycomp = np.sqrt(np.diag(Kyycomp))

	np.set_printoptions(precision=2, suppress=True)
	print("sigmaycomp =")
	print(sigmaycomp)

	# compléments des graphiques et remplacement du titre
	# ------------------------------------------------------

	y_gyro = np.append(0,sigmaficomp)
	yerr_gyro = np.append(0,sigmaycomp)


	plt.subplot(2, 1, 1)
	plt.plot(x,y,'r',label='Sans gyroscope')
	plt.plot(x,y_gyro,'b',label='Avec gyroscope')
	plt.xlabel('')
	plt.ylabel('gisement [mgon]')
	plt.title('écarts-types')
	plt.legend()

	plt.subplot(2, 1, 2)
	plt.errorbar(x,y_0,yerr=yerr,fmt='r',ecolor='r',label='Sans gyroscope')
	plt.errorbar(x, y_0, yerr=yerr_gyro, fmt='b', ecolor='b', label='Avec gyroscope')
	#plt.plot(x, y_0 + yerr_gyro, color='b', label='Avec gyroscope')
	#plt.plot(x, y_0 - yerr_gyro, color='b')
	#plt.errorbar(x,y_0,yerr=yerr_gyro,fmt='b',ecolor='b',label='Avec gyroscope')
	plt.xlabel('longueur [km]')
	plt.ylabel('latéral [mm]')
	plt.legend()
	plt.show()

	#-------------------------------------------------------

# et voilà le travail !
