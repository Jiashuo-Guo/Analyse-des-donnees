import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
import scipy.cluster.vq
import codage as cd

def readfile(filename):
	with open(filename) as f:
		data = f.read().splitlines()

	return data

def acm(data,noms_individus,noms_variables):
    # print(data)
    # print(data.shape)
    # print(noms_individus)
    # print(noms_variables)
    
    nb_modalites_par_var = data.max(0)
    nb_modalites = int(nb_modalites_par_var.sum())
    
    XTDC = np.zeros((data.shape[0],nb_modalites))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            XTDC[i, int(nb_modalites_par_var[:j].sum() + data[i,j]-1)] = 1
            
    noms_modalites = []
    for i in range(data.shape[1]):
        for j in range(int(nb_modalites_par_var[i])):
            noms_modalites.append(noms_variables[i]+str(j+1))
    
    print(noms_modalites)
    
    
    Xfreq = XTDC / XTDC.sum()
    
    marge_colonne = Xfreq.sum(1).reshape(Xfreq.shape[0],1)
    # print(marge_colonne.shape)
    marge_ligne = Xfreq.sum(0).reshape(1,Xfreq.shape[1])
    # print(marge_ligne.shape)
    
    Xindep = marge_ligne * marge_colonne
    # print(Xindep.shape)
    
    
    X = Xfreq / Xindep - 1
    
    M = np.diag(marge_ligne[0,:])
    D = np.diag(marge_colonne[:,0])
    # print(M.shape)
    # print(D.shape)
    
    # Calcul de la matrice de covariance pour les modalités en ligne
    Xcov_ind = X.T.dot(D.dot(X.dot(M)))
    
    # Calcul des valeurs et vecteurs propres de la matrice de covariance
    L,U = np.linalg.eig(Xcov_ind)
    
    # Tri par ordre décroissant des valeurs des valeurs propres
    indices = np.argsort(L)[::-1]
    print(type(indices))
    val_p_ind = np.float16(np.sort(L)[::-1]) # car des valeurs peuvent être complexes avec une partie imaginaire nulle en raison d'approximations numériques
    vect_p_ind = U[:,indices]
    
    # Suppression des éventuelles valeurs propres nulles
    indices = np.nonzero(val_p_ind > 0)[0]
    print(type(indices))
    val_p_ind = val_p_ind[indices]
    vect_p_ind = vect_p_ind[:,indices]
    
    # Calcul des facteurs pour les modalités en ligne 
    fact_ind = X.dot(M.dot(vect_p_ind))
    
    # Calcul des facteurs pour les modalités en colonne
    fact_mod = X.T.dot(D.dot(fact_ind)) / np.sqrt(val_p_ind)
    
    # Calcul des pourcentage d'inertie des axes factoriels
    inerties = 100*val_p_ind / val_p_ind.sum()
    
    print('Pourcentages d"inertie :')
    print(inerties)
    
    # Affichage du diagramme d'inertie
    plt.figure(1)
    plt.plot(inerties,'o-')
    plt.title('Diagramme des inerties')
    
    
    # Calcul de la contribution des individus
    contributions_ind = np.zeros(fact_ind.shape)
    for i in range(fact_ind.shape[1]):
        f = fact_ind[:,i]
        contributions_ind[:,i] = 100 * D.dot(f*f) / f.T.dot(D.dot(f))
        
    print('Contribution des individus :')
    print(contributions_ind)

    # Calcul de la contribution des modalités 
    contributions_mod = np.zeros(fact_mod.shape)
    for i in range(fact_mod.shape[1]):
        f = fact_mod[:,i]
        contributions_mod[:,i] = 100 * M.dot(f*f) / f.T.dot(M.dot(f))
        
    print('Contribution des modalités :')
    print(contributions_mod)    
    
    # Calcul de la qualité de représentation des modalités en ligne
    distance = (fact_ind**2).sum(1)
    distance = distance.reshape(fact_ind.shape[0],1)
    
    qualite_ind = fact_ind**2 / distance
    
    print('Qualité de représentation des individus :')
    print(qualite_ind)

    # Calcul de la qualité de représentation des modalités
    distance = (fact_mod**2).sum(1)
    distance = distance.reshape(fact_mod.shape[0],1)
    
    qualite_mod = fact_mod**2 / distance
    
    print('Qualité de représentation des modalités :')
    print(qualite_mod)
    
    # Affichage du plan factoriel pour les indidivus
    plt.figure(2)
    plt.plot(fact_ind[:,0],fact_ind[:,1],'x')
    plt.grid(True)
    plt.axvline(linewidth=0.5,color='k')
    plt.axhline(linewidth=0.5,color='k')
    plt.title('ACM Projection des individus')
    for label,x,y in zip(noms_individus,fact_ind[:,0],fact_ind[:,1]):
        plt.annotate(label,
                      xy=(x,y),
                      xytext=(-50,5),
                      textcoords="offset points",
                      arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0')
                      )
    
    # Affichage du plan factoriel pour les modalités en colonne
    plt.figure(3)
    plt.plot(fact_mod[:,0],fact_mod[:,1],'x')
    plt.grid(True)
    plt.axvline(linewidth=0.5,color='k')
    plt.axhline(linewidth=0.5,color='k')
    plt.title('ACM Projection des modalités')
    for label,x,y in zip(noms_modalites,fact_mod[:,0],fact_mod[:,1]):
        plt.annotate(label,
                     xy=(x,y),
                     xytext=(-50,5),
                     textcoords="offset points",
                     arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0')
                     )
    plt.show()

def cah(X,noms_lignes,noms_colonnes):
	Z = linkage(X, 'ward')

	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('sample index')
	plt.ylabel('distance')
	dendrogram(Z,leaf_rotation=90.,leaf_font_size=8.,)

	k=3
	clusters = fcluster(Z, k, criterion='maxclust')
	print(clusters)

	plt.figure(figsize=(10, 8))
	plt.scatter(X[:,0], X[:,1], c=clusters)

	for label,x,y in zip(noms_lignes,X[:,0],X[:,1]):
		plt.annotate(label,
			xy=(x,y),
			xytext=(-50,5),
			textcoords='offset points',
			ha='right', va='bottom',
			arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
			)
		
	plt.title('Classification ascendante hierarchique')
	plt.show()

if __name__ == '__main__':
    data = np.loadtxt('population_donnees.txt')
    data = cd.normalisation(data)
    nom_m1 = readfile('population_noms_individus.txt')
    nom_m2 = readfile('population_noms_variables.txt')
    r = cd.quantitatif_en_qualitatif2(data,5)
    acm(r,nom_m1,nom_m2)
    cah(r,nom_m1,nom_m2)


