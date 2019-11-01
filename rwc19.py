"""Ce package contient toutes les implémentations abstraites des méthodes utilisées
   pour la prédiction des scores de la coupe du monde de Rugby 2019 (Rugby World Cup 2019).
"""

# Importation des librairies nécessaires
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


class dataManager:
    """Cette classe permet de d'instancier des objets capables d'effectuer certaines tâches
       de feature engineering.
    
    Returns:
        dataManager object -- Objet de maniplation de tables de données
    """

    def __init__(self, dataframe):
        """Constructeur de la classe qui prend en entrée un dataframe de la classe pandas.
           Il s'agit d'un objet de type "pandas.core.frame.DataFrame"
        
        Arguments:
            dataframe {pandas dataframe} -- Table de données initiale pour créer 
            des features.
        """
        # Vérifier que la table donnée est bien un dataframe pandas
        assert type(dataframe) == pd.core.frame.DataFrame

        #Vérifier que le dataframe est une matrice pleine
        assert dataframe.isna().sum().sum() == 0
        
        #créer un attribut data pour le stocker
        self.data = dataframe


        

    def generate_date_features(self, column):
        """Cette méthode génère des colonnes supplémentaires basées sur une colonne de
           type date. Quatre nouvelles colonnes sont ajoutées au dataframe initial :
           * year : L'année extraite de la colonne date
           * month : Le numéro de mois extrait de la colonne date (dans [|1,12|])
           * dayOfYear : Le ième jour de l'année extrait de la colonne date (dans [|1,365|])
           * dayOfWeek : Le ième jour de la semane extrait de la colonne date (dans [|1,7|])
        
        Arguments:
            column {str} -- Nom de la colonne date dans le daframe initial
        
        Returns:
            str -- Message sur le succès de l'opération
        """
        
        # Table intermédiaire de données
        df  = self.data.copy()

        # Transformation des colonnes en type datetime
        df[column] = pd.to_datetime(df[column])

        # Création des features
        df['year']      = df[column].dt.year
        df['month']     = df[column].dt.month
        df['day0fYear'] = df[column].dt.dayofyear
        df['dayOfWeek'] = df[column].dt.dayofweek

        #Réaffectation des données
        self.data = df
        
        return "4 features successfully generated"
    
class MultiOutputRF(object):
    """Classe permettant de créer un classifieur RandamForest à sortie double.
       Ce classifieur est bien différent d'un double RandomForest implémenté indépendamment l'un de l'autre qui
       ignore un quelconque lien possible entre ces deux.
       Le MultiOutputRandomForest implémenté dans cette classe tient compte de la corrélation existante
       entre les différentes sorties, ce qui est très bien adapté dans le cas d'un match de sport où l'on
       a besoin de prédire simultanément les deux scores des participants.

       Voir : http://astrohackweek.org/blog/multi-output-random-forests.html

    
    Arguments:
        object {params} -- Des paramètres pour créer un objet random Forest de scikit-learn.
    
    Returns:
        Object -- Objet de type MultiRandomForest
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def fit(self, X, Y):
        X, Y = map(np.atleast_2d, (X, Y))
        assert X.shape[0] == Y.shape[0]
        Ny = Y.shape[1]
        
        self.clfs = []
        for i in range(Ny):
            clf = RandomForestRegressor(*self.args, **self.kwargs)
            Xi = np.hstack([X, Y[:, :i]])
            yi = Y[:, i]
            self.clfs.append(clf.fit(Xi, yi))
            
        return self
        
    def predict(self, X):
        Y = np.empty([X.shape[0], len(self.clfs)])
        for i, clf in enumerate(self.clfs):
            Y[:, i] = clf.predict(np.hstack([X, Y[:, :i]]))
        return Y


def get_loss_win_accuracy(y_test, y_pred):
    """ Cette fonction calcule à quel point l'algorithme est capable de prédire la victoire ou
        la défaite de l'équipe à domicile. Elle retourne un score compris entre de O et 1.
        Plus le score est élevé, plus l'algo est performant.
    
    Arguments:
        y_test {dataframe} -- dataframe de deux colonnes contenant les vrais scores des matchs
        y_pred {dataframe} -- dataframe de deux colonnes contenant les scores prédits des matchs

        Rq: Les deux dataframes doivent avoir comme noms des colonnes "For" et "Aga".
        "For" désigne le score de l'équipe à domicile et "Aga" le score de l'équipe à l'extérieur.
    
    Returns:
        float -- Pourcentage de (victoires/defaites) bien prédits
    """

    win_loss_true = pd.Series(np.where(y_test.For > y_test.Aga,1,0))
    win_loss_pred = pd.Series(np.where(y_pred.For > y_pred.Aga,1,0))

    return (win_loss_true == win_loss_pred).mean()

def norme2(y_test, y_pred):
    """Calcule le RMSE en dimension 2 de deux vecteurs en entrée.
    
    Arguments:
        y_test {dataframe} -- dataframe de deux colonnes contenant les vrais scores des matchs
        y_pred {dataframe} -- dataframe de deux colonnes contenant les scores prédits des matchs

        Rq: Les deux dataframes doivent avoir comme noms des colonnes "For" et "Aga".
        "For" désigne le score de l'équipe à domicile et "Aga" le score de l'équipe à l'extérieur.
    
    Returns:
        float -- La racine de l'erreur quadratique moyenne des deux vecteurs en entrée.
    """

    x1 = np.array(y_test.For)
    x2 = np.array(y_pred.For)
    y1 = np.array(y_test.Aga)
    y2 = np.array(y_pred.Aga)

    return np.sqrt(sum((x1-x2)**2 + (y1-y2)**2))


def fit_predict_new(data1, data2, n_estimators=2000, first_is_train=True, eval=False, cross_val=False):
    """Fonction générale pour faire de l'apprentissage et de la prédiction sur les données
       de rugby. Ces données sont généralement issues de la classe dataManager définit plus haut.
    
    Arguments:
        data1 {dataframe} -- Premier dataframe qui serait utilisé pour l'apprentissage
        data2 {dataframe} -- Dataframe de test (ou de validation)
    
    Keyword Arguments:
        n_estimators {int} -- Le nombre d'arbres à créer dans le RandomForest (default: {2000})
        first_is_train {bool} -- Indication que le 1er dataframe est pour entrainer le modèle (default: {True})
        eval {bool} -- S'il faut faire de l'évaluation grace au RMSE et la précision (default: {False})
        cross_val {bool} -- S'il faut faire de la validation croisée avec K=5 (default: {False})
    """

    data1['train_test_part'] = 'train'
    data2['train_test_part'] = 'test'

    final_data    = pd.concat([data1, data2], ignore_index=True, sort=False)
    final_data_cp = final_data.copy()
    drop_cols = ['Match Date', 'Result', 'Diff', 'HTf', 'HTa', 'Ground', 'train_test_part'] 
    
    try:
        final_data.drop(drop_cols, axis=1, inplace=True)
    except:
        print("Problème pour supprimer certaines colonnes : \
        Les noms d'une ou plusieurs colonnes sont mal spécifiées")
    
    last_full_X = pd.get_dummies(final_data)
    last_full_X['train_test_part'] = final_data_cp.train_test_part
    y_cols = ['For', 'Aga']
    X, y = last_full_X.drop(y_cols, axis=1), last_full_X[['For', 'Aga', 'train_test_part']]

    if cross_val:
        kf = KFold(n_splits=5,random_state=42)
    
        #Cross validation
        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]
            
            #drop
            X_train = X_train.drop('train_test_part', axis=1)
            X_test  = X_test.drop('train_test_part', axis=1)
            y_train = y_train.drop('train_test_part', axis=1)
            y_test  = y_test.drop('train_test_part', axis=1)
            
            #fit and predict
            clf = MultiOutputRF(n_estimators).fit(X_train, y_train)
            Ypred = clf.predict(X_test)

            #change y_pred to the right format
            y_pred_df = pd.DataFrame(Ypred.astype(int), columns=["For", 'Aga'])
            
            #eval
            print("Precision Loss/Win : ", get_loss_win_accuracy(y_test.copy(), y_pred_df.copy()))
            print("RMSE pour la norme 2 : ", norme2(y_test.copy(), y_pred_df.copy()))
            print("----------------------------------------------------------------------------------")

    else:
        X_train = X[X.train_test_part=='train']
        X_test  = X[X.train_test_part =='test']
        y_train = y[y.train_test_part=='train']
        y_test  = y[y.train_test_part=='test']

        X_train = X_train.drop('train_test_part', axis=1)
        X_test  = X_test.drop('train_test_part', axis=1)
        y_train = y_train.drop('train_test_part', axis=1)
        y_test  = y_test.drop('train_test_part', axis=1)

        #fit and predict
        clf = MultiOutputRF(n_estimators).fit(X_train, y_train)
        Ypred = clf.predict(X_test)

        #change y_pred to the right format
        y_pred_df = pd.DataFrame(Ypred.astype(int), columns=["For", 'Aga'])

        #eval
        if eval:
            print("Precision Loss/Win : ", get_loss_win_accuracy(y_test.copy(), y_pred_df.copy()))
            print("RMSE pour la norme 2 : ", norme2(y_test.copy(), y_pred_df.copy()))
            print("----------------------------------------------------------------------------------")

        y_pred_df['home'] = data2.Team
        y_pred_df['away'] = data2.Opposition

        return y_pred_df[['home', 'For', 'Aga', 'away']]
