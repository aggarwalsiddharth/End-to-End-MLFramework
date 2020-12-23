from sklearn import ensemble

MODELS = {

    'randomforest' : ensemble.RandomForestClassifier(n_estimators=200,verbose=2,n_jobs=-1),
    'extratrees' : ensemble.ExtraTreesClassifier(n_estimators=200,verbose=2,n_jobs=-1),
    'gradientboost': ensemble.GradientBoostingClassifier(n_estimators=200,verbose=2,)

}