import pandas as pd
import numpy as np

class custom_accuracy():
    def GetPrediction(model,X,full_data,labels):
        X_prob=model.predict(X)
        classes = labels.columns
        Xdf=pd.DataFrame(X_prob,index=X.index,columns=classes)
        def n_largest(c):
            large = c.nlargest(6)
            return pd.DataFrame([large.index.tolist()+large.values.tolist()]).loc[0]
        df=Xdf.apply(lambda c: n_largest(c),axis=1)
        df.columns = ['Prediction1','Prediction2','Prediction3','Prediction4','Prediction5','Prediction6','Confidence1','Confidence2','Confidence3','Confidence4','Confidence5','Confidence6']
        df['next_index']=full_data.loc[X.index.values]['next_index']
        return df
    def suppresed_top_accuracy(preds):
        preds['Predictions'] = preds['Prediction1'] + ':' + preds['Prediction2'] + ':' + preds['Prediction3'] + ':' + preds['Prediction4'] + ':' + preds['Prediction5'] + ':' + preds['Prediction6']
        preds['Predictions'] = preds['Predictions'].str.split(':').apply(lambda x: list(filter(lambda y: y not in ['General Inquiries','Other','Cancel Card/Account'], x)))
        preds['top_3'] = preds['Predictions'].str[:3]
        counts_3 = []
        counts_1 = []
        for each in preds.index:
            actual = preds['actual_action_list'].iloc[each]
            top_3_preds = preds.top_3.iloc[each]
            counts_3.append(bool(set(actual) & set(top_3_preds)))
            counts_1.append(top_3_preds[0] in actual)
        return(np.array(counts_3), np.array(counts_1))
