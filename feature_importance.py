import pandas as pd
import xgboost as xgb
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore") 

def get_FI(stock):

    dataset_total_df_ = pd.read_csv("./data/features"+stock+".csv")
    dataset_total_df_ = dataset_total_df_.drop(['Date'],axis=1)
    print('Total dataset has {} samples, and {} features.'.format(dataset_total_df_.shape[0], \
                                                                dataset_total_df_.shape[1]))

    def get_feature_importance_data(data_income):
        data = data_income.copy()
        y = data['Close']      
        X = data.iloc[:, 1:]    
        
        train_samples = int(X.shape[0] * 0.65)
        
        X_train = X.iloc[:train_samples]
        X_test = X.iloc[train_samples:]
        y_train = y.iloc[:train_samples]
        y_test = y.iloc[train_samples:]
    
        return (X_train, y_train),(X_test, y_test)

    print("computing feature importance")
    (X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_total_df_)
    regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,
                                colsample_bytree=0.5, colsample_bylevel=0.5, colsample_bynode=0.5,learning_rate=0.05,
                                objective ='reg:squarederror')

    xgbModel = regressor.fit(X_train_FI, y_train_FI, 
                            eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
                            verbose=False)

    eval_result = regressor.evals_result()  
    training_rounds = range(len(eval_result['validation_0']['rmse']))  

    plt.scatter(x=training_rounds, y=eval_result['validation_0']['rmse'], label='Training Error')
    plt.scatter(x=training_rounds, y=eval_result['validation_1']['rmse'], label='Validation Error')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Training Vs Validation Error')
    plt.legend()
    plt.savefig('./data/FI/'+stock+'_err.png', format='png', dpi=400)

    fig = plt.figure(figsize=(8,8))
    plt.xticks(rotation='vertical')
    plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(), tick_label=X_test_FI.columns)
    plt.title('Figure 6: Feature importance of the technical indicators.')
    plt.savefig('./data/FI/'+stock+'.png', format='png', dpi=400)