exp =lambda x: np.power(1.1, x)
normal = lambda x: np.power(x, 1)
squered = lambda x: np.power(x, 2)
qubed = lambda x: np.power(x, 3)
sqrt = lambda x: np.power(x, 0.5)
one_over = lambda x: np.power(x+0.01, -1)
log = lambda x: np.log(x + 0.01)


functions = [("normal", normal),("squered", squered),("qubed", qubed),("sqrt", sqrt),("one_over", one_over),("log", log),("exp", exp)]

features = [(fname, df_boston[fname].to_numpy()) for fname in df_boston if fname != "MEDV"]
lr = hw1linreg.LinearRegressor()

y =df_boston["MEDV"].to_numpy()
accuries = []
for fun1_name, fun1 in functions:
    for feature1, data1 in features:
        for fun2_name, fun2 in functions + [("1", lambda x: 1)]:
            for feature2, data2 in features:
                X = (fun1(data1)) * (fun2(data2))
                X = X.reshape(-1, 1)
                lr.fit(X, y)
                y_pred = lr.predict(X)
                accuries.append((hw1linreg.mse_score(y, y_pred), fun1_name, feature1, fun2_name, feature2))
                
accuries.sort(key=lambda t: t[0])
for mse, fun1_name, feature1, fun2_name, feature2 in list(filter(lambda t: t[2] != "RM" and t[4] != "RM", accuries))[:100]:
    print(mse, fun1_name, feature1, fun2_name, feature2)

