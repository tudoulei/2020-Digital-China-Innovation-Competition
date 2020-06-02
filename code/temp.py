# # y ############################## 不确定percent
    # percent=1000
    # all_data["y-1000"] = np.rint(all_data["y"]/percent)
    # w2c_feature(all_data,w2c_col="y-1000",vec_len=100,feature_path=feature_path)
    # # x ##############################不确定percent
    # percent=1000
    # all_data["x-1000"] = np.rint(all_data["x"]/percent)
    # w2c_feature(all_data,w2c_col="x-1000",vec_len=100,feature_path=feature_path)


    # # # d ##############################绝对没问题
    # w2c_feature(all_data,w2c_col="d",vec_len=100,feature_path=feature_path)
    # w2c_feature(all_data,w2c_col="x-y",vec_len=32,feature_path=feature_path,mode="min")
    # w2c_feature(all_data,w2c_col="x-y",vec_len=32,feature_path=feature_path,mode="max")
    # # label-v ############################## 百分之百确定100
    all_data['label'] = all_data.apply(lambda x: generalID(x['x'], x['y'],20,20,LON1,LON2,LAT1,LAT2), axis = 1)
    print("do")
    all_data['label-v'] = all_data['label'].astype("str")+"-"+all_data["v"].astype(int).astype("str")
    w2c_feature(all_data,w2c_col="label-v",vec_len=32,feature_path=feature_path,mode="std")
    
    LON1 = np.min(all_data.x)-1
    LON2 = np.max(all_data.x)+1
    LAT1 = np.min(all_data.y)-1
    LAT2 = np.max(all_data.y)+1
    all_data['label'] = all_data.apply(lambda x: generalID(x['x'], x['y'],20,20,LON1,LON2,LAT1,LAT2), axis = 1)
    all_data['label-v'] = all_data['label'].astype("str")+"-"+np.rint(all_data["v"]).astype("str")
    w2c_feature(all_data,w2c_col="label-v",vec_len=100,feature_path=feature_path)
    # # date ##############################没问题
    w2c_feature(all_data,w2c_col="date",vec_len=100,feature_path=feature_path)
    # label-date############################## 不知道label
    # LON1 = np.min(all_data.x)-1
    # LON2 = np.max(all_data.x)+1
    # LAT1 = np.min(all_data.y)-1
    # LAT2 = np.max(all_data.y)+1
    # all_data['label'] = all_data.apply(lambda x: generalID(x['x'], x['y'],100,100,LON1,LON2,LAT1,LAT2), axis = 1)
    # all_data['label-date'] = all_data['label'].astype(str)+"-"+all_data['date'].astype(str)
    # w2c_feature(all_data,w2c_col="label-date",vec_len=100,feature_path=feature_path)






    ###############################################################
    # percent = 100000
    # w2c_col ="x"
    # all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]/percent).astype("str")
    # w2c_col="y"
    # all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]/percent).astype("str")
    # all_data["x-y-100000"] = all_data["x_gai"]+"-"+all_data["y_gai"]
    # w2c_feature(all_data,w2c_col="x-y-100000",vec_len=32,feature_path=feature_path)
    
    # percent = 1000
    # w2c_col ="x"
    # all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]/percent).astype("str")
    # w2c_col="y"
    # all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]/percent).astype("str")
    # all_data["x-y-1000"] = all_data["x_gai"]+"-"+all_data["y_gai"]
    # w2c_feature(all_data,w2c_col="x-y-1000",vec_len=32,feature_path=feature_path,mode="std")
    # w2c_feature(all_data,w2c_col="x-y-1000",vec_len=32,feature_path=feature_path,mode="min")
    # w2c_feature(all_data,w2c_col="x-y-1000",vec_len=32,feature_path=feature_path,mode="max")
    
    
    


    # percent = 100
    # w2c_col ="x"
    # all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]/percent).astype("str")
    # w2c_col="y"
    # all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]/percent).astype("str")
    # all_data["x-y"] = all_data["x_gai"]+"-"+all_data["y_gai"]
    # w2c_feature(all_data,w2c_col="x-y",vec_len=32,feature_path=feature_path,mode="std")