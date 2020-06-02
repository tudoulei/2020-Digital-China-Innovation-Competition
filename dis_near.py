import shapefile
import numpy as np
import seaborn as sns
import pandas as pd
from shapely.geometry import LineString, Point

# feature_path = "../code//temp/basefea/"
# train_label = pd.read_csv(feature_path+'train_label.csv')
# from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# file=shapefile.Reader(r'E:\jupyter\智慧海洋\data\gadm36_CHN_shp/gadm36_CHN_0.shp')  #读取省界.shp文件
file=shapefile.Reader(r'./shp/my.shp')  #读取省界.shp文件
# file=shapefile.Reader(r'./shp/my2.shp')  #读取省界.shp文件
shapes=file.shapes()   #获取point 
records=file.records() #获取省名称

pro_points=[]  #建立省边界列表
pro_names=[] #建立省名称列表

for i in range(len(shapes)):
    points=shapes[i].points #h获取经纬度数据    
    lon =[]  
    lat  =[] 
    #将每个tuple的lon和lat组合起来
    [lon.append(points[i][0]) for i in range(len(points))]  
    [lat.append(points[i][1]) for i in range(len(points))]   
 
    lon=np.array(lon).reshape(-1,1)
    lat=np.array(lat).reshape(-1,1)
    loc=np.concatenate((lon,lat),axis=1)
    
    pro_points.append(loc)
    # pro_names.append(pro_name)

lat_min,lat_max=37,55
lon_min,lon_max=115,135
import numpy as np



lon_zong = np.array([])
lat_zong=np.array([])

for point in pro_points:
    

    lon=point[:,0]
    lat=point[:,1]
    lon_zong =np.append(lon_zong,lon)
    lat_zong =np.append(lat_zong,lat)

def func1(train_label):
    dis=[]
    line = LineString([(i,j) for i,j in zip(lon_zong,lat_zong)])
    from tqdm import tqdm 
    for i in tqdm(range(train_label.shape[0])):
        
        p = Point(train_label["y_mean"].values[i],train_label["x_mean"].values[i])
        dis.append(p.distance(line))
        
    # train["dis"] = dis
    return dis

def func2(train):
    dis=[]
    line = LineString([(i,j) for i,j in zip(lon_zong,lat_zong)])
    from tqdm import tqdm 
    for i in tqdm(range(train.shape[0])):
        
        p = Point(train["y"].values[i],train["x"].values[i])
        dis.append(p.distance(line))
        
    # train["dis"] = dis
    return dis
