

# import plot_time_series


import numpy as np 
from random import randint

import matplotlib.pyplot as plt 
# marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:

COMPOSITE_N = 23 
band_fields = list()
for bandi in range(6):
    # bandi=0
    for i in range(14+bandi*COMPOSITE_N,14+COMPOSITE_N+bandi*COMPOSITE_N):
        band_fields.append(str(i))


blue_band_fields = list()
bandi=0
for i in range(14+bandi*COMPOSITE_N,14+COMPOSITE_N+bandi*COMPOSITE_N):
    blue_band_fields.append(str(i))

green_band_fields = list()
bandi=1
for i in range(14+bandi*COMPOSITE_N,14+COMPOSITE_N+bandi*COMPOSITE_N):
    green_band_fields.append(str(i))

red_band_fields = list()
bandi=2
for i in range(14+bandi*COMPOSITE_N,14+COMPOSITE_N+bandi*COMPOSITE_N):
    red_band_fields.append(str(i))

nir_band_fields = list()
bandi=3
for i in range(14+bandi*COMPOSITE_N,14+COMPOSITE_N+bandi*COMPOSITE_N):
    nir_band_fields.append(str(i))

sw1_band_fields = list()
bandi=4
for i in range(14+bandi*COMPOSITE_N,14+COMPOSITE_N+bandi*COMPOSITE_N):
    sw1_band_fields.append(str(i))

sw2_band_fields = list()
bandi=5
for i in range(14+bandi*COMPOSITE_N,14+COMPOSITE_N+bandi*COMPOSITE_N):
    sw2_band_fields.append(str(i))

markers2 = 'o'
markerl8 = "^"
# plotting the points  
x=np.array(range(1,COMPOSITE_N+1))


def rgb_to_hex(rgb):
    r,g,b=rgb
    return '#%02x%02x%02x' % (r,g,b)

# print(rgb_to_hex((0, 34, 255)))
# minimal color 
min_color = 50
plot_time = 4


def plot_all(data_all):
    test_field = '13'
    yclasses = data_all[test_field]
    yclass_names = data_all['12']
    ids = data_all['0']
    unique_id = np.unique(ids)
    
    for idi in unique_id:
        unique_yclass = np.unique(yclasses[ids==idi])
        for yj in unique_yclass:
            index = np.where(np.logical_and (ids==idi, yclasses==yj))[0]
            unique_names = np.unique(yclass_names[index])
            # print(index)
            # print(np.unique(yclasses[index]))
            print(str(idi))
            print(unique_names)
            if unique_names.size>1:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!unique_names.size>1:")
            
            ## generate colors 
            blu_colors = []
            for i in range(index.size):
                blu_colors.append(rgb_to_hex((0,0,randint(min_color, 255))))
            gre_colors = []
            for i in range(index.size):
                gre_colors.append(rgb_to_hex((0,randint(min_color, 255),0)))
            red_colors = []
            for i in range(index.size):
                red_colors.append(rgb_to_hex((randint(min_color, 255),0,0)))
            
            # colors_arr = np.tile(blue_colors,COMPOSITE_N)
            x2 = np.transpose(np.tile(x,index.size).reshape(index.size,COMPOSITE_N))
            plt.scatter(x2, np.transpose((np.array(data_all[ blue_band_fields]))[index,:] ), c=np.tile(blu_colors,COMPOSITE_N) )
            plt.scatter(x2, np.transpose((np.array(data_all[green_band_fields]))[index,:] ), c=np.tile(gre_colors,COMPOSITE_N) )
            plt.scatter(x2, np.transpose((np.array(data_all[  red_band_fields]))[index,:] ), c=np.tile(red_colors,COMPOSITE_N) )
            plt.title(str(idi) + " "+ unique_names[0]); plt.xlabel("x-label"); plt.ylabel("y-label")            
            plt.show(block=False); plt.pause(plot_time)
            # break
            plt.close()            
            plt.scatter(x2, np.transpose((np.array(data_all[  nir_band_fields]))[index,:] ), c=np.tile(blu_colors,COMPOSITE_N) )
            plt.scatter(x2, np.transpose((np.array(data_all[  sw1_band_fields]))[index,:] ), c=np.tile(gre_colors,COMPOSITE_N) )
            plt.scatter(x2, np.transpose((np.array(data_all[  sw2_band_fields]))[index,:] ), c=np.tile(red_colors,COMPOSITE_N) )
            plt.title(str(idi) + " "+ unique_names[0]); plt.xlabel("x-label"); plt.ylabel("y-label")            
            plt.show(block=False); plt.pause(plot_time)
            plt.close()            
            # break
        # break

def plot_per_site(data_all, unique_id=[1]):
    test_field = '13'
    yclasses = data_all[test_field]
    yclass_names = data_all['12']
    ids = data_all['0']
    # unique_id = np.unique(ids)
    
    for idi in unique_id:
        unique_yclass = np.unique(yclasses[ids==idi])
        for yj in unique_yclass:
            index = np.where(np.logical_and (ids==idi, yclasses==yj))[0]
            unique_names = np.unique(yclass_names[index])
            # print(index)
            print(str(idi))
            # print(np.unique(yclasses[index]))
            print(unique_names)
            if unique_names.size>1:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!unique_names.size>1:")
            
            ## generate colors 
            blu_colors = []
            for i in range(index.size):
                blu_colors.append(rgb_to_hex((0,0,randint(min_color, 255))))
            gre_colors = []
            for i in range(index.size):
                gre_colors.append(rgb_to_hex((0,randint(min_color, 255),0)))
            red_colors = []
            for i in range(index.size):
                red_colors.append(rgb_to_hex((randint(min_color, 255),0,0)))
            
            # colors_arr = np.tile(blue_colors,COMPOSITE_N)
            x2 = np.transpose(np.tile(x,index.size).reshape(index.size,COMPOSITE_N))
            plt.scatter(x2, np.transpose((np.array(data_all[ blue_band_fields]))[index,:] ), c=np.tile(blu_colors,COMPOSITE_N) )
            plt.scatter(x2, np.transpose((np.array(data_all[green_band_fields]))[index,:] ), c=np.tile(gre_colors,COMPOSITE_N) )
            plt.scatter(x2, np.transpose((np.array(data_all[  red_band_fields]))[index,:] ), c=np.tile(red_colors,COMPOSITE_N) )
            plt.title(str(idi) + " "+ unique_names[0]); plt.xlabel("x-label"); plt.ylabel("y-label")            
            plt.show(block=False); plt.pause(plot_time)
            # break
            plt.close()            
            plt.scatter(x2, np.transpose((np.array(data_all[  nir_band_fields]))[index,:] ), c=np.tile(blu_colors,COMPOSITE_N) )
            plt.scatter(x2, np.transpose((np.array(data_all[  sw1_band_fields]))[index,:] ), c=np.tile(gre_colors,COMPOSITE_N) )
            plt.scatter(x2, np.transpose((np.array(data_all[  sw2_band_fields]))[index,:] ), c=np.tile(red_colors,COMPOSITE_N) )
            plt.title(str(idi) + " "+ unique_names[0]); plt.xlabel("x-label"); plt.ylabel("y-label")            
            plt.show(block=False); plt.pause(plot_time)
            plt.close()            
            # break
        # break


# plot_per_site(data_all)

# , label="marker='{0}'".format(markerl8)            # x2 = np.repeat(x,index.size).reshape(COMPOSITE_N,index.size)
            # y2 = np.transpose((np.array(data_all[blue_band_fields]))[index,:] )
            # y2 = (np.array(data_all[blue_band_fields]))[index,:].reshape(COMPOSITE_N,index.size)
            # colors_arr = np.repeat(colors,COMPOSITE_N)
            # x2 = np.tile(x,index.size).reshape(COMPOSITE_N,index.size)
        # print (unique_yclass)
                # colors.append('#%06X' % randint(0, 0xFFFFFF))
                # colors.append('#%06X' % randint(0, 0x0000FF))
        # break
            # for i in index:
                # plt.scatter(x, np.array(data_all[red_band_fields][i:(i+1)]).reshape(23), label="marker='{0}'".format(markerl8)) 
    
