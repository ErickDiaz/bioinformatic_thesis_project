#!/usr/bin/env python

from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

def MediaMovilOperation(ArrayDataOp,Position,WindowOp):
	Sum = 0
	for x in range(WindowOp):
		Sum = Sum + float(ArrayDataOp[Position-x])

	FilterData = Sum/float(WindowOp)
	return FilterData

def MediaMovilFilter(ArrayData,Window):
	FilterArray=[]
	for x in range(len(ArrayData)):
		if(x>=Window):
			FilterArray.insert(x,MediaMovilOperation(ArrayData,x,Window))
		else:
			FilterArray.insert(x,MediaMovilOperation(ArrayData,x,x+1))

	return FilterArray


file_object = open("datos1.log", "r")
data = file_object.read().split("(next mAP calculation at ")

Avg_IOU_List = []
Avg_Recall_List = []
Avg_Loss_List = []
Count_List = []
Counter = 0

for i in range(0,len(data)):
#for i in range(0,3):
	row = data[i].split("\n")

	Avg_IOU = 0
	Avg_Recall = 0
	Avg_Loss = 0
	Count = 0
	Avg = 0;

	for j in range(0,len(row)):
		if(row[j].find("avg loss")!=-1):
			
			column = row[j].split(', ')
			for k in range(0, len(column)):
				if(column[k].find("avg loss")!=-1):
					val = column[k].split(" ")
					Avg_Loss = val[0]

		if(row[j].find("Region")!=-1):

			column = row[j].split(', ')
			Avg += 1

			for k in range(0, len(column)):
				if(column[k].find('IOU')!=-1):
					val = column[k].split(': ')
					Avg_IOU += float(val[1])
				if(column[k].find('Recall')!=-1):
					val = column[k].split(': ')
					Avg_Recall += float(val[1])
				if(column[k].find('count')!=-1):
					val = column[k].split(': ')
					Count += float(val[1])


	if(Avg!=0):
		Avg_IOU = Avg_IOU/float(Avg)
		Avg_Recall = Avg_Recall/float(Avg)
		Avg_Loss = float(Avg_Loss)/100

		if(Avg_Loss>1):
			Avg_Loss=1

		Avg_IOU_List.append(Avg_IOU)
		Avg_Recall_List.append(Avg_Recall)
		Avg_Loss_List.append(Avg_Loss)
		Count_List.append(Count)

		Avg_IOU_List_Filter = MediaMovilFilter(Avg_IOU_List, 100)
		Avg_Recall_List_Filter = MediaMovilFilter(Avg_Recall_List, 100)
		Avg_Loss_List_Filter = MediaMovilFilter(Avg_Loss_List, 100)

		print(str(Counter) + "| AVG IoU: " + str(Avg_IOU) + "| Avg Recall: " + str(Avg_Recall) + "| Avg Loss: " + str(Avg_Loss) + "| Count: " + str(Count))
		Counter+=1

#print(len(Avg_IOU_List))
#print(len(Avg_Recall_List))
#print(len(Avg_Loss_List))
#print(len(Count_List))
#print(len(Avg_IOU_List_Filter))
#print(len(Avg_Recall_List_Filter))
#print(len(Avg_Loss_List_Filter))

fig = plt.figure(1)

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(Avg_IOU_List, color=colors['blue'], linewidth=0.5)
ax1.plot(Avg_IOU_List_Filter, color=colors['deepskyblue'], linewidth=2)
ax1.set_ylabel('Avg IoU')
ax1.grid(True)

ax2.plot(Avg_Recall_List, color=colors['orange'], linewidth=0.5)
ax2.plot(Avg_Recall_List_Filter, color=colors['lightcoral'], linewidth=2)
ax2.set_ylabel('Avg Recall')
ax2.grid(True)

ax3.plot(Avg_Loss_List, color=colors['green'], linewidth=0.5)
ax3.plot(Avg_Loss_List_Filter, color=colors['greenyellow'], linewidth=2)
ax3.set_ylabel('Avg Loss')
ax3.grid(True)

ax4.plot(Avg_Loss_List, color=colors['green'], linewidth=0.5)
ax4.plot(Avg_IOU_List, color=colors['blue'], linewidth=0.5)
ax4.plot(Avg_Loss_List_Filter, color=colors['greenyellow'], linewidth=2)
ax4.plot(Avg_IOU_List_Filter, color=colors['deepskyblue'], linewidth=2)
ax4.set_ylabel('Comparisson Avg Loss & Avg IoU')
ax4.grid(True)

plt.show()