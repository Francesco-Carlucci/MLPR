import os
import sys
import math


def pointDist(p1, p2):
    #return math.sqrt((p1.y - p2.y) ** 2 - (p1.x - p2.x) ** 2)
    return math.sqrt((int(p1[1]) - int(p2[1])) ** 2 - (int(p1[0]) - int(p2[0])) ** 2)

def computeDistance(coords):
    sortedTimes=sorted(coords)
    totTime = sortedTimes[-1] - sortedTimes[0]
    totDst=0
    p0=coords[sortedTimes[0]]
    for t in sortedTimes[1:]:
        p1=coords[t]
        totDst+=pointDist(p0,p1)
        p0=p1
    return totDst,totTime

def main():
    print('lab1 es.2')

    if len(sys.argv)<4:
        print('use: main.py <score file> [-b <busId> or -l <lineId>]')
        exit(1)
    flag=sys.argv[2]
    if flag=='-b':
        busId=sys.argv[3]
    elif flag=='-l':
        lineId=sys.argv[3]
    else:
        print("invalid flag, it must be -b or -l")
        exit(2)
    busCoords={}
    lineBuses={}
    with open(sys.argv[1]) as f:
        for line in f:
            bid,lid,x,y,time=line.rsplit()
            if flag == '-b' and bid==busId:
                busCoords[time]=(x,y)
            elif flag=='-l' and lid==lineId:
                if bid in lineBuses:
                    lineBuses[bid][int(time)]=(x,y)
                else:
                    lineBuses[bid]={int(time):(x,y)}

    if flag=='-b':
        totDist,_=computeDistance(busCoords)
        print(busId," - Total Distance: ",totDist)
    else:
        totTime=0
        totDst=0
        for bus in lineBuses.keys():
            busdst,busTime=computeDistance(lineBuses[bus])
            totDst+=busdst
            totTime+=busTime
        print(lineId," - Avg Speed: ",totDst/totTime)


if __name__ == '__main__':
    main()