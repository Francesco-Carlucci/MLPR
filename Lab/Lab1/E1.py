#import numpy as np
import sys
import os

def computeRecord(scores):
    scores=list(map(float,scores))
    record=sum(scores)-min(scores)-max(scores)
    return record


def main():
    print('lab 1 Es. 1')
    if len(sys.argv)<2:
        print('use: main.py <score file>')
        exit(0)

    #filename=os.path.join('.',sys.argv[1])
    ranking={}
    natRanking={}
    try:
        with open(sys.argv[1]) as f:
            for line in f:
                name,surname,nationality,*scores=line.rsplit()
                record=computeRecord(scores)
                ranking[record]=name+" "+surname+" "+"Score: "+str(record)
                natRanking[nationality]=natRanking.get(nationality,0)+record
    except:
        print('Error opening file')
        exit(1)
    print('final ranking:')
    [print(i+1,': ',ranking[idx]) for i,idx in enumerate(sorted(ranking,reverse=True)[0:3])]
    print('Best country:')
    bestNat=sorted(natRanking,key=natRanking.get,reverse=True)[0]
    print(bestNat," Total score: ",natRanking[bestNat])



if __name__=='__main__':
    main()