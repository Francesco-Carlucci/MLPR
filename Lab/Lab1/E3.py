import sys

months=['','January','February','March','April','May','June','July',
        'August','September','October','November','December']
"""
monthsIdx={'January':1,'February':2,'March':3,
           'April':4,'May':5,'June':6,'July':7,
           'August':8,'September':9,'October':10,
           'November':11,'December':12}
"""

class Birth:
    def __init__(self,name,surname,place,date):
        self.name=name
        self.surname=surname
        self.place=place
        self.day,month,self.year=date.split(sep='/')
        self.month=int(month)
        #self.month=months[int(monthIdx)-1]

def main():
    print("lab1 es. 3")

    if len(sys.argv)<2:
        print('use: main.py <score file>')

    birthList=[]
    cityCnt = {}
    monthCnt = {}
    try:
        with open(sys.argv[1]) as f:
            for line in f:
                name,surname,place,date=line.rsplit()
                day,month,year=date.split(sep='/')
                birthList.append(Birth(name,surname,place,date))
                cityCnt[place]=cityCnt.get(place,0)+1
                monthCnt[int(month)] = monthCnt.get(int(month), 0) + 1
    except:
        print("Error opening file")
        exit(1)
    """
    cityCnt={}
    monthCnt={}
    for birth in birthList:
        cityCnt[birth.place]=cityCnt.get(birth.place,0)+1
        monthCnt[birth.month]=monthCnt.get(birth.month,0)+1
    """
    print("Births per city:")
    [print("    ",k,': ',cityCnt[k]) for k in cityCnt.keys()]
    print("Births per month:")
    [print("    ", months[k], ': ', monthCnt[k]) for k in sorted(monthCnt)]
    print('Average number of births: %.2f' % (len(birthList)/len(cityCnt.keys())))


if __name__=="__main__":
    main()