import pandas
from sklearn.linear_model import LinearRegression
import pyfiglet
import warnings
warnings.simplefilter(action="ignore",category=FutureWarning)

text = pyfiglet.figlet_format("LinearRegression Model", font="digital")

print(text)
 
dataset = pandas.read_csv("mark.csv")

y = dataset["marks"]

x = dataset["hrs"].values.reshape(5,1)
 

mind = LinearRegression()

mind.fit(x,y)
#print(mind.predict([[ 4 ]]))

hours = input("enter the number of hours you study ")
result = mind.predict([[ hours ]])
print(" your result is nearly equal to = " , result)
