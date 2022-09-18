from statistics import mode
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Regresi[on lineal simple utilizando minimos cuadrados
#Ecuacion de la regresión
#A00829598
#Juan Pablo Yáñez González


#Y = m * x + b

# Primero comenzamos por importar nuestra base de datos

df = pd.read_csv('kc_house_data.csv')

# Definimos valores para nuestro espacio y precio

space = df['sqft_living']
price = df['price']

# definimos los valores de x y Y en base a que quiero predecir, en este caso el precio sera nuestra variable dependiente ya que queremos 
# ver como cambia con respecto a el espacio.

x = df.sqft_living
y = df.price


# Estos valores se usaran para la comprobación con la libreria.

x1 = df[['sqft_living']]
y1 = df[['price']]

#Ya que este es un modelo tenemos que dividir sus datos en entrenamiento y testeo.
# Para esto dividiremos la base de datos con una función de sklearn

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0)

xtrainSk, xtestSk, ytrainSk, ytestSk = train_test_split(x1,y1,test_size=1/3, random_state=0)

# Ya con los valores de entrenamiento y testeo establecidos nos falta calcular la pendiente "m" y b

# Para hacer esto calculamos la suma de cuadrados de x y la multiplicacion de ssxx y ssxy

#utilizaremos los valores del modelo de entrenamiento para ir entrenanto la prediccion del modelo y luego lo comprobaremos con la de testeo

mediaX = xtrain.mean()

df['xDiferencia'] = mediaX - xtrain

df['xDiferenciaCuadrada'] = df.xDiferencia**2

SSxx = df.xDiferenciaCuadrada.sum()


mediaY = ytrain.mean()

df['yDiferencia'] = mediaY - ytrain
SSxy = (df.xDiferencia * df.yDiferencia).sum()



m = SSxy/SSxx

b = mediaY - m * mediaX

# ahora definiremos una funcion que realice la ecuación de minimos cuadrados sustituyuendo los valores.


def predict(value):
    predict = m*value+b 
    return predict


print('La predicción del precio de acuerdo a un piso de 1180m**2 con nuestro modelo es: ')
print(predict(1180))
print('\n')


# Realizamos predicciones para todos nuestros valores de testeo
predManual = predict(xtest)


# finalmente comprararemos nuestro modelo con los datos de testeo previamente guardados y con el modelo de sklearn




# Para comprobar que esta bien nuestro modelo lo compararemos contra el de la libreria de sklearn
# contra su modelo de regresion lineal simple

from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(xtrainSk,ytrainSk)

# Realizamos predicciones para todos nuestros valores de testeo
pred = model.predict(xtestSk)

print(pred)

# Podemos observar que nuestro modelo funciona de manera correcta obteniendo el mismo resultado
print('La predicción del precio de acuerdo a un piso de 1180m**2 con el modelo de sklearn es: ')
print(model.predict([[1180]]))
print('\n')


print('Podemos observar que nuestro modelo tiene datos identicos con las predicciones de la libreria sklearn por lo que nos indica que funciona de manera adecuada \n')
print(predManual[0:10])
print(pred[0:10])

# Para obtener la exactitud del modelo utilizaremos la variable score

print('\n')
print('El score de la predicción es: ')
print(model.score(xtestSk,ytestSk))

print('El score nos indica que este no es un gran modelo para predecir los datos pero no porque se haya creado de forma erronea, sino porque la base de datos tienen más variables que no se tomaron en cuenta con la regresion lineal simple, por lo que una regressión multiple deberia de trabajar de forma más efectiva.')

