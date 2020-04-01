#Импорт моудлей
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
#Импорт данных
data = pd.read_csv('casc-resto.csv', sep=';', decimal=',', dtype='str')
dataConst = pd.read_csv('CASC_Constant.csv', sep=',', decimal=',', dtype='str')
#Привести колонку к типу данных datetime
data['RKDate'] = pd.to_datetime(data['RKDate'])
#поменять запятые на точки чтобы потом привести к float
data['SummBasic'] = [x.replace(',', '.') for x in data['SummBasic']]
data['SummAfterPointsUsage'] = [x.replace(',', '.') for x in data['SummAfterPointsUsage']]
data['SummBasic'] = data['SummBasic'].astype(float)
data['SummAfterPointsUsage'] = data['SummAfterPointsUsage'].astype(float)
#Объявить константы
START_DATE = dt.datetime(2017, 7, 1)
END_DATE = dt.datetime(2017, 12, 31)
#Дата между
data['Visited'] = (data['RKDate'] <= END_DATE) & (data['RKDate'] > START_DATE)
customers = data.groupby(['CustomerID'])
# Пустой датафрейм
df = pd.DataFrame(columns=['CustomerID', 'Visited', 'Recency', 'Frequency', 'Monetary Value', 'Mean Saved'])
#Перебор групп customerid
for state, frame in customers:
    dates = frame[frame['RKDate'] < START_DATE]
    if not dates.empty:
        #Дни между полседним визитом и 2017.07.01
        recency = START_DATE - max(dates['RKDate'])
        recency = recency.days
        #частота
        frequency = dates.shape[0] / ((START_DATE - min(dates['RKDate'])).days/30)
        #Средняя сумма
        monetaryValue = dates['SummBasic'].mean()
        #Средняя сумма сохраненных денег
        meanSaved = (dates['SummBasic'] - dates['SummAfterPointsUsage']).mean()
        #Добавить строку в датафрейм
        df = df.append(
            {'CustomerID': state, 'Visited': int(frame['Visited'].any()), 'Recency': recency, 'Frequency': frequency,
             'Monetary Value': monetaryValue, 'Mean Saved': meanSaved}, ignore_index=True)
# оставляем только где чек больше 200
df = df[df['Monetary Value'] > 200]
#оставляем посещения меньше 20 в месяц
df = df[df['Frequency'] < 20]
#Изменить имя колонки
dataConst.rename(columns={'CustomerId': 'CustomerID'}, inplace=True)
merged = pd.merge(df, dataConst, on='CustomerID')
#Удлить ненужное
merged = merged.drop(columns=['ActivationDate', 'SubscribedEmail', 'SubscribedPush'])
merged = merged.dropna()
#Привести к инт
merged['Visited'] = merged['Visited'].astype(int)
y = merged['Visited']
merged = merged.drop(columns=['Visited'])
#Привести к инт
merged['Age'] = merged['Age'].astype(int)
merged['Sex'] = (merged['Sex'] == 'Female').astype(int)
#Разделить выборки на тестовую и тренировочную
X_train, X_test, y_train, y_test = train_test_split(merged, y, test_size=0.2)
#Экземпляр LogReg
logreg = LogisticRegression()
#Подставить данные
logreg.fit(X_train, y_train)
#Предасказать y
y_pred = logreg.predict(X_test)
print(y_pred)
#Оценка качетва предсказания
scoreTrain = logreg.score(X_train, y_train)
print(scoreTrain)
score = logreg.score(X_test, y_test)
print(score)
# оценка показателей precision и recall
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))