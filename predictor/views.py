from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.template import loader
import pandas as pd
import pickle
import sklearn


def home(request):
    template = loader.get_template('homepage.html')
    return HttpResponse(template.render({}, request))

def predict(request):
    team1 = request.POST.get('team1', False)
    team2 = request.POST.get('team2', False)
    if team1 and team2:
        d = {'Name': {0: 'Qatar', 1: 'Ecuador', 2: 'Senegal', 3: 'Netherlands', 4: 'England', 5: 'Iran', 6: 'USA', 7: 'Wales', 8: 'Argentina', 9: 'Saudi', 10: 'Mexico', 11: 'Poland', 12: 'France', 13: 'Australia', 14: 'Denmark', 15: 'Tunisia', 16: 'Spain', 17: 'Costa Rica', 18: 'Germany', 19: 'Japan', 20: 'Belgium', 21: 'Canada', 22: 'Morocco', 23: 'Croatia', 24: 'Brazil', 25: 'Serbia', 26: 'Switzerland', 27: 'Cameroon', 28: 'Portugal', 29: 'Ghana', 30: 'Uruguay', 31: 'South Korea'}, 'Rank': {0: 1439.89, 1: 1464.39, 2: 1584.38, 3: 1694.51, 4: 1728.47, 5: 1564.61, 6: 1627.48, 7: 1569.82, 8: 1773.88, 9: 1437.78, 10: 1644.89, 11: 1548.59, 12: 1759.78, 13: 1488.72, 14: 1666.57, 15: 1507.54, 16: 1715.22, 17: 1503.59, 18: 1650.21, 19: 1559.54, 20: 1816.71, 21: 1475.0, 22: 1563.5, 23: 1645.64, 24: 1841.3, 25: 1563.62, 26: 1635.92, 27: 1471.44, 28: 1676.56, 29: 1393.0, 30: 1638.71, 31: 1530.3}, 'Att': {0: 71, 1: 68, 2: 78, 3: 83, 4: 87, 5: 81, 6: 82, 7: 72, 8: 87, 9: 70, 10: 78, 11: 80, 12: 85, 13: 70, 14: 77, 15: 72, 16: 82, 17: 73, 18: 76, 19: 77, 20: 85, 21: 74, 22: 75, 23: 77, 24: 82, 25: 79, 26: 76, 27: 75, 28: 83, 29: 77, 30: 81, 31: 74}, 'Mid': {0: 70, 1: 73, 2: 76, 3: 84, 4: 83, 5: 71, 6: 74, 7: 71, 8: 82, 9: 69, 10: 75, 11: 76, 12: 82, 13: 70, 14: 81, 15: 70, 16: 84, 17: 73, 18: 85, 19: 78, 20: 83, 21: 74, 22: 71, 23: 81, 24: 83, 25: 79, 26: 75, 27: 70, 28: 82, 29: 79, 30: 77, 31: 72}, 'Def': {0: 68, 1: 74, 2: 80, 3: 81, 4: 82, 5: 71, 6: 74, 7: 72, 8: 82, 9: 70, 10: 75, 11: 74, 12: 84, 13: 69, 14: 79, 15: 69, 16: 83, 17: 74, 18: 83, 19: 76, 20: 80, 21: 70, 22: 76, 23: 76, 24: 82, 25: 74, 26: 76, 27: 72, 28: 81, 29: 77, 30: 77, 31: 71}, 'Gk': {0: 68, 1: 70, 2: 86, 3: 72, 4: 82, 5: 70, 6: 77, 7: 74, 8: 84, 9: 71, 10: 85, 11: 86, 12: 87, 13: 78, 14: 81, 15: 68, 16: 83, 17: 87, 18: 90, 19: 72, 20: 90, 21: 77, 22: 84, 23: 80, 24: 89, 25: 78, 26: 86, 27: 66, 28: 80, 29: 72, 30: 78, 31: 73}, 'TAtt': {0: 76, 1: 77, 2: 79, 3: 85, 4: 89, 5: 82, 6: 82, 7: 81, 8: 91, 9: 76, 10: 81, 11: 91, 12: 91, 13: 75, 14: 81, 15: 71, 16: 82, 17: 77, 18: 84, 19: 78, 20: 86, 21: 84, 22: 83, 23: 83, 24: 89, 25: 84, 26: 77, 27: 75, 28: 90, 29: 81, 30: 82, 31: 89}, 'TMid': {0: 72, 1: 73, 2: 82, 3: 87, 4: 85, 5: 74, 6: 79, 7: 76, 8: 84, 9: 74, 10: 80, 11: 78, 12: 83, 13: 75, 14: 83, 15: 79, 16: 85, 17: 76, 18: 89, 19: 81, 20: 91, 21: 77, 22: 76, 23: 88, 24: 89, 25: 86, 26: 80, 27: 75, 28: 88, 29: 84, 30: 85, 31: 76}, 'TDef': {0: 71, 1: 79, 2: 87, 3: 85, 4: 84, 5: 72, 6: 77, 7: 79, 8: 85, 9: 73, 10: 77, 11: 76, 12: 85, 13: 72, 14: 81, 15: 71, 16: 86, 17: 77, 18: 87, 19: 79, 20: 82, 21: 72, 22: 84, 23: 81, 24: 88, 25: 80, 26: 81, 27: 75, 28: 88, 29: 78, 30: 83, 31: 79}, 'Form': {0: -2.08, 1: 0.69, 2: -0.21, 3: 15.1, 4: -8.99, 5: 5.97, 6: -7.53, 7: -12.31, 8: 3.23, 9: 2.04, 10: -4.68, 11: 2.41, 12: -5.07, 13: 4.99, 14: 1.1, 15: -0.32, 16: -1.71, 17: 3.53, 18: -8.75, 19: 4.85, 20: -5.21, 21: 1.18, 22: 5.15, 23: 13.49, 24: 3.74, 25: 14.09, 26: 14.49, 27: -13.51, 28: -2.09, 29: -0.47, 30: -2.24, 31: 4.28}}
        df = pd.DataFrame(d)
        model = pickle.load(open('knn_model', 'rb'))
        t1 = df[df['Name'] == team1].values.tolist()[0][1:]
        t2 = df[df['Name'] == team2].values.tolist()[0][1:]
        X1 = [t1 + t2]
        X2 = [t2 + t1]
        res1 = model.predict(X1)[0]
        res2 = model.predict(X2)[0]
        res = (res1 + res2)/2
        winner = 'Draw'
        if res > 0:
            winner = team1
        elif res < 0:
            winner = team2
        res = abs(round(res, 2))
        template = loader.get_template('predictor.html')
        context = {'winner': winner, 'gd': res}
        return HttpResponse(template.render(context, request))
    else:
        return HttpResponseRedirect(reverse('home'))

def exit(request):
    return HttpResponseRedirect(reverse('home'))
        
