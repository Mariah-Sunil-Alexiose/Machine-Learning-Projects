from flask import Flask, render_template, request
import joblib
from logging import debug
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import difflib

app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/gold_stock',methods=['GET', 'POST'])
def gold_stock():
    if request.method == 'POST':
        Open = request.form.get('Open')
        High = request.form.get('High')
        Low = request.form.get('Low')
        Volume = request.form.get('Volume')
        test_data = np.array([[Open, High, Low, Volume]])
        trained_model = joblib.load('notebooks/models/stock.pkl')
        prediction = trained_model.predict(test_data)
        return render_template('prediction.html', prediction=prediction)
    return render_template('gold_stock.html')

@app.route('/gold_price',methods=['GET', 'POST'])
def gold_price():
    if request.method == 'POST':
        spx = request.form.get('spx')
        uso = request.form.get('uso')
        slv = request.form.get('slv')
        eur_usd = request.form.get('eur_usd')
        test_data = np.array([[spx, uso, slv, eur_usd]])
        trained_model = joblib.load('notebooks/models/gold_price')
        prediction = trained_model.predict(test_data)
        return render_template('prediction.html', prediction=prediction)
    return render_template('gold_price.html')

@app.route('/movie_recommender',methods=['GET', 'POST'])
def movie_recommender():
    if request.method == 'POST':
        movie_name = request.form.get('movie')
        movies_data = pd.read_csv('notebooks/csv/movies.csv')
        similarity = joblib.load('notebooks/models/movie_recommender.pkl')
        list_of_all_titles = movies_data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
        i = 0   
        movies={}   
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_data[movies_data.index==index]['title'].values[0]
            if (i<30):
                i+=1 
                movies[i]=title_from_index
        return render_template('movies_prediction.html', movies=movies)
    return render_template('movie_recommender.html')

@app.route('/car',methods=['GET', 'POST'])
def car():
    if request.method == 'POST':
        year = request.form.get('year')
        present_price = request.form.get('present_price')
        kms_driven = request.form.get('kms_driven')
        fuel_type = request.form.get('fuel_type')
        seller_type = request.form.get('seller_type')
        transmission = request.form.get('transmission') 
        owner = request.form.get('owner')
        test_data = np.array([[year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]])
        trained_model = joblib.load('notebooks/models/car.pkl')
        prediction = trained_model.predict(test_data)
        return render_template('prediction.html', prediction=prediction)
    return render_template('car.html')

@app.route('/credit_card',methods=['GET', 'POST'])
def credit_card():
    if request.method == 'POST':
        time = request.form.get('time')
        v1 = request.form.get('v1')
        v2 = request.form.get('v2')
        v3 = request.form.get('v3')
        v4 = request.form.get('v4')
        v5 = request.form.get('v5')
        v6 = request.form.get('v6')
        v7 = request.form.get('v7')
        v8 = request.form.get('v8')
        v9 = request.form.get('v9')
        v10 = request.form.get('v10')
        v11 = request.form.get('v11')
        v12 = request.form.get('v12')
        v13 = request.form.get('v13')
        v14 = request.form.get('v14')
        v15 = request.form.get('v15')
        v16 = request.form.get('v16')
        v17 = request.form.get('v17')
        v18 = request.form.get('v18')
        v19 = request.form.get('v19')
        v20 = request.form.get('v20')
        v21 = request.form.get('v21')
        v22 = request.form.get('v22')
        v23 = request.form.get('v23')
        v24 = request.form.get('v24')
        v25 = request.form.get('v25')
        v26 = request.form.get('v26')
        v27 = request.form.get('v27')
        v28 = request.form.get('v28')
        amount = request.form.get('amount')
        test_data = np.array([[time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount]])
        trained_model = joblib.load('notebooks/models/credit_card.pkl')
        prediction = trained_model.predict(test_data)
        if prediction == 0:
            prediction = "Fraudulent transaction"
        else:
            prediction = "Genuine transaction"
        return render_template('prediction.html', prediction=prediction)
    return render_template('credit_card.html')

@app.route('/heart',methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        age = request.form.get('age')
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = request.form.get('trestbps')
        chol = request.form.get('chol')
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form.get('thalach')
        exang = request.form['exang']
        oldpeak = request.form.get('oldpeak')
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        input_data_as_numpy_array = np.asarray([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        # reshape the numpy array as we are predicting for only on instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        trained_model = joblib.load('notebooks/models/heart.pkl')
        prediction = trained_model.predict(input_data_reshaped)
        if (prediction[0] == 0):
            prediction = "The person does not have a heart disease"
        else:
            prediction = "The person does have a heart disease"
        return render_template('prediction.html', prediction=prediction)
    return render_template('heart.html')

@app.route('/titanic',methods=['GET', 'POST'])
def titanic():
    if request.method == 'POST':
        pclass = request.form.get('pclass')
        sex = request.form['sex']
        age = request.form['age']
        sibsp = request.form.get('sibsp')
        parch = request.form.get('parch')
        fare = request.form.get('fare')
        embarked = request.form['embarked']
        test_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        trained_model = joblib.load('notebooks/models/titanic.pkl')
        prediction = trained_model.predict(test_data)
        if prediction == 0:
            prediction = "The person has not survived"
        else:
            prediction = "The person has survived"
        return render_template('prediction.html', prediction=prediction)
    return render_template('titanic.html')

@app.route('/wine',methods=['GET', 'POST'])
def wine():
    if request.method == 'POST':
        fixed_acidity = request.form.get('fixed_acidity')
        volatile_acidity = request.form.get('volatile_acidity')
        citric_acid = request.form.get('citric_acid')
        residual_sugar = request.form.get('residual_sugar')
        chorides = request.form.get('chorides')
        free_sulfur_dioxide = request.form.get('free_sulfur_dioxide')
        total_sulfur_dioxide = request.form.get('total_sulfur_dioxide')
        density = request.form.get('density')
        pH = request.form.get('ph')
        sulphates = request.form.get('sulphates')
        alcohol = request.form.get('alcohol')
        # changing the input array to numpy array
        input_data_as_numpy_array = np.asarray([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)  
        trained_model = joblib.load('notebooks/models/wine.pkl')
        prediction = trained_model.predict(input_data_reshaped)
        if (prediction[0] == 1):
            prediction = "Wine is good quality"
        else:
            prediction = "Wine is bad quality"
        return render_template('prediction.html', prediction=prediction)
    return render_template('wine.html')

@app.route('/diabeties',methods=['GET', 'POST'])
def diabeties():
    if request.method == 'POST':
        pregnancies = request.form.get('pregnancies')
        glucose = request.form.get('glucose')
        blood_pressure = request.form.get('blood_pressure')
        skin_thickness = request.form.get('skin_thickness')
        insulin = request.form.get('insulin')
        bmi = request.form.get('bmi')
        pedigree_function = request.form.get('pedigree_function')
        age = request.form.get('age')
        # changing the input array to numpy array
        input_data_as_numpy_array = np.asarray([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age]])
        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        trained_models = joblib.load('notebooks/models/diabeties.pkl')
        std_data = trained_models[0].transform(input_data_reshaped)
        prediction = trained_models[1].predict(std_data)
        if (prediction[0] == 0):
            prediction = "The person is not diabetic"
        else:
            prediction = "The person is diabetic"
        return render_template('prediction.html', prediction=prediction)
    return render_template('diabeties.html')

@app.route('/loan',methods=['GET', 'POST'])
def loan():
    if request.method == 'POST':
        gender = request.form.get('gender')
        married = request.form.get('married')
        dependants = request.form.get('dependants')
        education = request.form.get('education')
        self_employed = request.form.get('self_employed')
        applicant_income = request.form.get('applicant_income')
        coapplicant_income = request.form.get('coapplicant_income')
        loan_amount = request.form.get('loan_amount')
        loan_amount_term = request.form.get('loan_amount_term')
        credit_history = request.form.get('credit_history')
        property_area = request.form.get('property_area')
        # changing the input array to numpy array
        input_data_as_numpy_array = np.array([[gender, married, dependants, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]])
        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        trained_model = joblib.load('notebooks/models/loan.pkl')
        prediction = trained_model.predict(input_data_reshaped)
        if (prediction[0] == 1):
            prediction = "The loan is approved"
        else:
            prediction = "The loan is rejected"
        return render_template('prediction.html', prediction=prediction)
    return render_template('loan.html')

@app.route('/parkinson',methods=['GET', 'POST'])
def parkinson():
    if request.method == 'POST':
        fo = request.form.get('fo')
        fhi = request.form.get('fhi')
        flo = request.form.get('flo')
        jitter_percent = request.form.get('jitter_percent')
        jitter_abs = request.form.get('jitter_abs')
        rap = request.form.get('rap')
        ppq = request.form.get('ppq')
        jitter_ddp = request.form.get('jitter_ddp')
        shimmer = request.form.get('shimmer')
        shimmer_db = request.form.get('shimmer_db')
        apq3 = request.form.get('apq3')
        apq5 = request.form.get('apq5')
        apq = request.form.get('apq')
        dda = request.form.get('dda')
        nhr = request.form.get('nhr')
        hnr = request.form.get('hnr')
        rpde = request.form.get('rpde')
        dfa = request.form.get('dfa')
        spread1 = request.form.get('spread1')
        spread2 = request.form.get('spread2')
        d2 = request.form.get('d2')
        ppe = request.form.get('ppe')
        input_data_as_numpy_array = np.asarray([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, jitter_ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1) # telling them we will be giving one input and not 156 data
        scaler = StandardScaler()
        parkinsons_data = pd.read_csv('notebooks/csv/parkinsons.csv')
        trained_models = joblib.load('notebooks/models/parkinson.pkl')
        std_data = trained_models[0].transform(input_data_reshaped)
        prediction = trained_models[1].predict(std_data)
        if (prediction[0] == 0):
            prediction = "The person does not have Parkinsons Disease"
        else:
            prediction = "Person has Parkinsons"
        return render_template('prediction.html', prediction=prediction)
    return render_template('parkinson.html')

@app.route('/spam_mail',methods=['GET', 'POST'])
def spam_mail():
    if request.method == 'POST':
        test_data = request.form.get('mail')
        test_data = [test_data]
        trained_models = joblib.load('notebooks/models/spam_mail.pkl')
        # converting text to feature vectors
        input_data_features = trained_models[1].transform(test_data)
        prediction = trained_models[0].predict(input_data_features)
        if (prediction[0] == 1):
            prediction = "Ham mail"
        else:
            prediction = "Spam mail"
        return render_template('prediction.html', prediction=prediction)
    return render_template('spam_mail.html')

@app.route('/fake_news',methods=['GET', 'POST'])
def fake_news():
    if request.method == 'POST':
        title = request.form.get('title')
        author = request.form.get('author')
        test_data = author + title
        test_data = [test_data]
        trained_models = joblib.load('notebooks/models/fake_news.pkl')
        test_data = trained_models[0].transform(test_data)
        prediction = trained_models[1].predict(test_data)
        if (prediction[0] == 0):
            prediction = "The news is real"
        else:
            prediction = "The news is fake"
        return render_template('prediction.html', prediction=prediction)
    return render_template('fake_news.html')
   
@app.route('/breast_cancer',methods=['GET', 'POST'])
def breast_cancer():
    if request.method == 'POST':
        radius_mean = request.form.get('radius_mean')
        texture_mean = request.form.get('texture_mean')
        perimeter_mean = request.form.get('perimeter_mean')
        area_mean = request.form.get('area_mean')
        smoothness_mean = request.form.get('smoothness_mean')
        compactness_mean = request.form.get('compactness_mean')
        concavity_mean = request.form.get('concavity_mean')
        concavep_mean = request.form.get('concavep_mean')
        symmetry_mean = request.form.get('symmetry_mean')
        fractionald_mean = request.form.get('fractionald_mean')
        radius_se = request.form.get('radius_se')
        texture_se = request.form.get('texture_se')
        perimeter_se = request.form.get('perimeter_se')
        texture_mean = request.form.get('texture_mean')
        area_se = request.form.get('area_se')
        smoothness_se = request.form.get('smoothness_se')
        compactness_se = request.form.get('compactness_se')
        concavity_se = request.form.get('concavity_se')
        concavep_se = request.form.get('concavep_se')
        symmetry_se = request.form.get('symmetry_se')
        fractionald_se = request.form.get('fractionald_se')
        radius_worst = request.form.get('radius_worst')
        texture_worst = request.form.get('texture_worst')
        perimeter_worst = request.form.get('perimeter_worst')
        texture_mean = request.form.get('texture_mean')
        area_worst = request.form.get('area_worst')
        smoothness_worst = request.form.get('smoothness_worst')
        compactness_worst = request.form.get('compactness_worst')
        concavity_worst = request.form.get('concavity_worst')
        concavep_worst = request.form.get('concavep_worst')
        symmetry_worst = request.form.get('symmetry_worst')
        fractionald_worst = request.form.get('fractionald_worst')
        test_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concavep_mean, symmetry_mean, fractionald_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concavep_se, symmetry_se, fractionald_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concavep_worst, symmetry_worst, fractionald_worst]])
        trained_model = joblib.load('notebooks/models/breast_cancer.pkl')
        # change the input data to a numpy array
        input_data_as_numpy_array = np.asarray(test_data)
        # reshape the numpy array as we are predicting for one datapoint
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = trained_model.predict(input_data_reshaped)
        if (prediction[0] == 0):
            prediction = 'Breast cancer is Malignant'
        else:
            prediction = 'Breast Cancer is Benign'
        return render_template('prediction.html', prediction=prediction)
    return render_template('breast_cancer.html')

@app.route('/medical_insurance',methods=['GET', 'POST'])
def medical_insurance():
    if request.method == 'POST':
        age = request.form.get('age')
        sex = request.form.get('sex')
        bmi = request.form.get('bmi')
        children = request.form.get('children')
        smoker = request.form.get('smoker')
        region = request.form.get('region')
        trained_model = joblib.load('notebooks/models/medical_insurance.pkl')
        # change the input data to a numpy array
        input_data_as_numpy_array = np.array([[age, sex, bmi, children, smoker, region]])
        # reshape the numpy array as we are predicting for one datapoint
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        value = trained_model.predict(input_data_reshaped)
        prediction = 'The insurance cost is USD ', value[0]
        return render_template('prediction.html', prediction=prediction)
    return render_template('medical_insurance.html')

@app.route('/sonar',methods=['GET', 'POST'])
def sonar():
    if request.method == 'POST':
        f1 = request.form.get('f1')
        f2 = request.form.get('f2')
        f3 = request.form.get('f3')
        f4 = request.form.get('f4')
        f5 = request.form.get('f5')
        f6 = request.form.get('f6')
        f7 = request.form.get('f7')
        f8 = request.form.get('f8')
        f9 = request.form.get('f9')
        f10 = request.form.get('f10')
        f11 = request.form.get('f11')
        f12 = request.form.get('f12')
        f13 = request.form.get('f13')
        f14 = request.form.get('f14')
        f15 = request.form.get('f15')
        f16 = request.form.get('f16')
        f17 = request.form.get('f17')
        f18 = request.form.get('f18')
        f19 = request.form.get('f19')
        f20 = request.form.get('f20')
        f21 = request.form.get('f21')
        f22 = request.form.get('f22')
        f23 = request.form.get('f23')
        f24 = request.form.get('f24')
        f25 = request.form.get('f25')
        f26 = request.form.get('f26')
        f27 = request.form.get('f27')
        f28 = request.form.get('f28')
        f29 = request.form.get('f29')
        f30 = request.form.get('f30')
        f31 = request.form.get('f31')
        f32 = request.form.get('f32')
        f33 = request.form.get('f33')
        f34 = request.form.get('f34')
        f35 = request.form.get('f35')
        f36 = request.form.get('f36')
        f37 = request.form.get('f37')
        f38 = request.form.get('f38')
        f39 = request.form.get('f39')
        f40 = request.form.get('f40')
        f41 = request.form.get('f41')
        f42 = request.form.get('f42')
        f43 = request.form.get('f43')
        f44 = request.form.get('f44')
        f45 = request.form.get('f45')
        f46 = request.form.get('f46')
        f47 = request.form.get('f47')
        f48 = request.form.get('f48')
        f49 = request.form.get('f49')
        f50 = request.form.get('f50')
        f51 = request.form.get('f51')
        f52 = request.form.get('f52')
        f53 = request.form.get('f53')
        f54 = request.form.get('f54')
        f55 = request.form.get('f55')
        f56 = request.form.get('f56')
        f57 = request.form.get('f57')
        f58 = request.form.get('f58')
        f59 = request.form.get('f59')
        f60 = request.form.get('f60')
        input_data_as_numpy_array = np.asarray([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, f31,f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45, f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60]])
        # reshape the np array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        trained_model = joblib.load('notebooks/models/sonar.pkl')
        prediction = trained_model.predict(input_data_reshaped)
        if (prediction[0] == 'R'):
            prediction = 'The object is a Rock'
        else:
            prediction = 'The object is a Mine'
        return render_template('prediction.html', prediction=prediction)
    return render_template('sonar.html')

@app.route('/big_mart',methods=['GET', 'POST'])
def big_mart():
    if request.method == 'POST':
        item_identifier = request.form.get('item_identifier')
        item_weight = request.form.get('item_weight')
        item_fat_content = request.form.get('item_fat_content')
        item_visibility = request.form.get('item_visibility')
        item_type = request.form.get('item_type')
        item_mrp = request.form.get('item_mrp')
        outlet_identifier = request.form.get('outlet_identifier')
        outlet_establishment_year = request.form.get('outlet_establishment_year')
        outlet_size = request.form.get('outlet_size')
        outlet_location_type = request.form.get('outlet_location_type')
        outlet_type = request.form.get('outlet_type')
        trained_model = joblib.load('notebooks/models/big_mart.pkl')
        #the outlet identifier must have options to choose from
        # change the input data to a numpy array
        input_data_as_numpy_array = np.array([[item_identifier, item_weight, item_fat_content, item_visibility, item_type, item_mrp, outlet_identifier, outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])
        # reshape the numpy array as we are predicting for one datapoint
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        value = trained_model.predict(input_data_reshaped)
        prediction = 'The item outlet sales is USD ', value[0]
        return render_template('prediction.html', prediction=prediction)
    return render_template('big_mart.html')

if __name__ == '__main__':
    app.run(debug=True)