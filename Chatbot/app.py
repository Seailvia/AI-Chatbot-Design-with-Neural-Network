from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import random
from torch.optim import LBFGS, Adam
from tqdm import tqdm

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


# Load the Excel file containing Victoria COVID data
data = pd.read_excel('victoria_table.xlsx')

# Convert column names to lowercase to allow flexible matching
data.columns = data.columns.str.lower()

# Define separate mappings for cumulative and non-cumulative columns
cumulative_column_mapping = {
    'confirmed': 'confirmed_cum',
    'deaths': 'deaths_cum(dt)',
    'people died': 'deaths_cum(dt)',
    'tests': 'tests_cum',
    'positives': 'positives_cum',
    'positive cases': 'positives_cum',
    'positive': 'positives_cum',
    'recovered': 'recovered_cum(rt)',
    'vaccines': 'vaccines_cum(vt)',
    'affected people': 'positives_cum'
}

non_cumulative_column_mapping = {
    'confirmed': 'confirmed',
    'new cases': 'confirmed',
    'deaths': 'deaths',
    'new deaths': 'deaths',
    'people died': 'deaths',
    'tests': 'tests(et)',
    'number of tests': 'tests(et)',
    'positives': 'positives',
    'positive cases': 'positives',
    'positive': 'positives',
    'recovered': 'recovered',
    'people recovered': 'recovered',
    'vaccines': 'vaccines'
}

column_name_mapping = {
    'confirmed_cum': 'confirmed cases',
    'deaths_cum(dt)': 'deaths',
    'tests_cum': 'tests conducted',
    'positives_cum': 'positive cases',
    'recovered_cum(rt)': 'recovered people',
    'vaccines_cum(vt)': 'people vaccinated',
    'confirmed': 'confirmed cases',
    'deaths': 'deaths',
    'tests(et)': 'tests conducted',
    'positives': 'positive cases',
    'recovered': 'recovered people',
    'vaccines': 'vaccines injected'
}

# Stopwords for filtering out common irrelevant words
stop_words = set(stopwords.words('english'))
stop_words.update(['number', 'of'])  # Add 'number' and 'of' to stopwords to remove them from user queries

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    # Function to preprocess user input
    def preprocess_input(user_input):
        user_input = user_input.lower()  # Convert to lowercase
        user_input = user_input.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        words = word_tokenize(user_input)  # Tokenize the input
        filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
        return filtered_words

# Function to get the relevant value based on user input
    def get_value(user_input):
        # Preprocess the user input
        user_input_words = preprocess_input(user_input)
    
        # Extract date information from the user input
        date_match = re.search(r'\d{4}/\d{1,2}/\d{1,2}', user_input)
        if not date_match:
            return 'sorry, no record'
        date = date_match.group()

        # Extract column name information based on whether cumulative is mentioned
        column_name = None
        is_cumulative = 'cumulative' in user_input or 'up to' in user_input or 'cumulative number of' in user_input or 'until' in user_input
    
        if is_cumulative:
            for key, col in cumulative_column_mapping.items():
                if key in user_input:
                    column_name = col
                    break
        else:
            for key, col in non_cumulative_column_mapping.items():
                if key in user_input:
                    column_name = col
                    break

        # If column name is not found, return an error message
        if not column_name or column_name.lower() not in data.columns:
            return 'sorry, no record'

        # Filter the data by the given date
        row = data[data['date'] == date]
        if not row.empty:
            # Return the value for the matching column and date
            value = row.iloc[0][column_name]
            if pd.isna(value):
                return 'sorry, no record'
            # Use key from the appropriate mapping to improve readability in output
            readable_column_name = None
            for a,b in column_name_mapping.items():
                if a in column_name:
                    readable_column_name = b
            if is_cumulative:
                return f'The cumulative number of {readable_column_name} until {date} is {value}.'
            else:
                return f'The number of {readable_column_name} on {date} is {value}.'
        else:
            return 'sorry, no record'
    
    def get_visual(user_input):
        """
        Plot a boxplot for a specified parameter in a given year and month based on user input.

        Parameters:
        - user_input: string, a user query in the format (e.g., "Show me the situation for 2021/5, confirmed").

        Returns:
        - None
        """
        data = pd.read_excel('victoria_table.xlsx')
        data['date'] = pd.to_datetime(data['date'])

        date_pattern = r'(\d{4})/(\d{2})'
        parameter_pattern = r'\s([a-zA-Z_]+)$'

        date_match = re.search(date_pattern, user_input)
        if not date_match:
            print("Invalid input format. Please include a date in YYYY/MM format.")
            return
        year, month = map(int, date_match.groups())

        parameter_match = re.search(parameter_pattern, user_input)
        if not parameter_match:
            print("Invalid input format. Please include a parameter name (e.g., 'confirmed').")
            return
        parameter = parameter_match.group(1)

        filtered_data = data[(data['date'].dt.year == year) & (data['date'].dt.month == month)]

        if filtered_data.empty:
            print(f"No data available for {year}/{month}.")
            return
        
        sns.set(style="whitegrid")

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=filtered_data[parameter], color='skyblue', fliersize=5, linewidth=1.5)
        plt.title(f'Boxplot of {parameter} for {year}-{month}', fontsize=16)
        plt.xlabel(parameter, fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()



    def get_statistics(user_input):
        data = pd.read_excel('victoria_table.xlsx')
        data['date'] = pd.to_datetime(data['date'])
        
        column_mapping = {'confirmed': 'confirmed', 'deaths': 'deaths', 'positives': 'positives', 'recovered': 'recovered', 'vaccines': 'vaccines'}
        
        date_pattern = r'(\d{4})/(\d{2})'
        stat_pattern = r'\b(mean|variance|standard deviation|median|mode|min|max)\b'
        param_pattern = r'\b(confirmed|deaths|positives|recovered|vaccines)\b'
        
        date_match = re.search(date_pattern, user_input)
        stat_match = re.search(stat_pattern, user_input, re.IGNORECASE)
        param_match = re.search(param_pattern, user_input.lower())
        
        if not date_match:
            return "Invalid input format. Please include a date in YYYY/MM format."
        
        if not stat_match:
            return "Invalid input format. Please include a valid statistic (e.g., mean, variance)."
        
        year, month = map(int, date_match.groups())
        statistic = stat_match.group(0).lower()
        
        if param_match:
            parameter = param_match.group(0)
        else:
            parameter = next((col for col in column_mapping.keys() if col in data.columns), None)
        
        if not parameter or parameter not in data.columns:
            return f"Parameter '{parameter}' not found in data."
        
        filtered_data = data[(data['date'].dt.year == year) & (data['date'].dt.month == month)]
        
        if filtered_data.empty:
            return f"No data available for {year}/{month}."
        
        stat_map = {
            'mean': 'mean',
            'variance': 'var',
            'standard deviation': 'std',
            'median': 'median',
            'mode': 'mode',
            'min': 'min',
            'max': 'max'
        }
        
        if statistic in stat_map:
            if statistic == 'mode':
                result = filtered_data[parameter].mode()
                return f"The mode of {parameter} for {year}-{month} is: {result.iloc[0] if not result.empty else 'No mode found.'}"
            else:
                result = getattr(filtered_data[parameter], stat_map[statistic])()
                return f"The {statistic} of {parameter} for {year}-{month} is: {result}"
        else:
            return f"Statistic '{statistic}' not supported."

    def get_general(user_input):
        """
        Retrieve a specific statistic for a given parameter in a specified year and month, based on user input.

        Parameters:
        - user_input: string, a user query in the format (e.g., "Show me the situation for 2021/5").

        Returns:
        - result: string, judgment about the situation for the specified year and month.
        """
        data = pd.read_excel('victoria_table.xlsx')
        data['date'] = pd.to_datetime(data['date'])

        date_pattern = r'(\d{4})/(\d{2})'
        date_match = re.search(date_pattern, user_input)

        if not date_match:
            return "Invalid input format. Please include a date in YYYY/MM format."
        
        year, month = map(int, date_match.groups())

        filtered_data = data[(data['date'].dt.year == year) & (data['date'].dt.month == month)]

        if filtered_data.empty:
            return f"No data available for {year}/{month}."
        
        stats = filtered_data.describe()

        if filtered_data['confirmed'].mean() > 52320:
            confirmpace = 'rapid growth'
        elif filtered_data['confirmed'].mean() < 218:
            confirmpace = 'slow growth'
        else:
            confirmpace = 'normal growth'

        if filtered_data['deaths'].mean() > 266:
            deathrate = 'high'
        elif filtered_data['deaths'].mean() == 0:
            deathrate = 'low'
        else:
            deathrate = 'medium'

        if filtered_data['recovered'].mean() > 330:
            recoverpace = 'rapid'
        else:
            recoverpace = 'medium'

        if filtered_data['vaccines'].mean() > 36573:
            vaccinerate = 'more people chose to get vaccinated'
        else:
            vaccinerate = 'few people chose to get vaccinated'
        
        if year <= 2021 and month <= 9:
            period = 'Early stage of the epidemic'
        elif year == 2023:
            period = 'Late stage of the epidemic'
        else:
            period = 'Middle stage of the epidemic'

        result = f'The situation in Victoria in {year}-{month} is: {period}. There is a {confirmpace} trend in confirmed cases, the rate of death is {deathrate} during this period and a {recoverpace} trend of recovery appears. With the development of medical technology, vaccination status and changes in people\'s attitudes to vaccination, {vaccinerate}.'
        
        return result

    def get_information(t):
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import matplotlib.pyplot as plt
        import random
        from torch.optim import LBFGS, Adam
        from tqdm import tqdm
        import time
        
        seed = 0
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')    

        step_size = 1e-4
        T = [t]
        T = np.array(T)
        T = torch.tensor(T, dtype=torch.float32, requires_grad=True).to(device)
        T_test = T.reshape(len(T), 1)
        ### Set activation function
        ### We use the activation function  tanh in our model, maybe user can choose to custom activation functions, like following that ASW 
        class WaveAct(nn.Module):
            def __init__(self):
                super(WaveAct, self).__init__() 
                self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
                self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)
                self.a1 = nn.Parameter(torch.ones(1), requires_grad=True)
                self.a2 = nn.Parameter(torch.ones(1), requires_grad=True)

            def forward(self, x):
                return self.w1 * torch.sin(self.a1*x)+ self.w2 * torch.cos(self.a2*x)
            
        ### Set model structure
        class PINN(nn.Module):
            def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
                super(PINN, self).__init__()

                layers = []
                for i in range(num_layer-1):
                    if i == 0:
                        layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                        layers.append(nn.Tanh())
                    else:
                        layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                        layers.append(nn.Tanh())

                layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

                self.linear = nn.Sequential(*layers)

            def forward(self,T):
                src = T
                return self.linear(src)

        ### Model Weights and Bias Initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        ### Model Parameter Statistics
        def get_n_params(model):
            pp=0
            for p in list(model.parameters()):
                nn=1
                for s in list(p.size()):
                    nn = nn*s
                pp += nn
            return pp
        
        model = PINN(in_dim=1, hidden_dim=256, out_dim=6, num_layer=10).to(device)
        model.load_state_dict(torch.load('./pinn_model_parameters.pt'))
        pred = model(T_test)
        pred_list = pred.tolist()
        pred_S = pred_list[0][0]
        pred_V = pred_list[0][1]
        pred_I = pred_list[0][2]
        pred_R = pred_list[0][3]
        pred_E = pred_list[0][4]
        pred_D = pred_list[0][5]

        return f"The rate of susceptible is '{pred_S * 10:.4f}', the rate of being vaccinated is '{pred_V * 10:.4f}', the rate of infected is '{pred_I * 10:.4f}', the rate of predicted recovery rate is '{pred_R * 10:.4f}', exposing rate is '{pred_E * 10:.4f}', death rate is '{pred_D * 10:.4f}'."

####################################################################################
    def classify_input(user_input):
        # Check if the input has two dates (yyyy/mm or similar) and keywords for interval
        date_pattern = r'\b(\d{4}/\d{1,2})\b'
        dates = re.findall(date_pattern, user_input)
        
        # Check if the input contains keywords for visualization
        if re.search(r'\b(boxplot|frequency plot|histogram|chart|graph|table|visualization|plot)\b', user_input, re.IGNORECASE):
            return get_visual(user_input)
        
        if len(dates) == 2:
            if re.search(r'\b(from.*to|between.*and)\b', user_input, re.IGNORECASE):
                return get_trend(user_input)
        
        # Check if the input contains keywords for descriptive statistics
        if re.search(r'\b(mean|variance|standard deviation|median|mode|statistics)\b', user_input, re.IGNORECASE):
            return get_statistics(user_input)
        
        if re.search(r'\b(after|predict|forecast|prediction)\b', user_input, re.IGNORECASE):
            match = re.search(r'\d+', user_input, re.IGNORECASE)
            d = match.group()
            d = int(d)
            return get_information(d)

        # Check if the input contains a specific date (yyyy/mm/dd)
        specific_date_pattern = r'\d{4}/\d{1,2}/\d{1,2}'
        if re.search(specific_date_pattern, user_input):
            return get_value(user_input)
        
        # Check if the input has one date (yyyy/mm)
        if len(dates) == 1:
            return get_general(user_input)   
        
        # Default case
        return 'sorry, no record, can you ask in another way?'

    user_message = request.json.get('message')
    
    if "hello" in user_message.lower():
        response = "Hi there! How can I help you today?"
    elif "bye" in user_message.lower():
        response = "Goodbye! Have a great day!"
    else:
        response = f"{classify_input(user_message)}\n"

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)