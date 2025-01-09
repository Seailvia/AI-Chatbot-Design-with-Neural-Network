# AI-Chatbot-Design-with-Neural-Network
## Abstract
A Chatbot with UI design is created, according to some certain datasets (can be replaced). Through statistical analysis and PINN model, it can answer many kinds of questions. The structure of the chatbot can be used as a source of reference.

## Structure
The construction of our chatbot exemplifies the integration of back-end and front-end technologies to achieve a complete structure. Flask, HTML, and JavaScript each contribute distinct functionalities that, when combined, create an efficient chatbot system.

<div align=center>
<img src="https://github.com/Seailvia/AI-Chatbot-Design-with-Neural-Network/blob/main/Images%20for%20Readme/chatbot.png" width="360" height="360">
</div>

Flask serves as the backbone of the chatbot, managing server-side logic and facilitating communication between the user interface and the chatbot’s natural language processing (NLP) model or pre-defined response logic. We can use Flask package in Python. It can interface with NLP libraries (here we use NLTK) to generate intelligent responses. Python scripts running under Flask preprocess input, pass it to the NLP model, and handle post-processing of the output. Flask uses Jinja2 templating to generate HTML pages dynamically. It can render the chat interface and provide contextual responses in real-time, so we can link our functions to Flask, in order to answer users' questions based on the analysis of certain data.

The front-end of the chatbot was constructed by HTML and Javascript. HTML provides the structural foundation for the chatbot’s user interface. It defines the layout and elements that users interact with, such as input fields, buttons, and the message display area.

JavaScript brings interactivity to the chatbot, allowing real-time exchanges between users and the server. By leveraging client-side programming, JavaScript enhances the user experience and ensures seamless communication without frequent page reloads. It dynamically updates the chat interface by appending user messages and bot responses to the conversation area, listens for user actions such as button clicks or the pressing of the "Enter" key to trigger chat submission.

For example, user only ask the question, that 'Tell me the prediction for future 5 days after the appearance of the vaccine.', for our chatbot. Since our Epidemic PINNs model has been trained and plugged into our chatbot, the chatbot can directly call the trained Epidemic PINNs to give specific predictions of epidemic information.

<div align=center>
<img src="https://github.com/Seailvia/AI-Chatbot-Design-with-Neural-Network/blob/main/Images%20for%20Readme/UI.png" width="400" height="350">
</div>

## User Guide
### 1. Access the Chatbot
Use the provided URL link to access the Chatbot interface.

### 2. Enter Your Query
In the input box on the Chatbot interface, type your query. Example queries include:

"Query the death toll on October 1, 2021"

"Calculate the total number of confirmed cases in October 2021"

"Analyze and evaluate the overall pandemic situation in October 2021"

"Generate charts such as bar graphs, line graphs, or mosaic plots for the number of confirmed cases in October 2021"

"Predict the trend of infected cases over the next 5 days"


### 3. Submit Your Query
Once you’ve entered your query, click the Submit button on the right side of the input box to send your query to the Chatbot for processing.

### 4. View Results
After processing your query, the Chatbot will display the results on the interface. These results may include:

Specific numerical data from your query

Analytical and evaluative text descriptions

Visualized charts

Predictions of future trends along with explanatory details

## Frequently Asked Questions (FAQ)
### How do I know if my query was processed correctly?

After submitting a query, the Chatbot will display a processing status. If an error message appears, check whether your input format is correct.

### Does the Chatbot support multiple chart types?

Yes, the system supports various chart types, such as bar graphs, line graphs, and mosaic plots. Users can choose the type that best suits their needs.

### How reliable are the prediction results?

The Chatbot uses historical data and prediction models to make forecasts. However, the results are for reference only, as real-world outcomes may be influenced by various factors.

