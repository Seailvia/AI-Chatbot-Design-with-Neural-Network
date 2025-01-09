# AI-Chatbot-Design-with-Neural-Network
## Abstract
A Chatbot with UI design is created, according to some certain datasets (can be replaced). Through statistical analysis and PINN model, it can answer many kinds of questions. The structure of the chatbot can be used as a source of reference.

## Structure
The construction of our chatbot exemplifies the integration of back-end and front-end technologies to achieve a complete structure. Flask, HTML, and JavaScript each contribute distinct functionalities that, when combined, create an efficient chatbot system.

<div align=center>
<img src="https://github.com/Seailvia/AI-Chatbot-Design-with-Neural-Network/blob/main/Images%20for%20Readme/chatbot.png" width="240" height="240">
</div>

Flask serves as the backbone of the chatbot, managing server-side logic and facilitating communication between the user interface and the chatbot’s natural language processing (NLP) model or pre-defined response logic. We can use Flask package in Python. It can interface with NLP libraries (here we use NLTK) to generate intelligent responses. Python scripts running under Flask preprocess input, pass it to the NLP model, and handle post-processing of the output. Flask uses Jinja2 templating to generate HTML pages dynamically. It can render the chat interface and provide contextual responses in real-time, so we can link our functions to Flask, in order to answer users' questions based on the analysis of certain data.

The front-end of the chatbot was constructed by HTML and Javascript. HTML provides the structural foundation for the chatbot’s user interface. It defines the layout and elements that users interact with, such as input fields, buttons, and the message display area.

JavaScript brings interactivity to the chatbot, allowing real-time exchanges between users and the server. By leveraging client-side programming, JavaScript enhances the user experience and ensures seamless communication without frequent page reloads. It dynamically updates the chat interface by appending user messages and bot responses to the conversation area, listens for user actions such as button clicks or the pressing of the "Enter" key to trigger chat submission.

For example, user only ask the question, that 'Tell me the prediction for future 5 days after the appearance of the vaccine.', for our chatbot. Since our Epidemic PINNs model has been trained and plugged into our chatbot, the chatbot can directly call the trained Epidemic PINNs to give specific predictions of epidemic information.

<div align=center>
<img src="https://github.com/Seailvia/AI-Chatbot-Design-with-Neural-Network/blob/main/Images%20for%20Readme/UI.png" width="240" height="210">
</div>
