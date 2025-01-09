# AI-Chatbot-Design-with-Neural-Network
A Chatbot with UI design is created, according to some certain datasets (can be replaced). Through statistical analysis and PINN model, it can answer many kinds of questions. The structure of the chatbot can be used as a source of reference.

The construction of our chatbot exemplifies the integration of back-end and front-end technologies to achieve a complete structure. Flask, HTML, and JavaScript each contribute distinct functionalities that, when combined, create an efficient chatbot system.

Flask serves as the backbone of the chatbot, managing server-side logic and facilitating communication between the user interface and the chatbot’s natural language processing (NLP) model or pre-defined response logic. We can use Flask package in Python.

The front-end of the chatbot was constructed by HTML and Javascript. HTML provides the structural foundation for the chatbot’s user interface. It defines the layout and elements that users interact with, such as input fields, buttons, and the message display area.

For example, user only ask the question, that 'Tell me the prediction for future 5 days after the appearance of the vaccine.', for our chatbot. Since our Epidemic PINNs model has been trained and plugged into our chatbot, the chatbot can directly call the trained Epidemic PINNs to give specific predictions of epidemic information.
