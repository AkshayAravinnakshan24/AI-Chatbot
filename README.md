# AI-Chatbot
Chatbot using Bag-of-words and UI development 


# GPT2-Stance-Analysis
## File Description
1. The full code used for model training is saved in mytrain.py. 
2. Intents.json is the custom made dataset file. It consists of different intent on which the BOT is trained on. It also includes input sentences and their corresponding responses
3.  Chat.py: consists of the code to run the chatbot in Terminal window.
4.  ChatforUI.py consists of the code to send the chatbot responses to app.py.
5.  app.py includes the code for chatbot GUI.  
6.  Model.py: Model definition is done in this py file. Consists of all the different layers of the  Nueral network model that is used for training.
7.  data.pth: The model parameters after training is stored in this file. 
## 1. Usage
#Chatbot in GUI:
1. Download the repository 
2. Run the chatforUI.py file to get the GUI of the bot.

#Chatbot in Terminal:
1. Download the repository 
2. Run the app.py file to get the GUI of the bot.

## 2. Dataset
The chatbot can be easily customized to perform tasks related to the required intent.
Just change the intents.json file, and train the model for the required intent
The  Keys in the intents.json file stand for 
tag: with the intent name,
patterns: example input sentences 
response: How the bot should respond to the sentences
```json
{
      "tag": "goodbye",  
      "patterns": [ "Bye", "See you later", "Goodbye" ],     
      "responses": [
        "See you later, thanks for visiting",
        "Have a nice day",
        "Bye! Come back again soon."
      ]                                                    
    },
```

