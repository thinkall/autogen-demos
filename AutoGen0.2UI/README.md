# Microsoft AutoGen: Multi-Round Human Interaction Chatbot Demo

This demo shows how to build a chatbot which can handle multi-round conversations with human interactions.

## Run app
```
# Install dependencies
pip install -U -r requirements.txt

# Launch app
python app.py
```

## Run docker locally
```
docker build -t autogen/groupchat .
docker run -it autogen/groupchat -p 7860:7860
```

#### [AutoGen](https://github.com/microsoft/autogen) [SourceCode](https://github.com/thinkall/autogen-demos)

![](autogen-human-input.gif)
