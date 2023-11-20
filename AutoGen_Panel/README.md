# Microsoft AutoGen: Play Ground

This demo is an AutoGen playground implemented with [Panel](https://panel.holoviz.org/index.html).

## Run app
```
# Install dependencies
pip install -U -r requirements.txt

# Launch app
bash run.sh
```

## Run docker locally
```
docker build -t autogen/groupchat .
docker run -it autogen/groupchat -p 7860:7860
```

#### [AutoGen](https://github.com/microsoft/autogen) [SourceCode](https://github.com/thinkall/autogen-demos)
