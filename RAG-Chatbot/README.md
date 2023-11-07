---
title: Autogen Demos
emoji: 🌖
colorFrom: pink
colorTo: blue
sdk: gradio
sdk_version: 3.47.1
app_file: app.py
pinned: false
license: mit
---

# Microsoft AutoGen: Retrieve Chat Demo

This demo shows how to use the RetrieveUserProxyAgent and RetrieveAssistantAgent to build a chatbot.

## Run app
```
# Install dependencies
pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip3 install --no-cache-dir -r requirements.txt

# Launch app
python app.py
```

## Run docker locally
```
docker build -t autogen/rag .
docker run -it autogen/rag -p 7860:7860
```

#### [GitHub](https://github.com/microsoft/autogen) [Discord](https://discord.gg/pAbnFJrkgZ) [Blog](https://microsoft.github.io/autogen/blog/2023/10/18/RetrieveChat) [Paper](https://arxiv.org/abs/2308.08155) [SourceCode](https://github.com/thinkall/autogen-demos) [OnlineApp](https://huggingface.co/spaces/thinkall/autogen-demos)

- Watch the demo video
  
[![Watch the video](https://img.youtube.com/vi/R3cB4V7dl70/hqdefault.jpg)](https://www.youtube.com/embed/R3cB4V7dl70)

![](autogen-rag.gif)