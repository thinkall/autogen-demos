FROM python:3.10.13-slim-bookworm

# Setup user to not run as root
RUN adduser --disabled-password --gecos '' autogen
RUN adduser autogen sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER autogen

# Setup working directory
WORKDIR /home/autogen
COPY . /home/autogen/

# Install app requirements
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -U pip && pip3 install --no-cache-dir -r requirements.txt
ENV PATH="${PATH}:/home/autogen/.local/bin"

EXPOSE 7860
ENTRYPOINT ["python3", "app.py"]
