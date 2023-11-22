FROM python:3.10.13-slim-bookworm

# Add user
RUN adduser --disabled-password --gecos '' autogen
RUN adduser autogen sudo

# Setup working directory
WORKDIR /home/autogen
COPY . /home/autogen/

# Setup permissions
RUN chown -R autogen:autogen /home/autogen
RUN chmod -R 755 /home/autogen

# Setup user to not run as root
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER autogen

# Install app requirements
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -U pip && pip3 install --no-cache-dir -r requirements.txt
ENV PATH="${PATH}:/home/autogen/.local/bin"

EXPOSE 5006
ENTRYPOINT ["panel", "serve", "app.py"]
