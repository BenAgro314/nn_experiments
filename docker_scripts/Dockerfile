FROM ubuntu:18.04

RUN apt-get -y update
RUN apt-get install -y wget unzip git g++
RUN apt-get -y install libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libopenmpi-dev libglew-dev libglfw3 libglfw3-dev

# for display through vnc
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y xfce4 tigervnc-standalone-server

RUN apt-get update
RUN apt-get -y install curl
RUN apt-get -y install python3-venv
RUN apt-get -y install python3-pip
RUN apt-get -y install ipython

# drake
RUN apt-get update
RUN curl -o drake.tar.gz https://drake-packages.csail.mit.edu/drake/nightly/drake-latest-bionic.tar.gz

# python venv setup with drake
RUN mkdir -p /python_venvs/py3
RUN tar -xvzf drake.tar.gz -C /python_venvs/py3 --strip-components=1
RUN yes | /python_venvs/py3/share/drake/setup/install_prereqs
WORKDIR /python_venvs/
RUN python3 -m venv py3 --system-site-packages
WORKDIR /python_venvs/py3

RUN apt-get update
RUN apt-get -y install firefox

# utilities 
RUN apt-get update
RUN apt-get -y install firefox
RUN apt-get -y install vim
RUN apt-get -y install tmux
RUN apt-get -y install openssh-server
RUN apt-get -y install sudo
RUN apt-get -y install xfce4-terminal
RUN apt-get -y install gedit
RUN apt-get -y install trash-cli
RUN apt-get -y install ffmpeg
RUN apt-get -y install xdg-utils

RUN printf "2\n" | update-alternatives --config x-terminal-emulator

ARG password
ARG id
ARG user

ENV PASS=$password

RUN groupadd -g 1004 $user
RUN useradd -s /bin/bash -u $id -g $id -m $user && echo "$user:$password" | chpasswd && adduser $user sudo
RUN chown -R $user /python_venvs 
USER $user

WORKDIR /home/$user/

# No conda with drake
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
#    mkdir ~/.conda && \
#    /bin/bash miniconda.sh -b -p ~/miniconda && \
#    rm miniconda.sh && \
#    echo "export PATH=~/miniconda/bin:$PATH" >> ~/.bashrc 
#
#SHELL ["/bin/bash", "--login", "-c"]

#ENV PATH=~/miniconda/bin:$PATH

#COPY env.yml env.yml
#RUN conda env create -f env.yml

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
ENV PYTHONPATH=${PYTHONPATH}:/home/$user/nn_experiments/
ENV PATH="/python_venvs/py3/bin:$PATH"
RUN echo "source /python_venvs/py3/bin/activate" >> ~/.bashrc

RUN echo "export USER=$user" >> ~/.bashrc

WORKDIR /home/$user/nn_experiments

COPY .tmux.conf /home/$user/.tmux.conf
COPY .vimrc /home/$user/.vimrc

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["bash","/entrypoint.sh"]

CMD ["bash"]

