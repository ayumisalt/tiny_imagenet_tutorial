FROM tensorflow/tensorflow:2.1.0-gpu-py3

# Set working directory
WORKDIR /workspace

ENV PYTHONPATH "${PYTHONPATH}:/workspace"

COPY requirements_apt.txt .
# Avoiding errors caused by tzdata
ENV DEBIAN_FRONTEND=noninteractive

# https://tech-blog.optim.co.jp/entry/2022/05/18/100000
RUN rm -f /etc/apt/sources.list.d/cuda.list \
 && apt-get update && apt-get install -y --no-install-recommends \
    wget \
 && distro=$(. /usr/lib/os-release; echo $ID$VERSION_ID | tr -d ".") \
 && arch=$(/usr/bin/arch) \
 && wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb \
 && dpkg -i cuda-keyring_1.0-1_all.deb \
 && rm -f cuda-keyring_1.0-1_all.deb

RUN apt-get update
RUN cat requirements_apt.txt | xargs apt-get install -y \
      && apt-get clean \
            && rm -rf /var/lib/apt/lists/*

# Necessary pip packages
RUN pip install --upgrade pip --user

COPY requirements_pip.txt .
RUN pip install -r requirements_pip.txt

# run zsh
COPY zshrc /root/.zshrc
RUN zsh
ENV SHELL /usr/bin/zsh
RUN sed -i.bak "s|$HOME:|$HOME:$SHELL|" /etc/passwd
CMD ["/usr/bin/zsh"]

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


COPY . .
