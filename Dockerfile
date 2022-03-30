FROM python:3.9
MAINTAINER Andrea Costanzo (andreacos82@gmail.com)

# Install dependencies needed for opencv
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y

# Install dependencies from requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

# Create an user to avoid to lock output masks with root permission only
#RUN useradd --create-home detector
#WORKDIR /home/detector
#RUN mkdir ./app
#COPY . /app
#RUN chown -R detector:detector app
#USER detector

COPY . /app
WORKDIR /app
ENTRYPOINT [ "python3", "run_detector.py" ]