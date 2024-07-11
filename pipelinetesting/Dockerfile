# Use the official Python slim image from Docker Hub
FROM python:3.7.3-slim

# Set the maintainer, version, and description labels
LABEL maintainer="Your Name Here"
LABEL version="0.1"
LABEL description="A mini Data Science environment with Jupyter, Pandas, Matplotlib, Numpy, and more."

# Set the working directory inside the container
WORKDIR /data

# Copy the contents of the current directory to the working directory in the container
COPY . /data

# Install the necessary Python libraries
RUN pip install jupyter numpy matplotlib seaborn pandas

# Expose the port that Jupyter Notebook will run on
EXPOSE 8888

# Set the command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
