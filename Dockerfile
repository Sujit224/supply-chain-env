FROM python:3.11-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUTF8="1"

# Switch to the "user" user
USER user

# Set the working directory
WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user server/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . $HOME/app

# Hugging Face Spaces routes traffic to port 8000 by default
EXPOSE 8000

# Run uvicorn on port 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
