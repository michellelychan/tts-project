mlflow server --host 0.0.0.0 --port 5000
# Use a persistent storage for your MLflow artifacts:
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts

# Or for more scalability, use a database backend:
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql://username:password@localhost/mlflow_db --default-artifact-root ./mlflow-artifacts

# To run the server in the background (so it keeps running even if you close your terminal):
nohup mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts > mlflow.log 2>&1 &


# To access the MLflow UI from your laptop, set up SSH port forwarding:
ssh -L 5000:localhost:5000 username@paperspace-server-ip

# If your Paperspace server has a public IP address and 
# allows incoming connections on port 5000, you could also 
# access the MLflow UI directly at 
# http://paperspace-server-ip:5000