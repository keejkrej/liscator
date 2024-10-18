# Flask Application Installation Guide

This guide provides steps to set up and run the Flask application on a remote server.

## Prerequisites

- SSH access to the remote server
- Python 3.10 or higher installed on the remote server

## Installation Steps

1. SSH into the remote server:

   ```
   ssh username@server_address
   ```

   Example:

   ```
   ssh j.mercado@lsr-cl-nv01.hpc.physik.uni-muenchen.de
   ```

2. Create a new directory for the project and navigate to it:

   ```
   mkdir flask_project
   cd flask_project
   ```

3. Create a virtual environment:

   ```
   python3 -m venv fsk_py
   ```

4. Activate the virtual environment:

   ```
   source fsk_py/bin/activate
   ```

5. Upgrade pip:

   ```
   pip install --upgrade pip
   ```

6. Clone the project repository (if applicable) or copy your project files to this directory:

   ```
   git clone [repository_url]
   ```

7. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. To run the Flask application:

   ```
   flask run --host 0.0.0.0 --port 8000
   ```

2. To access the application from your local machine, use SSH port forwarding:

   ```
   ssh -L 8000:localhost:8000 username@server_address
   ```

   Example:

   ```
   ssh -L 8000:localhost:8000 j.mercado@lsr-cl-nv01.hpc.physik.uni-muenchen.de
   ```

3. Open a web browser on your local machine and go to:

   ```
   http://localhost:8000
   ```

## Troubleshooting

- If you encounter permission issues, ensure you have the necessary rights to create directories and files on the server.
- If the port 8000 is already in use, you can choose a different port number.
- Make sure your server's firewall allows connections on the chosen port.

## Additional Notes

- Remember to keep your `requirements.txt` file updated if you add or remove packages.
- For production deployment, consider using a production WSGI server like Gunicorn.